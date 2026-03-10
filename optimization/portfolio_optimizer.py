"""
optimization/portfolio_optimizer.py
─────────────────────────────────────
Core portfolio optimization engine.

Supported strategies
────────────────────
  'mvo'          — Markowitz Mean-Variance Optimization
  'min_vol'      — Global Minimum Variance
  'max_sharpe'   — Tangency portfolio (maximize Sharpe ratio)
  'risk_parity'  — Equal Risk Contribution
  'cvar'         — CVaR (Expected Shortfall) minimization
  'equal_weight' — 1/N benchmark

Constraints implemented
───────────────────────
  • Long-only  (w_i ≥ 0)
  • Max weight  (w_i ≤ max_weight)
  • Min weight  (w_i ≥ min_weight)
  • Fully invested  (Σ w_i = 1)
  • Portfolio vol cap
  • Max turnover
  • Sector exposure cap (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CVXPY = False
try:
    import cvxpy as cp
    _CVXPY = True
except ImportError:
    pass

from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# Constraint / Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    strategy:        str   = "mvo"
    risk_free_rate:  float = 0.04 / 12    # monthly
    max_weight:      float = 0.35
    min_weight:      float = 0.0
    vol_target:      float = 0.10 / np.sqrt(12)  # monthly
    cvar_alpha:      float = 0.05
    n_scenarios:     int   = 1000
    max_turnover:    float = 0.50
    risk_aversion:   float = 1.0          # λ in MVO objective
    sector_limits:   dict  = field(default_factory=dict)  # {sector: max_weight}


# ─────────────────────────────────────────────────────────────────────────────
# Main Optimizer Class
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Computes optimal portfolio weights given expected returns and a covariance
    matrix, subject to user-defined constraints.

    Parameters
    ----------
    assets  : list of asset tickers
    config  : OptimizationConfig instance
    """

    def __init__(self, assets: list[str], config: OptimizationConfig | None = None):
        self.assets = assets
        self.n = len(assets)
        self.config = config or OptimizationConfig()

    # ── Main entry point ──────────────────────────────────────────────────────

    def optimize(
        self,
        expected_returns: pd.Series | np.ndarray,
        cov_matrix: pd.DataFrame | np.ndarray,
        current_weights: np.ndarray | None = None,
        scenario_returns: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute optimal weights.

        Parameters
        ----------
        expected_returns  : (n,) monthly expected return per asset
        cov_matrix        : (n, n) monthly covariance matrix
        current_weights   : (n,) current portfolio weights (for turnover)
        scenario_returns  : (S, n) return scenarios (required for CVaR)

        Returns
        -------
        weights : np.ndarray (n,)
        """
        mu  = np.array(expected_returns).flatten()
        cov = np.array(cov_matrix)

        if current_weights is None:
            current_weights = np.ones(self.n) / self.n

        strategy = self.config.strategy

        if strategy == "equal_weight":
            return np.ones(self.n) / self.n

        if strategy == "risk_parity":
            from risk.covariance_estimator import CovarianceEstimator
            est = CovarianceEstimator()
            est._cov = pd.DataFrame(cov, index=self.assets, columns=self.assets)
            return est.risk_parity_weights()

        if strategy == "cvar":
            if scenario_returns is None:
                rng = np.random.default_rng(42)
                L = np.linalg.cholesky(cov + 1e-7 * np.eye(self.n))
                z = rng.standard_normal((self.config.n_scenarios, self.n))
                scenario_returns = mu + z @ L.T
            if _CVXPY:
                return self._cvar_optimize(mu, cov, scenario_returns, current_weights)
            else:
                return self._scipy_cvar_optimize(mu, cov, scenario_returns, current_weights)

        if _CVXPY:
            return self._cvxpy_optimize(mu, cov, current_weights, strategy)
        else:
            return self._scipy_optimize(mu, cov, current_weights, strategy)

    # ── CVaR Optimization (CVXPY) ─────────────────────────────────────────────

    def _cvar_optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        scenarios: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Minimize CVaR at level alpha using the Rockafellar-Uryasev formulation."""
        S = scenarios.shape[0]
        alpha = self.config.cvar_alpha

        w   = cp.Variable(self.n)
        eta = cp.Variable()              # VaR threshold
        z   = cp.Variable(S)            # excess losses

        port_losses = -scenarios @ w     # (S,)

        objective = cp.Minimize(eta + 1 / (alpha * S) * cp.sum(z))
        constraints = [
            z >= port_losses - eta,
            z >= 0,
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
            cp.sum(cp.abs(w - current_weights)) <= self.config.max_turnover,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            weights = np.array(w.value).flatten()
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum()
            return weights
        else:
            logger.warning("CVaR optimisation failed (%s); returning equal weights.", prob.status)
            return np.ones(self.n) / self.n

    # ── CVXPY MVO / Min-Vol / Max-Sharpe ─────────────────────────────────────

    def _cvxpy_optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        current_weights: np.ndarray,
        strategy: str,
    ) -> np.ndarray:
        w = cp.Variable(self.n)
        port_var = cp.quad_form(w, cov)

        # ── Objective ─────────────────────────────────────────────────────────
        if strategy == "min_vol":
            objective = cp.Minimize(port_var)
        elif strategy == "max_sharpe":
            # Maximize (μ^T w - rf) / σ  via Lagrangian trick (y = w/κ)
            y = cp.Variable(self.n)
            k = cp.Variable(nonneg=True)
            rf = self.config.risk_free_rate
            obj = cp.Minimize(cp.quad_form(y, cov))
            cons = [
                (mu - rf) @ y == 1,
                cp.sum(y) == k,
                y >= self.config.min_weight * k,
                y <= self.config.max_weight * k,
                k >= 0,
            ]
            prob = cp.Problem(obj, cons)
            prob.solve(verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None and k.value and k.value > 1e-10:
                w_val = (y.value / k.value).flatten()
                w_val = np.clip(w_val, 0, 1)
                return w_val / w_val.sum()
            return np.ones(self.n) / self.n
        else:  # mvo
            lam = self.config.risk_aversion
            objective = cp.Minimize(lam * port_var - mu @ w)

        # ── Constraints ───────────────────────────────────────────────────────
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
            cp.sum(cp.abs(w - current_weights)) <= self.config.max_turnover,
            port_var <= self.config.vol_target ** 2,
        ]

        # Optional sector constraints
        if self.config.sector_limits:
            for sector, limit in self.config.sector_limits.items():
                indices = [
                    i for i, a in enumerate(self.assets)
                    if sector in a  # simplified: match by substring
                ]
                if indices:
                    constraints.append(cp.sum(w[indices]) <= limit)

        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            weights = np.clip(np.array(w.value).flatten(), 0, 1)
            weights /= weights.sum()
            return weights
        else:
            logger.warning(
                "CVXPY optimisation failed (%s); falling back to min-vol via scipy.",
                prob.status,
            )
            return self._scipy_optimize(mu, cov, current_weights, "min_vol")

    # ── SciPy fallback ────────────────────────────────────────────────────────

    def _scipy_optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        current_weights: np.ndarray,
        strategy: str,
    ) -> np.ndarray:
        n = self.n
        rf = self.config.risk_free_rate

        def port_var(w):  return float(w @ cov @ w)
        def port_vol(w):  return float(np.sqrt(port_var(w)))
        def port_ret(w):  return float(mu @ w)
        def neg_sharpe(w): return -(port_ret(w) - rf) / (port_vol(w) + 1e-9)

        obj = {
            "mvo":       lambda w: -port_ret(w) + self.config.risk_aversion * port_var(w),
            "min_vol":   port_var,
            "max_sharpe": neg_sharpe,
        }.get(strategy, port_var)

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: self.config.vol_target - port_vol(w)},
            {"type": "ineq", "fun": lambda w: self.config.max_turnover - np.abs(w - current_weights).sum()},
        ]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        w0 = np.ones(n) / n

        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.success:
            w = np.clip(res.x, 0, 1)
            return w / w.sum()
        logger.warning("Scipy optimisation did not converge; returning equal weights.")
        return w0


    def _scipy_cvar_optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        scenarios: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """CVaR minimization via scipy (no cvxpy required)."""
        alpha = self.config.cvar_alpha
        n = self.n

        def cvar_obj(w):
            port_rets = scenarios @ w
            var = np.percentile(port_rets, alpha * 100)
            tail = port_rets[port_rets <= var]
            return -tail.mean() if len(tail) > 0 else 0.0

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: self.config.max_turnover - np.abs(w - current_weights).sum()},
        ]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        res = minimize(cvar_obj, np.ones(n)/n, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-8})
        if res.success:
            w = np.clip(res.x, 0, 1)
            return w / w.sum()
        return np.ones(n) / n

    # ── Efficient Frontier ────────────────────────────────────────────────────

    def efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Trace the efficient frontier by sweeping risk-aversion λ.

        Returns a DataFrame with columns: [ret, vol, sharpe, weights...]
        """
        lambdas = np.logspace(-2, 2, n_points)
        records = []
        cfg0 = self.config

        for lam in lambdas:
            self.config = OptimizationConfig(
                strategy="mvo",
                risk_aversion=lam,
                max_weight=cfg0.max_weight,
                min_weight=cfg0.min_weight,
                vol_target=1e6,          # no vol cap for frontier
                max_turnover=2.0,        # no turnover limit
                risk_free_rate=cfg0.risk_free_rate,
            )
            w = self.optimize(mu, cov)
            ret = float(mu @ w) * 12
            vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(12)
            sharpe = (ret - cfg0.risk_free_rate * 12) / (vol + 1e-9)
            row = {"ret": ret, "vol": vol, "sharpe": sharpe}
            row.update({a: w[i] for i, a in enumerate(self.assets)})
            records.append(row)

        self.config = cfg0
        return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys; sys.path.insert(0, "..")
    from config import TICKERS, START_DATE, END_DATE
    from data.data_loader import DataLoader
    from risk.covariance_estimator import CovarianceEstimator

    loader = DataLoader(TICKERS, START_DATE, END_DATE).load()
    monthly = loader.get_monthly_returns()

    est = CovarianceEstimator("ledoit_wolf").fit(monthly)
    mu  = monthly.mean().values
    cov = est.cov.values

    for strat in ("mvo", "min_vol", "max_sharpe", "risk_parity", "cvar"):
        cfg = OptimizationConfig(strategy=strat)
        opt = PortfolioOptimizer(TICKERS, cfg)
        w   = opt.optimize(mu, cov)
        print(f"\n── {strat} ──")
        print(pd.Series(w, index=TICKERS).round(4))
