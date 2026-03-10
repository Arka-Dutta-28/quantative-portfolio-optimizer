"""
risk/covariance_estimator.py
─────────────────────────────
Robust covariance matrix estimation for portfolio risk modelling.

Methods implemented:
  1. Sample Covariance          — standard baseline
  2. Exponentially Weighted     — emphasises recent observations
  3. Ledoit-Wolf Shrinkage      — regularised, best for small-T / large-N
  4. Constant-Correlation       — shrinks toward equicorrelation structure
  5. Oracle Approximating (OAS) — adaptive shrinkage intensity
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

logger = logging.getLogger(__name__)


class CovarianceEstimator:
    """
    Estimates the asset return covariance matrix.

    Parameters
    ----------
    method   : 'sample' | 'ewm' | 'ledoit_wolf' | 'constant_corr' | 'oas'
    halflife : half-life in months for EWM (ignored for other methods)
    """

    METHODS = ("sample", "ewm", "ledoit_wolf", "constant_corr", "oas")

    def __init__(self, method: str = "ledoit_wolf", halflife: int = 12):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method   = method
        self.halflife = halflife
        self._cov: pd.DataFrame | None = None
        self._cor: pd.DataFrame | None = None
        self._vols: pd.Series | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, returns: pd.DataFrame) -> "CovarianceEstimator":
        """
        Estimate covariance from a DataFrame of returns.

        Parameters
        ----------
        returns : pd.DataFrame  shape (T, N)  — monthly or daily returns
        """
        method_fn = {
            "sample":        self._sample_cov,
            "ewm":           self._ewm_cov,
            "ledoit_wolf":   self._ledoit_wolf_cov,
            "constant_corr": self._constant_corr_cov,
            "oas":           self._oas_cov,
        }
        cov_np = method_fn[self.method](returns.values)

        self._cov  = pd.DataFrame(cov_np, index=returns.columns, columns=returns.columns)
        self._vols = pd.Series(np.sqrt(np.diag(cov_np)), index=returns.columns)
        self._cor  = self._cov_to_corr(self._cov)

        logger.debug(
            "Cov (%s): condition number=%.1f  avg vol=%.4f",
            self.method,
            np.linalg.cond(cov_np),
            self._vols.mean(),
        )
        return self

    @property
    def cov(self) -> pd.DataFrame:
        if self._cov is None:
            raise RuntimeError("Call fit() first.")
        return self._cov

    @property
    def corr(self) -> pd.DataFrame:
        if self._cor is None:
            raise RuntimeError("Call fit() first.")
        return self._cor

    @property
    def vols(self) -> pd.Series:
        """Annualised (×√12) volatilities per asset."""
        if self._vols is None:
            raise RuntimeError("Call fit() first.")
        return self._vols * np.sqrt(12)

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """w^T Σ w — monthly variance."""
        return float(weights @ self._cov.values @ weights)

    def portfolio_volatility(self, weights: np.ndarray, annualise: bool = True) -> float:
        """√(w^T Σ w), optionally annualised."""
        v = np.sqrt(self.portfolio_variance(weights))
        return v * np.sqrt(12) if annualise else v

    def marginal_risk_contribution(self, weights: np.ndarray) -> pd.Series:
        """Fractional marginal risk contribution per asset."""
        w = np.array(weights)
        cov_np = self._cov.values
        port_vol = np.sqrt(w @ cov_np @ w)
        mrc = (cov_np @ w) * w / (port_vol + 1e-12)
        return pd.Series(mrc / mrc.sum(), index=self._cov.columns)

    def risk_parity_weights(self, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
        """
        Compute risk-parity weights (equal risk contribution) via
        Newton-Raphson iteration.
        """
        n = len(self._cov)
        w = np.ones(n) / n
        cov_np = self._cov.values

        for _ in range(max_iter):
            port_var = w @ cov_np @ w
            mrc = cov_np @ w / np.sqrt(port_var)
            rc  = w * mrc
            target = port_var / n

            gradient = 2 * (rc - target)
            hessian  = 2 * np.diag(mrc)
            step = np.linalg.solve(hessian + 1e-6 * np.eye(n), gradient)
            w = w - 0.1 * step
            w = np.clip(w, 1e-8, None)
            w /= w.sum()

            if np.max(np.abs(step)) < tol:
                break

        return w

    # ── Estimation Methods ────────────────────────────────────────────────────

    @staticmethod
    def _sample_cov(R: np.ndarray) -> np.ndarray:
        return np.cov(R, rowvar=False)

    def _ewm_cov(self, R: np.ndarray) -> np.ndarray:
        decay = 0.5 ** (1.0 / self.halflife)
        T, N = R.shape
        weights = np.array([decay ** i for i in range(T - 1, -1, -1)])
        weights /= weights.sum()
        mu = (weights[:, None] * R).sum(axis=0)
        diff = R - mu
        return (weights[:, None] * diff).T @ diff

    @staticmethod
    def _ledoit_wolf_cov(R: np.ndarray) -> np.ndarray:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(R)
        return lw.covariance_

    @staticmethod
    def _oas_cov(R: np.ndarray) -> np.ndarray:
        oas = OAS(assume_centered=False)
        oas.fit(R)
        return oas.covariance_

    @staticmethod
    def _constant_corr_cov(R: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf (2004) constant-correlation shrinkage target.
        Target: all off-diagonal correlations set to cross-sectional mean.
        """
        T, N = R.shape
        sample = np.cov(R, rowvar=False)
        vols = np.sqrt(np.diag(sample))
        corr = sample / np.outer(vols, vols)
        np.fill_diagonal(corr, 1.0)

        # Shrinkage target: equicorrelation matrix
        rho_bar = (corr.sum() - N) / (N * (N - 1))
        target_corr = np.full((N, N), rho_bar)
        np.fill_diagonal(target_corr, 1.0)
        target_cov = target_corr * np.outer(vols, vols)

        # Analytical shrinkage intensity (Ledoit-Wolf formula)
        delta2 = np.sum((sample - target_cov) ** 2)
        pi_hat = (
            1 / (T * T) *
            np.sum([
                np.sum((np.outer(R[t], R[t]) - sample) ** 2)
                for t in range(T)
            ])
        )
        alpha = min(pi_hat / delta2, 1.0) if delta2 > 0 else 0.0
        return (1 - alpha) * sample + alpha * target_cov

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
        vols = np.sqrt(np.diag(cov.values))
        corr_np = cov.values / np.outer(vols, vols)
        np.fill_diagonal(corr_np, 1.0)
        return pd.DataFrame(corr_np, index=cov.index, columns=cov.columns)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys; sys.path.insert(0, "..")
    from config import TICKERS, START_DATE, END_DATE
    from data.data_loader import DataLoader

    loader = DataLoader(TICKERS, START_DATE, END_DATE).load()
    monthly = loader.get_monthly_returns()

    for method in ("sample", "ewm", "ledoit_wolf", "oas"):
        est = CovarianceEstimator(method=method).fit(monthly)
        print(f"\n── {method} — annualised vols ──")
        print(est.vols.round(4))

    # Risk parity weights
    est = CovarianceEstimator(method="ledoit_wolf").fit(monthly)
    rp_w = est.risk_parity_weights()
    print("\n── Risk Parity Weights ──")
    print(pd.Series(rp_w, index=monthly.columns).round(4))
