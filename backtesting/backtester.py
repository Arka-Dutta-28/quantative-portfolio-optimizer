"""
backtesting/backtester.py  —  Walk-forward backtesting engine (fast version).

Optimisations vs naive approach:
  • Regime labels precomputed ONCE on full history (no per-step GMM refit).
  • ML model retrained every `ml_retrain_freq` rebalances, not every month.
  • Simple Ridge regression inside the backtest loop (fast, accurate enough).
  • Quarterly rebalancing default (still realistic for institutional SAA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtesting (all fields have sensible defaults)."""
    strategy:           str   = "mvo"
    cov_method:         str   = "ledoit_wolf"
    initial_capital:    float = 1_000_000.0
    transaction_cost:   float = 0.001
    rebalance_freq:     str   = "QE"        # quarterly
    min_train_periods:  int   = 36
    ml_retrain_freq:    int   = 4           # retrain every N rebalances
    use_ml_forecasts:   bool  = True
    use_regime_filter:  bool  = True
    risk_free_rate:     float = 0.04 / 12
    max_weight:         float = 0.35
    min_weight:         float = 0.00
    vol_target:         float = 0.10 / np.sqrt(12)
    max_turnover:       float = 0.60


class Backtester:
    """
    Walk-forward backtesting engine for portfolio strategies.

    On each rebalance date (quarterly by default) the engine:
      1. Re-estimates covariance on the expanding training window.
      2. Optionally retrains a fast Ridge ML model for expected returns.
      3. Optionally applies regime tilts to the return forecast.
      4. Runs the configured optimizer to produce new weights.
      5. Deducts transaction costs proportional to turnover.

    Results stored after .run():
      portfolio_returns   — net-of-cost monthly return series
      portfolio_weights   — weight history (DataFrame)
      turnover_series     — per-rebalance turnover
      regime_labels       — detected regime at each rebalance
      transaction_costs   — dollar cost per rebalance

    Parameters
    ----------
    returns : pd.DataFrame  — monthly asset returns (T x N)
    config  : BacktestConfig — strategy, constraints, and toggles
    macro   : pd.DataFrame   — optional macro features for ML
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        config: BacktestConfig | None = None,
        macro: pd.DataFrame | None = None,
    ):
        self.returns = returns
        self.config  = config or BacktestConfig()
        self.macro   = macro
        self.assets  = returns.columns.tolist()
        self.n       = len(self.assets)

        self.portfolio_returns: pd.Series | None = None
        self.portfolio_weights: pd.DataFrame | None = None
        self.turnover_series:   pd.Series | None = None
        self.regime_labels:     pd.Series | None = None
        self.transaction_costs: pd.Series | None = None

    def run(self) -> "Backtester":
        """Execute the full walk-forward backtest loop."""
        logger.info(
            "Backtest [%s | cov=%s | ML=%s | regime=%s]",
            self.config.strategy, self.config.cov_method,
            self.config.use_ml_forecasts, self.config.use_regime_filter,
        )

        # ── Precompute regime labels once ─────────────────────────────────────
        regime_map: pd.Series | None = None
        if self.config.use_regime_filter:
            regime_map = self._precompute_regimes()

        dates   = self.returns.index
        min_t   = self.config.min_train_periods
        rebal_dates = set(
            pd.date_range(start=dates[min_t], end=dates[-1], freq=self.config.rebalance_freq)
        )

        port_rets    = []
        port_dates   = []
        weights_hist = []
        to_list      = []
        tc_list      = []
        reg_list     = []

        current_w   = np.ones(self.n) / self.n
        cached_mu   = None
        rebal_count = 0

        for t, date in enumerate(dates):
            if t < min_t:
                continue

            period_ret = float(self.returns.iloc[t].values @ current_w)

            if date in rebal_dates:
                train = self.returns.iloc[:t]
                rebal_count += 1

                # Covariance
                from risk.covariance_estimator import CovarianceEstimator
                cov = CovarianceEstimator(self.config.cov_method).fit(train).cov.values

                # Expected returns (Ridge ML or shrunk sample mean)
                if self.config.use_ml_forecasts and (
                    cached_mu is None or rebal_count % self.config.ml_retrain_freq == 1
                ):
                    cached_mu = self._fast_ml(train)

                mu = cached_mu.copy() if cached_mu is not None else train.mean().values

                # Regime tilt
                regime = 1
                if regime_map is not None and date in regime_map.index:
                    regime = int(regime_map.loc[date])
                    mu = self._apply_tilt(mu, regime)

                # Optimize
                new_w = self._optimize(mu, cov, current_w)

                # Transaction costs
                to  = float(np.abs(new_w - current_w).sum())
                tc  = to * self.config.transaction_cost
                period_ret -= tc

                current_w = new_w
                to_list.append((date, to))
                tc_list.append((date, tc))
                reg_list.append((date, regime))

            port_rets.append(period_ret)
            port_dates.append(date)
            weights_hist.append(current_w.copy())

        # ── Store results ─────────────────────────────────────────────────────
        idx = pd.DatetimeIndex(port_dates)
        self.portfolio_returns = pd.Series(port_rets, index=idx, name="portfolio")
        self.portfolio_weights = pd.DataFrame(weights_hist, index=idx, columns=self.assets)

        def _to_series(lst, name):
            if not lst: return None
            d, v = zip(*lst)
            return pd.Series(v, index=pd.DatetimeIndex(d), name=name)

        self.turnover_series   = _to_series(to_list,  "turnover")
        self.transaction_costs = _to_series(tc_list,  "tc")
        self.regime_labels     = _to_series(reg_list, "regime")

        total = (np.prod(1 + np.array(port_rets)) - 1) * 100
        logger.info("Done. Total return: %.1f%%  Rebalances: %d", total, rebal_count)
        return self

    def get_benchmark_returns(self, benchmark: str = "equal_weight") -> pd.Series:
        """Compute benchmark return series ('equal_weight', 'spy_only', 'risk_parity')."""
        r = self.returns.reindex(self.portfolio_returns.index)
        if benchmark == "equal_weight":
            return r.mean(axis=1).rename("Equal Weight")
        elif benchmark == "spy_only":
            col = "SPY" if "SPY" in r.columns else r.columns[0]
            return r[col].rename("S&P 500 (SPY)")
        elif benchmark == "risk_parity":
            from risk.covariance_estimator import CovarianceEstimator
            w_rp = CovarianceEstimator("ledoit_wolf").fit(self.returns).risk_parity_weights()
            return r.dot(w_rp).rename("Risk Parity")
        return r.mean(axis=1).rename("Equal Weight")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _precompute_regimes(self) -> pd.Series:
        """Fit GMM once on full history to avoid per-step refitting."""
        from models.regime_detector import RegimeDetector
        try:
            return RegimeDetector(n_regimes=3).fit_predict(self.returns)
        except Exception as e:
            logger.warning("Regime precompute failed: %s", e)
            return pd.Series(1, index=self.returns.index, name="regime")

    def _fast_ml(self, train: pd.DataFrame) -> np.ndarray:
        """Quick Ridge regression on momentum/vol features for expected returns."""
        try:
            feats = []
            for w in [1, 3, 6, 12]:
                f = train.rolling(w).sum()
                f.columns = [f"{c}_m{w}" for c in train.columns]
                feats.append(f)
            for w in [3, 6, 12]:
                f = train.rolling(w).std()
                f.columns = [f"{c}_v{w}" for c in train.columns]
                feats.append(f)

            X = pd.concat(feats, axis=1).dropna()
            y = train.shift(-1).reindex(X.index).dropna()
            X = X.reindex(y.index)
            if len(X) < 15:
                return train.mean().values

            sc   = StandardScaler()
            Xsc  = sc.fit_transform(X.values)
            mu_l = []
            for i in range(self.n):
                m = Ridge(alpha=1.0)
                m.fit(Xsc[:-1], y.values[:-1, i])
                mu_l.append(float(m.predict(Xsc[[-1]])[0]))

            return np.clip(np.array(mu_l), -0.10, 0.10)
        except Exception as e:
            logger.debug("ML failed: %s", e)
            return train.mean().values

    def _apply_tilt(self, mu: np.ndarray, regime: int) -> np.ndarray:
        """Multiply expected returns by regime-specific asset-class tilts."""
        from models.regime_detector import RegimeDetector
        from config import ASSET_CLASSES
        tilts = RegimeDetector().get_regime_weights(regime)
        mu_t  = mu.copy()
        for i, a in enumerate(self.assets):
            mu_t[i] *= tilts.get(ASSET_CLASSES.get(a, "Equity"), 1.0)
        return mu_t

    def _optimize(self, mu, cov, cur_w):
        """Run the configured portfolio optimizer with current constraints."""
        from optimization.portfolio_optimizer import PortfolioOptimizer, OptimizationConfig
        cfg = OptimizationConfig(
            strategy       = self.config.strategy,
            max_weight     = self.config.max_weight,
            min_weight     = self.config.min_weight,
            vol_target     = self.config.vol_target,
            max_turnover   = self.config.max_turnover,
            risk_free_rate = self.config.risk_free_rate,
        )
        return PortfolioOptimizer(self.assets, cfg).optimize(mu, cov, cur_w)
