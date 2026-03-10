"""
models/regime_detector.py
──────────────────────────
Unsupervised market regime detection using Gaussian Mixture Models.

Detected regimes:
  0 → Bull  (positive momentum, low volatility)
  1 → Neutral / Transition
  2 → Bear / Crisis (negative momentum, high volatility)

The detector outputs:
  • regime labels per period
  • regime probabilities (soft assignments)
  • regime-specific statistics
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_NAMES_3 = {0: "Bull", 1: "Neutral", 2: "Bear/Crisis"}
REGIME_NAMES_4 = {0: "Bull", 1: "Recovery", 2: "Slowdown", 3: "Bear/Crisis"}
REGIME_NAMES = REGIME_NAMES_3
REGIME_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}


class RegimeDetector:
    """
    Detects market regimes from financial time series using a GMM.

    Parameters
    ----------
    n_regimes    : number of regimes (default 3)
    random_state : reproducibility
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._gmm: GaussianMixture | None = None
        self._scaler = StandardScaler()
        self._regime_map: dict[int, int] = {}  # GMM label → canonical label
        self.regime_stats: pd.DataFrame | None = None
        self.regime_names: dict[int, str] = (
            REGIME_NAMES_4 if n_regimes >= 4 else REGIME_NAMES_3
        )

    # ── Core Methods ──────────────────────────────────────────────────────────

    def fit(self, monthly_returns: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the GMM on regime features derived from monthly returns.
        """
        features = self._build_features(monthly_returns)
        F = self._scaler.fit_transform(features.values)

        self._gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=10,
            random_state=self.random_state,
        )
        raw_labels = self._gmm.fit_predict(F)

        # Canonical ordering: sort regimes by mean market return
        labels_series = pd.Series(raw_labels, index=features.index)
        regime_return = monthly_returns.mean(axis=1)

        mean_returns = {}
        for lbl in range(self.n_regimes):
            mask = labels_series == lbl
            idx = labels_series.index[mask]
            valid_idx = idx.intersection(regime_return.index)
            mean_returns[lbl] = regime_return.loc[valid_idx].mean() if len(valid_idx) > 0 else 0.0

        sorted_lbls = sorted(mean_returns, key=mean_returns.get, reverse=True)
        self._regime_map = {orig: new for new, orig in enumerate(sorted_lbls)}

        logger.info(
            "GMM regime detection: BIC=%.1f  labels=%s",
            self._gmm.bic(F),
            dict(pd.Series(raw_labels).map(self._regime_map).value_counts()),
        )
        return self

    def predict(self, monthly_returns: pd.DataFrame) -> pd.Series:
        """Return canonical regime labels for each period."""
        features = self._build_features(monthly_returns)
        F = self._scaler.transform(features.values)
        raw_labels = self._gmm.predict(F)
        return pd.Series(
            [self._regime_map[l] for l in raw_labels],
            index=features.index,
            name="regime",
        )

    def predict_proba(self, monthly_returns: pd.DataFrame) -> pd.DataFrame:
        """Return regime probability distributions per period."""
        features = self._build_features(monthly_returns)
        F = self._scaler.transform(features.values)
        proba_raw = self._gmm.predict_proba(F)
        # Rearrange columns to canonical order
        n = self.n_regimes
        proba_canon = np.zeros_like(proba_raw)
        for orig, canon in self._regime_map.items():
            proba_canon[:, canon] = proba_raw[:, orig]
        cols = [self.regime_names.get(i, f"Regime {i}") for i in range(n)]
        return pd.DataFrame(proba_canon, index=features.index, columns=cols)

    def fit_predict(self, monthly_returns: pd.DataFrame) -> pd.Series:
        """Convenience method: fit the GMM and return regime labels in one call."""
        return self.fit(monthly_returns).predict(monthly_returns)

    def compute_regime_stats(
        self, monthly_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Return per-regime mean return, volatility, Sharpe ratio."""
        labels = self.predict(monthly_returns)
        aligned = monthly_returns.reindex(labels.index)
        ew = aligned.mean(axis=1)

        rows = []
        for regime in range(self.n_regimes):
            mask = labels == regime
            r = ew[mask]
            rows.append({
                "Regime":          self.regime_names.get(regime, str(regime)),
                "Count":           int(mask.sum()),
                "Mean Return (m)": round(r.mean() * 100, 3),
                "Ann. Return":     round(r.mean() * 12 * 100, 2),
                "Ann. Volatility": round(r.std() * np.sqrt(12) * 100, 2),
                "Sharpe (ann.)":   round(
                    r.mean() * 12 / (r.std() * np.sqrt(12) + 1e-9), 3
                ),
            })
        self.regime_stats = pd.DataFrame(rows).set_index("Regime")
        return self.regime_stats

    def get_regime_weights(self, regime: int) -> dict[str, float]:
        """
        Return suggested portfolio tilt weights based on the regime.
        These multipliers are applied to the optimizer's expected returns.
        """
        if self.n_regimes >= 4:
            tilts = {
                0: {"Equity": 1.3, "Bond": 0.7, "Commodity": 1.0, "Real Estate": 1.1},  # Bull
                1: {"Equity": 1.1, "Bond": 0.9, "Commodity": 1.1, "Real Estate": 1.0},  # Recovery
                2: {"Equity": 0.9, "Bond": 1.1, "Commodity": 1.0, "Real Estate": 0.9},  # Slowdown
                3: {"Equity": 0.6, "Bond": 1.4, "Commodity": 1.3, "Real Estate": 0.7},  # Bear/Crisis
            }
        else:
            tilts = {
                0: {"Equity": 1.3, "Bond": 0.7, "Commodity": 1.0, "Real Estate": 1.1},  # Bull
                1: {"Equity": 1.0, "Bond": 1.0, "Commodity": 1.0, "Real Estate": 1.0},  # Neutral
                2: {"Equity": 0.6, "Bond": 1.4, "Commodity": 1.3, "Real Estate": 0.7},  # Bear
            }
        return tilts.get(regime, tilts[1])

    # ── Feature Builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_features(monthly_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute regime-detection features."""
        ew = monthly_returns.mean(axis=1)
        vol = monthly_returns.std(axis=1)

        df = pd.DataFrame(index=monthly_returns.index)
        df["market_return"]    = ew
        df["volatility_3m"]    = vol.rolling(3).mean()
        df["momentum_6m"]      = ew.rolling(6).sum()
        df["momentum_12m"]     = ew.rolling(12).sum()
        df["vol_change_3m"]    = df["volatility_3m"].pct_change(3)
        df["drawdown_6m"]      = (
            ew.rolling(6).apply(
                lambda x: (x[-1] - x.max()) / (x.max() + 1e-9),
                raw=True
            )
        )
        return df.dropna()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys; sys.path.insert(0, "..")
    from config import TICKERS, START_DATE, END_DATE
    from data.data_loader import DataLoader

    loader = DataLoader(TICKERS, START_DATE, END_DATE).load()
    monthly = loader.get_monthly_returns()

    detector = RegimeDetector(n_regimes=3)
    regimes = detector.fit_predict(monthly)

    print("\n── Regime Counts ──")
    print(regimes.value_counts().sort_index())
    print("\n── Regime Stats ──")
    print(detector.compute_regime_stats(monthly))
