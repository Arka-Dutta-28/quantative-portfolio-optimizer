"""
features/feature_engineering.py
────────────────────────────────
Constructs predictive features for the ML return-forecast model.

Inputs  : monthly returns DataFrame, optional macro DataFrame
Outputs : feature matrix X and target matrix y (next-period returns)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Build a feature matrix from monthly asset returns + optional macro data.

    Features constructed per asset:
    ─────────────────────────────────────────────────────────
    Momentum          : 1m, 3m, 6m, 12m trailing returns
    Mean-reversion    : return vs. 12m moving average
    Volatility        : 3m, 6m, 12m rolling standard deviation (annualised)
    Correlation shift : rolling correlation vs. equal-weight index
    Cross-sectional   : z-score of 1m return vs. peers
    Macro (if given)  : VIX, yield spread, CPI trend (shared across assets)
    ─────────────────────────────────────────────────────────

    Parameters
    ----------
    lookback_windows   : months for momentum features
    vol_windows        : months for volatility features
    forecast_horizon   : months ahead for the target return
    scale_features     : whether to StandardScale before returning
    """

    def __init__(
        self,
        lookback_windows: list[int] = [1, 3, 6, 12],
        vol_windows: list[int] = [3, 6, 12],
        forecast_horizon: int = 1,
        scale_features: bool = True,
    ):
        self.lookback_windows = lookback_windows
        self.vol_windows = vol_windows
        self.forecast_horizon = forecast_horizon
        self.scale_features = scale_features
        self.scaler = StandardScaler()
        self._feature_names: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        monthly_returns: pd.DataFrame,
        macro: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute features and targets from the full history.

        Returns
        -------
        X : pd.DataFrame  shape (n_obs, n_features)
        y : pd.DataFrame  shape (n_obs, n_assets)  — next-period returns
        """
        logger.info("Building features for %d assets …", len(monthly_returns.columns))

        feat_list = []

        # ── Momentum features ────────────────────────────────────────────────
        for w in self.lookback_windows:
            mom = monthly_returns.rolling(w).sum()
            mom.columns = [f"{c}_mom_{w}m" for c in monthly_returns.columns]
            feat_list.append(mom)

        # ── Volatility features ───────────────────────────────────────────────
        for w in self.vol_windows:
            vol = monthly_returns.rolling(w).std() * np.sqrt(12)
            vol.columns = [f"{c}_vol_{w}m" for c in monthly_returns.columns]
            feat_list.append(vol)

        # ── Mean-reversion: distance from 12m MA ─────────────────────────────
        ma12 = monthly_returns.rolling(12).mean()
        mean_rev = monthly_returns - ma12
        mean_rev.columns = [f"{c}_mean_rev" for c in monthly_returns.columns]
        feat_list.append(mean_rev)

        # ── Cross-sectional momentum z-score ─────────────────────────────────
        for w in [1, 3]:
            xsec = monthly_returns.rolling(w).sum()
            xsec_z = xsec.sub(xsec.mean(axis=1), axis=0).div(
                xsec.std(axis=1) + 1e-9, axis=0
            )
            xsec_z.columns = [f"{c}_xsec_z_{w}m" for c in monthly_returns.columns]
            feat_list.append(xsec_z)

        # ── Rolling correlation with equal-weight index ───────────────────────
        ew_index = monthly_returns.mean(axis=1)
        corr_ew = monthly_returns.rolling(12).apply(
            lambda x: x.corr(ew_index.loc[x.index]), raw=False
        )
        corr_ew.columns = [f"{c}_corr_ew_12m" for c in monthly_returns.columns]
        feat_list.append(corr_ew)

        # ── Macro features (broadcast to all dates) ───────────────────────────
        if macro is not None:
            macro_aligned = macro.reindex(monthly_returns.index).ffill().bfill()
            feat_list.append(macro_aligned)

        # ── Combine all features ──────────────────────────────────────────────
        X_raw = pd.concat(feat_list, axis=1)
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

        # ── Target: forward return ─────────────────────────────────────────────
        y = monthly_returns.shift(-self.forecast_horizon)

        # Align and drop NaN rows
        combined = pd.concat([X_raw, y.add_suffix("_target")], axis=1).dropna()
        target_cols = [f"{c}_target" for c in monthly_returns.columns]
        feat_cols = [c for c in combined.columns if c not in target_cols]

        X_clean = combined[feat_cols]
        y_clean = combined[target_cols]
        y_clean.columns = monthly_returns.columns

        self._feature_names = feat_cols
        logger.info("Feature matrix shape: %s", X_clean.shape)

        if self.scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_clean),
                index=X_clean.index,
                columns=feat_cols,
            )
            return X_scaled, y_clean
        return X_clean, y_clean

    def transform(
        self,
        monthly_returns: pd.DataFrame,
        macro: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Transform new data using the already-fitted scaler.
        Returns feature matrix only (no target shift).
        """
        feat_list = []

        for w in self.lookback_windows:
            mom = monthly_returns.rolling(w).sum()
            mom.columns = [f"{c}_mom_{w}m" for c in monthly_returns.columns]
            feat_list.append(mom)

        for w in self.vol_windows:
            vol = monthly_returns.rolling(w).std() * np.sqrt(12)
            vol.columns = [f"{c}_vol_{w}m" for c in monthly_returns.columns]
            feat_list.append(vol)

        ma12 = monthly_returns.rolling(12).mean()
        mean_rev = monthly_returns - ma12
        mean_rev.columns = [f"{c}_mean_rev" for c in monthly_returns.columns]
        feat_list.append(mean_rev)

        for w in [1, 3]:
            xsec = monthly_returns.rolling(w).sum()
            xsec_z = xsec.sub(xsec.mean(axis=1), axis=0).div(
                xsec.std(axis=1) + 1e-9, axis=0
            )
            xsec_z.columns = [f"{c}_xsec_z_{w}m" for c in monthly_returns.columns]
            feat_list.append(xsec_z)

        ew_index = monthly_returns.mean(axis=1)
        corr_ew = monthly_returns.rolling(12).apply(
            lambda x: x.corr(ew_index.loc[x.index]), raw=False
        )
        corr_ew.columns = [f"{c}_corr_ew_12m" for c in monthly_returns.columns]
        feat_list.append(corr_ew)

        if macro is not None:
            macro_aligned = macro.reindex(monthly_returns.index).ffill().bfill()
            feat_list.append(macro_aligned)

        X_raw = pd.concat(feat_list, axis=1)
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan).dropna()

        # Only keep columns seen during training
        X_aligned = X_raw.reindex(columns=self._feature_names, fill_value=0)

        if self.scale_features:
            return pd.DataFrame(
                self.scaler.transform(X_aligned),
                index=X_aligned.index,
                columns=self._feature_names,
            )
        return X_aligned

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys; sys.path.insert(0, "..")
    from config import TICKERS, START_DATE, END_DATE
    from data.data_loader import DataLoader

    loader = DataLoader(TICKERS, START_DATE, END_DATE).load()
    monthly = loader.get_monthly_returns()
    macro   = loader.get_macro_features()

    fe = FeatureEngineer()
    X, y = fe.fit_transform(monthly, macro)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nSample features:\n{X.iloc[:3, :5]}")
