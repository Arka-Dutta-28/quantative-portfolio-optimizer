"""
models/return_forecaster.py
────────────────────────────
Ensemble ML model for forecasting next-period asset returns.

Architecture
────────────
 ┌──────────────────────────────────────────────────────────────────┐
 │  For each asset independently:                                   │
 │    • Linear Ridge (baseline)                                     │
 │    • Random Forest Regressor                                     │
 │    • XGBoost Regressor                                           │
 │    → Ensemble: inverse-MSE weighted average of the three models  │
 └──────────────────────────────────────────────────────────────────┘

Evaluation metrics reported:
  • MSE / RMSE
  • Directional Accuracy  (fraction of correct sign predictions)
  • Information Coefficient (rank correlation of predicted vs actual)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    logger.warning("xgboost not found; XGBoost model will be skipped.")
    _XGB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Single-asset model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _AssetForecaster:
    """Ensemble forecaster for one asset."""

    def __init__(self, asset: str, random_state: int = 42):
        self.asset = asset
        self._models: dict[str, Any] = {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        if _XGB_AVAILABLE:
            self._models["xgb"] = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=random_state,
                verbosity=0,
            )
        self._weights: dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_AssetForecaster":
        val_split = int(len(X) * 0.8)
        X_tr, X_val = X[:val_split], X[val_split:]
        y_tr, y_val = y[:val_split], y[val_split:]

        mse_scores: dict[str, float] = {}
        self._per_model_metrics: dict[str, dict] = {}

        for name, model in self._models.items():
            model.fit(X_tr, y_tr)
            pred_val = model.predict(X_val)
            pred_all = model.predict(X)  # in-sample for IC
            mse_val = mean_squared_error(y_val, pred_val)
            mse_scores[name] = mse_val
            dir_acc = float(np.mean(np.sign(pred_val) == np.sign(y_val)))
            ic_val, _ = spearmanr(pred_all, y)
            self._per_model_metrics[name] = {
                "val_rmse":    round(float(np.sqrt(mse_val)), 6),
                "dir_acc":     round(dir_acc, 4),
                "ic":          round(float(ic_val) if not np.isnan(ic_val) else 0.0, 4),
            }

        # Inverse-MSE weighting (lower error → higher weight)
        inv = {k: 1.0 / (v + 1e-12) for k, v in mse_scores.items()}
        total = sum(inv.values())
        self._weights = {k: v / total for k, v in inv.items()}

        # Refit on full training set
        for model in self._models.values():
            model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction: sum(w_m * model_m.predict(X))."""
        ensemble = np.zeros(len(X))
        for name, model in self._models.items():
            ensemble += self._weights.get(name, 0.0) * model.predict(X)
        return ensemble

    @property
    def weights(self) -> dict[str, float]:
        return self._weights

    @property
    def per_model_metrics(self) -> dict[str, dict]:
        return getattr(self, "_per_model_metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset forecaster
# ─────────────────────────────────────────────────────────────────────────────

class ReturnForecaster:
    """
    Trains one _AssetForecaster per asset and collects evaluation metrics.

    Parameters
    ----------
    assets        : list of ticker strings
    random_state  : reproducibility seed
    """

    def __init__(self, assets: list[str], random_state: int = 42):
        self.assets = assets
        self.random_state = random_state
        self._forecasters: dict[str, _AssetForecaster] = {}
        self.metrics: dict[str, dict[str, float]] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "ReturnForecaster":
        """
        Fit one ensemble model per asset.

        Parameters
        ----------
        X : feature matrix  (n_obs, n_features)
        y : target returns  (n_obs, n_assets)
        """
        logger.info("Training return forecasters for %d assets …", len(self.assets))
        X_np = X.values

        for asset in self.assets:
            if asset not in y.columns:
                logger.warning("Asset %s not in targets; skipping.", asset)
                continue

            y_np = y[asset].values
            forecaster = _AssetForecaster(asset, self.random_state)
            forecaster.fit(X_np, y_np)
            self._forecasters[asset] = forecaster

            # Compute in-sample metrics for logging
            preds = forecaster.predict(X_np)
            mse  = mean_squared_error(y_np, preds)
            rmse = np.sqrt(mse)
            dir_acc = np.mean(np.sign(preds) == np.sign(y_np))
            ic, _ = spearmanr(preds, y_np)

            self.metrics[asset] = {
                "mse":          round(mse, 6),
                "rmse":         round(rmse, 6),
                "dir_accuracy": round(dir_acc, 4),
                "ic":           round(ic if not np.isnan(ic) else 0.0, 4),
            }
            logger.debug(
                "  %s — RMSE=%.4f  DirAcc=%.2f%%  IC=%.3f",
                asset, rmse, dir_acc * 100, ic,
            )

        logger.info("Training complete. Avg IC: %.3f", self._avg_ic())
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict next-period expected returns for all assets.

        Returns
        -------
        pd.DataFrame (n_obs, n_assets) — annualised return forecasts
        """
        X_np = X.values
        preds = {}
        for asset, forecaster in self._forecasters.items():
            preds[asset] = forecaster.predict(X_np)

        return pd.DataFrame(preds, index=X.index)[self.assets]

    def predict_latest(self, X: pd.DataFrame) -> pd.Series:
        """Return a single-row expected return vector (latest date)."""
        all_preds = self.predict(X)
        return all_preds.iloc[-1]

    # ── Metrics ───────────────────────────────────────────────────────────────

    def metrics_summary(self) -> pd.DataFrame:
        """Returns a summary DataFrame of per-asset metrics (ensemble)."""
        return pd.DataFrame(self.metrics).T

    def per_model_summary(self) -> pd.DataFrame:
        """Returns a DataFrame showing each sub-model performance across assets.
        Rows = assets, columns = (model, metric) MultiIndex."""
        rows = {}
        for asset, forecaster in self._forecasters.items():
            row = {}
            for model_name, m in forecaster.per_model_metrics.items():
                for metric, val in m.items():
                    row[f"{model_name}_{metric}"] = val
            # also add ensemble weight per model
            for model_name, w in forecaster.weights.items():
                row[f"{model_name}_weight"] = round(w, 4)
            rows[asset] = row
        return pd.DataFrame(rows).T

    def _avg_ic(self) -> float:
        if not self.metrics:
            return 0.0
        ics = [v["ic"] for v in self.metrics.values()]
        return float(np.nanmean(ics))

    # ── Walk-forward evaluation ───────────────────────────────────────────────

    def walk_forward_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        initial_train_pct: float = 0.6,
    ) -> pd.DataFrame:
        """
        Out-of-sample evaluation using expanding window walk-forward.

        Returns
        -------
        DataFrame with columns [date, asset, predicted, actual]
        """
        n = len(X)
        split = int(n * initial_train_pct)
        records = []

        logger.info("Walk-forward evaluation: %d folds …", n - split)

        for t in range(split, n):
            X_tr = X.iloc[:t]
            y_tr = y.iloc[:t]
            X_te = X.iloc[[t]]
            y_te = y.iloc[[t]]

            tmp = ReturnForecaster(self.assets, self.random_state)
            tmp.fit(X_tr, y_tr)
            pred = tmp.predict(X_te)

            for asset in self.assets:
                if asset in pred.columns and asset in y_te.columns:
                    records.append({
                        "date":      X.index[t],
                        "asset":     asset,
                        "predicted": pred[asset].iloc[0],
                        "actual":    y_te[asset].iloc[0],
                    })

        df = pd.DataFrame(records)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys; sys.path.insert(0, "..")
    from config import TICKERS, START_DATE, END_DATE
    from data.data_loader import DataLoader
    from features.feature_engineering import FeatureEngineer

    loader = DataLoader(TICKERS, START_DATE, END_DATE).load()
    monthly = loader.get_monthly_returns()
    macro   = loader.get_macro_features()

    fe = FeatureEngineer()
    X, y = fe.fit_transform(monthly, macro)

    split = int(len(X) * 0.7)
    forecaster = ReturnForecaster(TICKERS)
    forecaster.fit(X.iloc[:split], y.iloc[:split])

    print("\n── Metrics Summary ──")
    print(forecaster.metrics_summary())
    print("\n── Latest Forecasts ──")
    print(forecaster.predict_latest(X))
