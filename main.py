"""
main.py
────────
Main pipeline orchestrator for the ML-Enhanced Quantitative Portfolio Optimizer.

Run:
    python main.py

This script runs the complete end-to-end pipeline:
  1. Data loading / generation
  2. Feature engineering
  3. ML model training + evaluation
  4. Regime detection
  5. Portfolio optimization
  6. Walk-forward backtest (multiple strategies)
  7. Performance analytics
  8. All visualisations saved to outputs/
  9. Summary report printed to console
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Project imports ───────────────────────────────────────────────────────────
import config as cfg
from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from models.return_forecaster import ReturnForecaster
from models.ar_forecaster import ARForecaster
from models.regime_detector import RegimeDetector
from risk.covariance_estimator import CovarianceEstimator
from optimization.portfolio_optimizer import PortfolioOptimizer, OptimizationConfig
from backtesting.backtester import Backtester, BacktestConfig
from analytics.performance_metrics import PerformanceAnalytics
from visualization import plots as viz


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────────────────────────────────────

def step_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP 1: Load price data and compute monthly returns + macro features."""
    logger.info("─" * 60)
    logger.info("STEP 1 — Loading market data")
    loader = DataLoader(cfg.TICKERS, cfg.START_DATE, cfg.END_DATE)
    loader.load()
    monthly = loader.get_monthly_returns()
    macro   = loader.get_macro_features()
    logger.info("Loaded %d months × %d assets", len(monthly), len(monthly.columns))
    return monthly, macro


def step_features(
    monthly: pd.DataFrame, macro: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureEngineer]:
    """STEP 2: Build momentum / volatility / macro feature matrix and targets."""
    logger.info("─" * 60)
    logger.info("STEP 2 — Feature engineering")
    fe = FeatureEngineer(
        lookback_windows=cfg.LOOKBACK_WINDOWS,
        vol_windows=[3, 6, 12],
        forecast_horizon=cfg.FORECAST_HORIZON,
        scale_features=True,
    )
    X, y = fe.fit_transform(monthly, macro)
    logger.info("Feature matrix: %s   Target matrix: %s", X.shape, y.shape)
    return X, y, fe


def step_ml_model(
    X: pd.DataFrame, y: pd.DataFrame
) -> ReturnForecaster:
    """STEP 3: Train the ML ensemble (Ridge, RF, XGBoost) on the training split."""
    logger.info("─" * 60)
    logger.info("STEP 3 — ML return forecasting")
    split = int(len(X) * 0.7)
    forecaster = ReturnForecaster(cfg.TICKERS, cfg.RANDOM_STATE)
    forecaster.fit(X.iloc[:split], y.iloc[:split])

    metrics = forecaster.metrics_summary()
    logger.info("\n%s", metrics.to_string())
    logger.info(
        "Avg IC: %.3f   Avg Dir. Accuracy: %.1f%%",
        metrics["ic"].astype(float).mean(),
        metrics["dir_accuracy"].astype(float).mean() * 100,
    )
    return forecaster



def step_ar_model(monthly: pd.DataFrame) -> ARForecaster:
    """STEP 3b: Fit AR / ARMA / ARCH / GARCH / EGARCH models with BIC auto-selection."""
    logger.info("-" * 60)
    logger.info("STEP 3b - Autoregressive / GARCH time-series forecasting (BIC auto-selection)")
    split = int(len(monthly) * 0.7)
    ar_fc = ARForecaster(cfg.TICKERS, ar_order=2)
    ar_fc.fit(monthly.iloc[:split])
    metrics = ar_fc.metrics_summary()
    logger.info("\nAR Ensemble Metrics:\n%s", metrics.to_string())
    sel = ar_fc.selection_summary()
    if sel is not None and not sel.empty:
        logger.info("\nBIC Model Selection:\n%s", sel.to_string())
    logger.info("Avg AR IC: %.3f   Avg Dir. Accuracy: %.1f%%",
        metrics["ic"].astype(float).mean(),
        metrics["dir_accuracy"].astype(float).mean() * 100)
    return ar_fc


def step_regime(monthly: pd.DataFrame) -> RegimeDetector:
    """STEP 4: Detect market regimes (Bull / Neutral / Bear) via GMM."""
    logger.info("─" * 60)
    logger.info("STEP 4 — Market regime detection")
    detector = RegimeDetector(n_regimes=cfg.N_REGIMES, random_state=cfg.RANDOM_STATE)
    regimes = detector.fit_predict(monthly)
    stats   = detector.compute_regime_stats(monthly)
    logger.info("\n%s", stats.to_string())
    return detector


def step_optimizer_snapshot(
    monthly: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Single-period optimization for frontier and strategy comparison."""
    logger.info("─" * 60)
    logger.info("STEP 5 — Portfolio optimization (snapshot)")

    est = CovarianceEstimator(cfg.COVARIANCE_METHOD).fit(monthly)
    mu  = monthly.mean().values

    weights = {}
    for strat in ("mvo", "min_vol", "max_sharpe", "risk_parity", "cvar", "equal_weight"):
        opt_cfg = OptimizationConfig(
            strategy       = strat,
            max_weight     = cfg.MAX_WEIGHT,
            min_weight     = cfg.MIN_WEIGHT,
            vol_target     = cfg.TARGET_VOLATILITY / np.sqrt(12),
            max_turnover   = cfg.MAX_TURNOVER,
            risk_free_rate = cfg.RISK_FREE_RATE / 12,
        )
        opt = PortfolioOptimizer(cfg.TICKERS, opt_cfg)
        w   = opt.optimize(mu, est.cov.values)
        weights[strat] = w

    for name, w in weights.items():
        wdf = pd.Series(w, index=cfg.TICKERS).round(4)
        logger.info("  [%s]  %s", name, dict(wdf[wdf > 0.01]))

    return weights


def step_backtest(
    monthly: pd.DataFrame, macro: pd.DataFrame
) -> dict[str, pd.Series]:
    """STEP 6: Run walk-forward backtests for five strategies plus benchmarks."""
    logger.info("─" * 60)
    logger.info("STEP 6 — Walk-forward backtesting")

    strategy_configs = {
        "MVO + ML + Regime": BacktestConfig(
            strategy="mvo", use_ml_forecasts=True, use_regime_filter=True,
        ),
        "MVO (No ML)": BacktestConfig(
            strategy="mvo", use_ml_forecasts=False, use_regime_filter=False,
        ),
        "Min-Vol": BacktestConfig(
            strategy="min_vol", use_ml_forecasts=False, use_regime_filter=False,
        ),
        "Max-Sharpe": BacktestConfig(
            strategy="max_sharpe", use_ml_forecasts=True, use_regime_filter=False,
        ),
        "CVaR": BacktestConfig(
            strategy="cvar", use_ml_forecasts=True, use_regime_filter=True,
        ),
    }

    results: dict[str, pd.Series] = {}
    backtester_objects: dict[str, Backtester] = {}

    for name, bt_cfg in strategy_configs.items():
        logger.info("  Running: %s", name)
        bt = Backtester(monthly, bt_cfg, macro=macro)
        bt.run()
        results[name] = bt.portfolio_returns
        backtester_objects[name] = bt

    # Add benchmarks using the first backtester
    primary_bt = list(backtester_objects.values())[0]
    results["Equal Weight"] = primary_bt.get_benchmark_returns("equal_weight")
    results["Risk Parity"]  = primary_bt.get_benchmark_returns("risk_parity")
    results["S&P 500"]      = primary_bt.get_benchmark_returns("spy_only")

    return results, backtester_objects


def step_analytics(results: dict[str, pd.Series]) -> pd.DataFrame:
    """STEP 7: Compute and log the cross-strategy comparison table."""
    logger.info("─" * 60)
    logger.info("STEP 7 — Performance analytics")

    sp500 = results.get("S&P 500")
    comparison = PerformanceAnalytics.compare(results, benchmark=sp500)
    logger.info("\n%s", comparison.to_string())
    return comparison


def step_visualisations(
    results: dict[str, pd.Series],
    backtester_objects: dict[str, Backtester],
    monthly: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """STEP 8: Generate all matplotlib charts and save to the outputs/ directory."""
    logger.info("─" * 60)
    logger.info("STEP 8 — Generating visualisations → %s", OUT)

    primary_name = "MVO + ML + Regime"
    primary_bt   = backtester_objects[primary_name]
    port_rets    = results[primary_name]
    benchmarks   = {k: v for k, v in results.items() if k != primary_name}

    # 1. Cumulative returns
    viz.plot_cumulative_returns(
        results, output_path=OUT / "01_cumulative_returns.png", cfg=cfg
    )

    # 2. Allocation over time
    viz.plot_allocation_over_time(
        primary_bt.portfolio_weights,
        output_path=OUT / "02_allocation_over_time.png",
    )

    # 3. Efficient frontier
    est = CovarianceEstimator(cfg.COVARIANCE_METHOD).fit(monthly)
    mu  = monthly.mean().values
    opt_cfg = OptimizationConfig(max_weight=cfg.MAX_WEIGHT, vol_target=10.0)
    opt     = PortfolioOptimizer(cfg.TICKERS, opt_cfg)
    frontier = opt.efficient_frontier(mu, est.cov.values, n_points=50)

    ind_assets = pd.DataFrame({
        "vol": monthly.std() * np.sqrt(12),
        "ret": monthly.mean() * 12,
    }).rename_axis("ticker")

    viz.plot_efficient_frontier(
        frontier, individual_assets=ind_assets,
        output_path=OUT / "03_efficient_frontier.png",
    )

    # 4. Drawdown
    viz.plot_drawdown(
        {k: v for k, v in list(results.items())[:4]},
        output_path=OUT / "04_drawdown.png",
    )

    # 5. Return distribution
    viz.plot_return_distribution(
        port_rets, output_path=OUT / "05_return_distribution.png"
    )

    # 6. Rolling Sharpe
    viz.plot_rolling_sharpe(
        {k: v for k, v in list(results.items())[:4]},
        output_path=OUT / "06_rolling_sharpe.png",
    )

    # 7. Correlation heatmap
    viz.plot_correlation_heatmap(
        monthly, output_path=OUT / "07_correlation_heatmap.png"
    )

    # 8. Regime timeline
    if primary_bt.regime_labels is not None and len(primary_bt.regime_labels) > 0:
        regime_series = primary_bt.regime_labels.reindex(port_rets.index).ffill().fillna(1).astype(int)
        viz.plot_regime_timeline(
            port_rets, regime_series,
            output_path=OUT / "08_regime_timeline.png",
        )

    # 9. Monthly heatmap
    viz.plot_monthly_heatmap(
        port_rets, output_path=OUT / "09_monthly_heatmap.png"
    )

    # 10. Full dashboard
    viz.plot_full_dashboard(
        port_rets, benchmarks, primary_bt.portfolio_weights,
        regimes=primary_bt.regime_labels,
        output_path=OUT / "10_full_dashboard.png",
    )

    logger.info("All plots saved to %s", OUT)


def step_console_summary(comparison: pd.DataFrame) -> None:
    """Print the final results table and best-strategy highlights to stdout."""
    border = "=" * 70
    print(f"\n{border}")
    print("   ML-ENHANCED QUANTITATIVE PORTFOLIO OPTIMIZER - RESULTS")
    print(border)
    print(comparison.to_string())
    print(f"{border}\n")
    best_sharpe = comparison["Sharpe"].astype(float).idxmax()
    best_calmar = comparison["Calmar"].astype(float).idxmax()
    best_cagr   = comparison["CAGR"].astype(float).idxmax()
    print(f"  Best Sharpe : {best_sharpe}")
    print(f"  Best Calmar : {best_calmar}")
    print(f"  Best CAGR   : {best_cagr}")
    print(f"  Output plots -> {OUT}\n")


def step_pdf_report(results, comparison, backtester_objs, forecaster, ar_forecaster, regime_stats, snap_weights):
    """STEP 9: Build the multi-section PDF research report."""
    logger.info("-" * 60)
    logger.info("STEP 9 - Generating PDF research report")
    from reports.report_generator import generate_report
    report_dir = ROOT / cfg.REPORT_DIR
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "portfolio_optimizer_report.pdf"
    primary_name = "MVO + ML + Regime"
    primary_bt = backtester_objs.get(primary_name, list(backtester_objs.values())[0])

    final_weights = {}
    for name, bt in backtester_objs.items():
        if bt.portfolio_weights is not None and len(bt.portfolio_weights) > 0:
            final_weights[name] = bt.portfolio_weights.iloc[-1]
    n = len(cfg.TICKERS)
    final_weights["Equal Weight"] = pd.Series(1.0 / n, index=cfg.TICKERS)
    if "S&P 500" not in final_weights and "SPY" in cfg.TICKERS:
        spy_w = pd.Series(0.0, index=cfg.TICKERS)
        spy_w["SPY"] = 1.0
        final_weights["S&P 500"] = spy_w

    generate_report(
        output_path=report_path,
        plots_dir=OUT,
        results=results,
        comparison_df=comparison,
        weights_history=primary_bt.portfolio_weights,
        forecaster=forecaster,
        ar_forecaster=ar_forecaster,
        regime_stats=regime_stats,
        snap_weights=snap_weights,
        final_weights=final_weights,
        cfg=cfg,
    )
    logger.info("PDF report saved: %s", report_path)
    return report_path

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute all nine pipeline steps end-to-end."""
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   ML-Enhanced Quantitative Portfolio Optimizer           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    monthly, macro            = step_data()
    X, y, fe                  = step_features(monthly, macro)
    forecaster                = step_ml_model(X, y)
    ar_forecaster             = step_ar_model(monthly)
    detector                  = step_regime(monthly)
    snap_weights              = step_optimizer_snapshot(monthly)
    results, backtester_objs  = step_backtest(monthly, macro)
    comparison                = step_analytics(results)
    step_visualisations(results, backtester_objs, monthly, comparison)
    step_console_summary(comparison)
    step_pdf_report(results, comparison, backtester_objs, forecaster, ar_forecaster, detector.regime_stats, snap_weights)


if __name__ == "__main__":
    main()
