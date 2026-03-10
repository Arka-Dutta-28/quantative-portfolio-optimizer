"""
webapp.py
─────────
Interactive Streamlit dashboard for the ML-Enhanced Quantitative Portfolio
Optimizer.

Launch
------
    streamlit run webapp.py

Features
--------
  - Live ticker validation via Yahoo Finance
  - Configurable date range, risk parameters, and covariance method
  - Full pipeline execution with progress bar
  - Seven interactive tabs: Overview, Strategy Comparison, Portfolio Weights,
    Risk Analytics, ML Models (including AR model-selection), Market Regimes,
    and Download (CSV + PDF export)
  - All charts rendered with Plotly for interactivity
"""
from __future__ import annotations

import sys, logging, tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import date

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import config as default_cfg

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from models.return_forecaster import ReturnForecaster
from models.ar_forecaster import ARForecaster
from models.regime_detector import RegimeDetector
from risk.covariance_estimator import CovarianceEstimator
from optimization.portfolio_optimizer import PortfolioOptimizer, OptimizationConfig
from backtesting.backtester import Backtester, BacktestConfig
from analytics.performance_metrics import PerformanceAnalytics

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PAL = ["#2c7bb6", "#d7191c", "#fdae61", "#1a9641", "#756bb1",
       "#f46d43", "#66c2a5", "#e7298a", "#a6761d", "#636363"]

REGIME_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}
REGIME_NAMES_3 = {0: "Bull", 1: "Neutral", 2: "Bear / Crisis"}
REGIME_NAMES_4 = {0: "Bull", 1: "Recovery", 2: "Slowdown", 3: "Bear / Crisis"}

_LY = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Roboto, sans-serif", size=13),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Portfolio Optimizer", page_icon="\U0001F4CA", layout="wide")

st.markdown("""<style>
.block-container {padding-top:1.2rem;}
div[data-testid="stMetric"] {
    background:#1e2130; border:1px solid #3a3d4d; border-radius:8px;
    padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,.25);
}
div[data-testid="stMetric"] label {font-size:.82rem;color:#a0a4b8;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-size:1.35rem;color:#ffffff;}
@media (prefers-color-scheme: light) {
    div[data-testid="stMetric"] {background:#f8f9fb; border-color:#e1e5eb; box-shadow:0 1px 3px rgba(0,0,0,.06);}
    div[data-testid="stMetric"] label {color:#555;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#1a1a2e;}
}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# yfinance ticker validation
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def validate_tickers(tickers: tuple[str, ...], start: str, end: str) -> dict[str, bool]:
    """Check each ticker against yfinance. Returns {ticker: is_valid}."""
    if not _HAS_YF:
        return {t: False for t in tickers}
    result = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            result[t] = df is not None and len(df) >= 10
        except Exception:
            result[t] = False
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(tickers, start, end, risk_free, target_vol, max_weight,
                 max_turnover, tx_cost, cov_method, n_regimes, bar):
    """Run the full optimisation pipeline and return a result dict consumed by all tabs."""
    bar.progress(5, "Loading market data from Yahoo Finance ...")
    loader = DataLoader(tickers, start, end, strict_yfinance=True)
    loader.load()
    monthly = loader.get_monthly_returns()
    macro = loader.get_macro_features()

    bar.progress(15, "Engineering features ...")
    fe = FeatureEngineer(
        lookback_windows=default_cfg.LOOKBACK_WINDOWS,
        vol_windows=[3, 6, 12],
        forecast_horizon=default_cfg.FORECAST_HORIZON,
        scale_features=True,
    )
    X, y = fe.fit_transform(monthly, macro)

    bar.progress(25, "Training ML ensemble ...")
    ml_split = int(len(X) * 0.7)
    forecaster = ReturnForecaster(tickers, default_cfg.RANDOM_STATE)
    forecaster.fit(X.iloc[:ml_split], y.iloc[:ml_split])

    bar.progress(35, "Fitting AR / GARCH models ...")
    ar_split = int(len(monthly) * 0.7)
    ar_forecaster = ARForecaster(tickers, ar_order=2)
    ar_forecaster.fit(monthly.iloc[:ar_split])

    bar.progress(45, "Detecting market regimes ...")
    detector = RegimeDetector(n_regimes=n_regimes, random_state=default_cfg.RANDOM_STATE)
    regime_labels = detector.fit_predict(monthly)
    regime_stats = detector.compute_regime_stats(monthly)
    regime_proba = detector.predict_proba(monthly)

    bar.progress(55, "Running single-period optimisation ...")
    est = CovarianceEstimator(cov_method).fit(monthly)
    mu = monthly.mean().values
    snap_weights = {}
    for strat in ("mvo", "min_vol", "max_sharpe", "risk_parity", "cvar", "equal_weight"):
        oc = OptimizationConfig(
            strategy=strat, max_weight=max_weight, min_weight=0.0,
            vol_target=target_vol / np.sqrt(12), max_turnover=max_turnover,
            risk_free_rate=risk_free / 12,
        )
        snap_weights[strat] = PortfolioOptimizer(tickers, oc).optimize(mu, est.cov.values)

    bar.progress(60, "Computing efficient frontier ...")
    oc_f = OptimizationConfig(max_weight=max_weight, vol_target=10.0)
    frontier = PortfolioOptimizer(tickers, oc_f).efficient_frontier(mu, est.cov.values, n_points=50)
    ind_assets = pd.DataFrame({"vol": monthly.std() * np.sqrt(12),
                                "ret": monthly.mean() * 12}).rename_axis("ticker")

    bar.progress(70, "Walk-forward backtesting ...")
    bt_common = dict(cov_method=cov_method, transaction_cost=tx_cost,
                     max_weight=max_weight, vol_target=target_vol / np.sqrt(12),
                     max_turnover=max_turnover, risk_free_rate=risk_free / 12)
    strat_cfgs = {
        "MVO + ML + Regime": BacktestConfig(strategy="mvo", use_ml_forecasts=True, use_regime_filter=True, **bt_common),
        "MVO (No ML)":       BacktestConfig(strategy="mvo", use_ml_forecasts=False, use_regime_filter=False, **bt_common),
        "Min-Vol":           BacktestConfig(strategy="min_vol", use_ml_forecasts=False, use_regime_filter=False, **bt_common),
        "Max-Sharpe":        BacktestConfig(strategy="max_sharpe", use_ml_forecasts=True, use_regime_filter=False, **bt_common),
        "CVaR":              BacktestConfig(strategy="cvar", use_ml_forecasts=True, use_regime_filter=True, **bt_common),
    }
    bt_results: dict[str, pd.Series] = {}
    bt_objects: dict[str, Backtester] = {}
    for name, bcfg in strat_cfgs.items():
        bt = Backtester(monthly, bcfg, macro=macro)
        bt.run()
        bt_results[name] = bt.portfolio_returns
        bt_objects[name] = bt

    primary_bt = bt_objects["MVO + ML + Regime"]
    bt_results["Equal Weight"] = primary_bt.get_benchmark_returns("equal_weight")
    bt_results["Risk Parity"] = primary_bt.get_benchmark_returns("risk_parity")
    bt_results["S&P 500"] = primary_bt.get_benchmark_returns("spy_only")

    bar.progress(90, "Computing analytics ...")
    comparison = PerformanceAnalytics.compare(bt_results, benchmark=bt_results.get("S&P 500"),
                                              risk_free=risk_free)

    final_weights: dict[str, pd.Series] = {}
    for name, bt in bt_objects.items():
        if bt.portfolio_weights is not None and len(bt.portfolio_weights) > 0:
            final_weights[name] = bt.portfolio_weights.iloc[-1]
    n = len(tickers)
    final_weights["Equal Weight"] = pd.Series(1.0 / n, index=tickers)
    if "SPY" in tickers:
        spy_w = pd.Series(0.0, index=tickers); spy_w["SPY"] = 1.0
        final_weights["S&P 500"] = spy_w

    bar.progress(100, "Done!")
    return dict(
        tickers=tickers, start=start, end=end, risk_free=risk_free,
        monthly=monthly, macro=macro,
        forecaster=forecaster, ar_forecaster=ar_forecaster,
        detector=detector, regime_labels=regime_labels,
        regime_stats=regime_stats, regime_proba=regime_proba,
        snap_weights=snap_weights, frontier=frontier, ind_assets=ind_assets,
        results=bt_results, bt_objects=bt_objects,
        comparison=comparison, final_weights=final_weights,
        weights_history=primary_bt.portfolio_weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart helpers
# ─────────────────────────────────────────────────────────────────────────────
def _color(i):
    """Return a palette colour by index (wraps around)."""
    return PAL[i % len(PAL)]


def chart_cumulative(results):
    """Plotly line chart of cumulative returns for all strategies."""
    fig = go.Figure()
    for i, (name, rets) in enumerate(results.items()):
        cum = (1 + rets).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name,
                                 line=dict(color=_color(i), width=2.2)))
    fig.update_layout(title="Cumulative Portfolio Returns", yaxis_title="Growth of $1",
                      yaxis_tickformat=".2f", **_LY)
    return fig


def chart_drawdown(results):
    """Plotly filled-area drawdown chart for all strategies."""
    fig = go.Figure()
    for i, (name, rets) in enumerate(results.items()):
        cum = (1 + rets).cumprod()
        dd = cum / cum.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name,
                                 fill="tozeroy", line=dict(color=_color(i), width=1.5),
                                 fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(_color(i))) + [0.15])}"))
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown",
                      yaxis_tickformat=".0%", **_LY)
    return fig


def chart_rolling_sharpe(results, window=12, rf_m=0.04/12):
    """Plotly rolling Sharpe ratio line chart."""
    fig = go.Figure()
    for i, (name, rets) in enumerate(results.items()):
        excess = rets - rf_m
        roll = excess.rolling(window).mean() / rets.rolling(window).std() * np.sqrt(12)
        roll = roll.dropna()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name=name,
                                 line=dict(color=_color(i), width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title=f"Rolling {window}-Month Sharpe Ratio", yaxis_title="Sharpe",
                      yaxis_tickformat=".2f", **_LY)
    return fig


def chart_frontier(frontier, ind_assets, tickers):
    """Plotly efficient frontier scatter with individual asset markers."""
    hover = []
    for _, r in frontier.iterrows():
        parts = [f"{t}: {r[t]:.1%}" for t in tickers if t in r and r[t] > 0.01]
        hover.append(f"Ret: {r['ret']:.1%}<br>Vol: {r['vol']:.1%}<br>Sharpe: {r['sharpe']:.2f}<br>{'<br>'.join(parts)}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["vol"], y=frontier["ret"], mode="markers",
        marker=dict(color=frontier["sharpe"], colorscale="RdYlGn", size=9,
                    colorbar=dict(title="Sharpe", len=0.6)),
        text=hover, hoverinfo="text", name="Frontier"))
    fig.add_trace(go.Scatter(
        x=ind_assets["vol"], y=ind_assets["ret"], mode="markers+text",
        text=ind_assets.index, textposition="top center",
        marker=dict(color="#c0392b", size=11, symbol="diamond"), name="Assets"))
    fig.update_layout(title="Efficient Frontier", xaxis_title="Annualised Volatility",
                      yaxis_title="Annualised Return", xaxis_tickformat=".1%",
                      yaxis_tickformat=".1%", **_LY)
    return fig


def chart_allocation(weights_history):
    """Plotly stacked-area chart of portfolio allocation over time."""
    fig = go.Figure()
    for col in weights_history.columns:
        fig.add_trace(go.Scatter(
            x=weights_history.index, y=weights_history[col],
            name=col, stackgroup="one", mode="lines",
            line=dict(width=0.4)))
    fig.update_layout(title="Portfolio Allocation Over Time", yaxis_title="Weight",
                      yaxis_tickformat=".0%", yaxis_range=[0, 1], **_LY)
    return fig


def chart_return_dist(returns, risk_free=0.04):
    """Plotly histogram of monthly returns with VaR/CVaR markers."""
    pa = PerformanceAnalytics(returns, risk_free=risk_free)
    var95 = pa.var(0.05)
    cvar95 = pa.cvar(0.05)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=40, name="Returns",
                                marker_color="#2c7bb6", opacity=0.75))
    fig.add_vline(x=var95, line_dash="dash", line_color="#d7191c",
                  annotation_text=f"VaR 95%: {var95:.2%}")
    fig.add_vline(x=cvar95, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"CVaR 95%: {cvar95:.2%}")
    fig.add_vline(x=returns.mean(), line_dash="dot", line_color="#1a9641",
                  annotation_text=f"Mean: {returns.mean():.2%}")
    fig.update_layout(title="Monthly Return Distribution", xaxis_title="Return",
                      yaxis_title="Count", xaxis_tickformat=".1%", **_LY)
    return fig


def chart_monthly_heatmap(returns):
    """Plotly Year x Month heatmap of returns."""
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    df = pd.DataFrame({"ret": r.values}, index=r.index)
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    pivot = df.pivot_table(values="ret", index="Year", columns="Month", aggfunc="sum")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [months[m - 1] for m in pivot.columns]
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                    labels=dict(color="Return"), text_auto=".1%")
    fig.update_layout(title="Monthly Returns Heatmap", **_LY)
    return fig


def chart_regime_timeline(returns, regimes):
    """Plotly cumulative return line with regime-shaded background."""
    cum = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Cumulative",
                             line=dict(color="#2c7bb6", width=2)))
    start_idx = regimes.index[0]
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != regimes.iloc[i - 1] or i == len(regimes) - 1:
            r = regimes.iloc[i - 1]
            end_idx = regimes.index[i]
            fig.add_vrect(x0=start_idx, x1=end_idx,
                          fillcolor=REGIME_COLORS.get(r, "gray"), opacity=0.15,
                          line_width=0, layer="below")
            start_idx = regimes.index[i]
    fig.update_layout(title="Cumulative Returns with Regime Overlay",
                      yaxis_title="Growth of $1", yaxis_tickformat=".2f", **_LY)
    return fig


def chart_regime_proba(regime_proba):
    """Plotly stacked-area chart of regime posterior probabilities."""
    fig = go.Figure()
    for i, col in enumerate(regime_proba.columns):
        fig.add_trace(go.Scatter(
            x=regime_proba.index, y=regime_proba[col], name=str(col),
            stackgroup="one", mode="lines",
            line=dict(width=0.5, color=REGIME_COLORS.get(i, _color(i)))))
    fig.update_layout(title="Regime Probability Over Time", yaxis_title="Probability",
                      yaxis_range=[0, 1], yaxis_tickformat=".0%", **_LY)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────────────────────────────────────
def render_overview(D):
    """Render the *Overview* tab: KPI cards, cumulative chart, strategy table."""
    comp = D["comparison"]
    best_sharpe = comp["Sharpe"].astype(float).idxmax()
    best_cagr = comp["CAGR"].astype(float).idxmax()
    row = comp.loc[best_sharpe]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Strategy (Sharpe)", best_sharpe)
    c2.metric("CAGR", f"{float(comp.loc[best_sharpe, 'CAGR']):.1%}")
    c3.metric("Sharpe Ratio", f"{float(row['Sharpe']):.2f}")
    c4.metric("Annual Vol", f"{float(row['Vol']):.1%}")
    c5.metric("Max Drawdown", f"{float(row['Max Drawdown']):.1%}")

    st.plotly_chart(chart_cumulative(D["results"]), use_container_width=True)

    st.subheader("Strategy Comparison")
    pct_cols = ["CAGR", "Vol", "Max Drawdown", "CVaR 95%", "Total Return"]
    flt_cols = ["Sharpe", "Sortino", "Calmar"]
    fmt = {c: "{:.1%}" for c in pct_cols if c in comp.columns}
    fmt.update({c: "{:.2f}" for c in flt_cols if c in comp.columns})
    good_cols = [c for c in ["Sharpe", "CAGR", "Sortino", "Calmar", "Total Return"] if c in comp.columns]
    bad_cols = [c for c in ["Vol", "Max Drawdown", "CVaR 95%"] if c in comp.columns]
    styled = comp.style.format(fmt)
    if good_cols:
        styled = styled.background_gradient(cmap="RdYlGn", subset=good_cols)
    if bad_cols:
        styled = styled.background_gradient(cmap="RdYlGn_r", subset=bad_cols)
    st.dataframe(styled, use_container_width=True, height=340)


def render_comparison(D):
    """Render the *Strategy Comparison* tab: bar charts, drawdown, rolling Sharpe."""
    comp = D["comparison"]

    st.subheader("Key Metrics Comparison")
    metrics = ["Sharpe", "CAGR", "Calmar", "Sortino"]
    available = [m for m in metrics if m in comp.columns]
    cols = st.columns(len(available))
    for i, m in enumerate(available):
        vals = comp[m].astype(float)
        fig = go.Figure(go.Bar(
            x=vals.index, y=vals.values,
            marker_color=[PAL[j % len(PAL)] for j in range(len(vals))],
            text=[f"{v:.2f}" if m != "CAGR" else f"{v:.1%}" for v in vals],
            textposition="outside"))
        fig.update_layout(title=m, yaxis_title=m, showlegend=False,
                          margin=dict(l=40, r=20, t=40, b=80), height=350,
                          template="plotly_white",
                          font=dict(family="Segoe UI, sans-serif", size=11))
        cols[i].plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_drawdown(D["results"]), use_container_width=True)
    with right:
        st.plotly_chart(chart_rolling_sharpe(D["results"], rf_m=D["risk_free"] / 12),
                        use_container_width=True)


def render_weights(D):
    """Render the *Portfolio Weights* tab: pie chart, grouped bar, allocation, frontier."""
    tickers = D["tickers"]
    fw = D["final_weights"]
    strat_names = list(fw.keys())

    sel = st.selectbox("Select strategy", strat_names, index=0)
    ws = fw[sel]

    c1, c2 = st.columns(2)
    with c1:
        labels = [t for t in tickers if t in ws.index and ws[t] > 0.005]
        values = [ws[t] for t in labels]
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45,
                                marker_colors=PAL[:len(labels)],
                                textinfo="label+percent", textposition="outside"))
        fig.update_layout(title=f"{sel} — Final Weights", showlegend=False,
                          margin=dict(l=20, r=20, t=50, b=20), height=420,
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        df_bar = pd.DataFrame(fw).reindex(tickers).fillna(0)
        fig = go.Figure()
        for i, col in enumerate(df_bar.columns):
            fig.add_trace(go.Bar(name=col, x=df_bar.index, y=df_bar[col],
                                 marker_color=_color(i)))
        fig.update_layout(barmode="group", title="All Strategies — Weight Comparison",
                          yaxis_title="Weight", yaxis_tickformat=".0%",
                          height=420, template="plotly_white",
                          font=dict(family="Segoe UI, sans-serif", size=12),
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                          margin=dict(l=50, r=20, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(chart_allocation(D["weights_history"]), use_container_width=True)
    st.plotly_chart(chart_frontier(D["frontier"], D["ind_assets"], tickers),
                    use_container_width=True)


def render_risk(D):
    """Render the *Risk Analytics* tab: VaR, CVaR, drawdown, heatmap, full summary."""
    strat_names = list(D["results"].keys())
    sel = st.selectbox("Select strategy for risk analysis", strat_names, index=0, key="risk_sel")
    rets = D["results"][sel]
    pa = PerformanceAnalytics(rets, risk_free=D["risk_free"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("VaR 95%", f"{pa.var(0.05):.2%}")
    c2.metric("CVaR 95%", f"{pa.cvar(0.05):.2%}")
    c3.metric("Max Drawdown", f"{pa.max_drawdown():.2%}")
    c4.metric("Sortino", f"{pa.sortino():.2f}")
    c5.metric("Calmar", f"{pa.calmar():.2f}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_return_dist(rets, D["risk_free"]), use_container_width=True)
    with right:
        cum = (1 + rets).cumprod()
        dd = cum / cum.cummax() - 1
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                                    line=dict(color="#c0392b", width=1.5),
                                    fillcolor="rgba(192,57,43,0.2)"))
        fig.update_layout(title=f"Drawdown — {sel}", yaxis_title="Drawdown",
                          yaxis_tickformat=".0%", **_LY)
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Top Drawdown Periods")
        try:
            ddt = pa.drawdown_table(top_n=5)
            if not ddt.empty:
                for c in ["Start", "Trough", "Recovery"]:
                    if c in ddt.columns:
                        ddt[c] = pd.to_datetime(ddt[c], errors="coerce").dt.strftime("%Y-%m")
                if "Depth" in ddt.columns:
                    ddt["Depth"] = ddt["Depth"].apply(
                        lambda x: f"{float(x):.2%}" if str(x) != "nan" else "-")
                if "Duration" in ddt.columns:
                    ddt["Duration"] = ddt["Duration"].apply(
                        lambda x: f"{int(float(x))}d" if str(x) != "nan" else "-")
                st.dataframe(ddt, use_container_width=True)
        except Exception:
            st.caption("Drawdown table unavailable.")
    with right2:
        st.plotly_chart(chart_monthly_heatmap(rets), use_container_width=True)

    st.subheader("Full Risk Summary")
    summary = pa.summary()
    st.dataframe(summary, use_container_width=True)


def render_ml(D):
    """Render the *ML Models* tab: ensemble metrics, per-model breakdown, AR selection, ML vs AR comparison."""
    fc = D["forecaster"]
    ar_fc = D["ar_forecaster"]

    st.subheader("ML Ensemble — Per-Asset Metrics")
    try:
        ml_metrics = fc.metrics_summary()
        if not ml_metrics.empty:
            fmt_ml = {}
            for c in ml_metrics.columns:
                if c == "dir_accuracy":
                    fmt_ml[c] = "{:.1%}"
                elif ml_metrics[c].dtype in (np.float64, float):
                    fmt_ml[c] = "{:.5f}"
            styled_ml = ml_metrics.style.format(fmt_ml)
            if "dir_accuracy" in ml_metrics.columns:
                styled_ml = styled_ml.background_gradient(cmap="RdYlGn", subset=["dir_accuracy"])
            if "ic" in ml_metrics.columns:
                styled_ml = styled_ml.background_gradient(cmap="RdYlGn", subset=["ic"])
            st.dataframe(styled_ml, use_container_width=True)
    except Exception as e:
        st.caption(f"ML metrics unavailable: {e}")

    st.subheader("ML Per-Model Breakdown")
    try:
        pm = fc.per_model_summary()
        if pm is not None and not pm.empty:
            suffixes = ["_val_rmse", "_dir_acc", "_ic", "_weight"]
            model_names, seen = [], set()
            for col in pm.columns:
                for sfx in suffixes:
                    if col.endswith(sfx):
                        mn = col[:-len(sfx)]
                        if mn not in seen:
                            model_names.append(mn); seen.add(mn)
            for mn in model_names:
                cols = [c for c in pm.columns if c.startswith(mn + "_")]
                if not cols:
                    continue
                sub = pm[cols].rename(columns={c: c[len(mn)+1:] for c in cols})
                st.markdown(f"**{mn.replace('_', ' ').title()}**")
                st.dataframe(sub.style.format("{:.4f}"), use_container_width=True)
    except Exception as e:
        st.caption(f"Per-model data unavailable: {e}")

    st.markdown("---")
    st.subheader("AR / GARCH Ensemble — Per-Asset Metrics")
    try:
        ar_metrics = ar_fc.metrics_summary()
        if not ar_metrics.empty:
            fmt_ar = {}
            for c in ar_metrics.columns:
                if c == "dir_accuracy":
                    fmt_ar[c] = "{:.1%}"
                elif ar_metrics[c].dtype in (np.float64, float):
                    fmt_ar[c] = "{:.5f}"
            styled_ar = ar_metrics.style.format(fmt_ar)
            if "dir_accuracy" in ar_metrics.columns:
                styled_ar = styled_ar.background_gradient(cmap="RdYlGn", subset=["dir_accuracy"])
            if "ic" in ar_metrics.columns:
                styled_ar = styled_ar.background_gradient(cmap="RdYlGn", subset=["ic"])
            st.dataframe(styled_ar, use_container_width=True)
    except Exception as e:
        st.caption(f"AR metrics unavailable: {e}")

    st.subheader("AR Per-Model Breakdown")
    try:
        ar_pm = ar_fc.per_model_summary()
        if ar_pm is not None and not ar_pm.empty:
            suffixes = ["_val_rmse", "_dir_acc", "_ic", "_weight"]
            ar_models, seen = [], set()
            for col in ar_pm.columns:
                for sfx in suffixes:
                    if col.endswith(sfx):
                        mn = col[:-len(sfx)]
                        if mn not in seen:
                            ar_models.append(mn); seen.add(mn)
            for mn in ar_models:
                cols = [c for c in ar_pm.columns if c.startswith(mn + "_")]
                if not cols:
                    continue
                sub = ar_pm[cols].rename(columns={c: c[len(mn)+1:] for c in cols})
                st.markdown(f"**{mn}**")
                st.dataframe(sub.style.format("{:.4f}"), use_container_width=True)
    except Exception as e:
        st.caption(f"AR per-model data unavailable: {e}")

    st.markdown("---")
    st.subheader("AR Model Selection (BIC Auto-Selection)")
    st.markdown(
        "For each asset, multiple candidate orders are fitted per model family. "
        "The best order is selected by **BIC** (Bayesian Information Criterion — lower is better)."
    )
    try:
        sel_df = ar_fc.selection_summary()
        if sel_df is not None and not sel_df.empty:
            display_cols = [c for c in sel_df.columns
                           if c.endswith("_selected") or c.endswith("_bic")]
            if display_cols:
                sel_display = sel_df[display_cols].copy()
                rename = {}
                for c in sel_display.columns:
                    parts = c.rsplit("_", 1)
                    if len(parts) == 2:
                        rename[c] = f"{parts[0]} {parts[1].upper()}"
                sel_display = sel_display.rename(columns=rename)
                fmt_sel = {c: "{:.1f}" for c in sel_display.columns if "BIC" in c}
                st.dataframe(sel_display.style.format(fmt_sel), use_container_width=True)
    except Exception as e:
        st.caption(f"Model selection data unavailable: {e}")

    st.markdown("---")
    st.subheader("ML vs AR — Head-to-Head Comparison")
    try:
        ml_m = fc.metrics_summary()
        ar_m = ar_fc.metrics_summary()
        common = ml_m.index.intersection(ar_m.index)
        if len(common) > 0:
            cmp = pd.DataFrame({
                "ML Dir Acc": ml_m.loc[common, "dir_accuracy"].astype(float),
                "AR Dir Acc": ar_m.loc[common, "dir_accuracy"].astype(float),
                "ML IC": ml_m.loc[common, "ic"].astype(float),
                "AR IC": ar_m.loc[common, "ic"].astype(float),
            })
            cmp["Winner (Dir Acc)"] = np.where(
                cmp["ML Dir Acc"] >= cmp["AR Dir Acc"], "ML", "AR")
            cmp["Winner (IC)"] = np.where(
                cmp["ML IC"] >= cmp["AR IC"], "ML", "AR")

            ml_da_w = int((cmp["Winner (Dir Acc)"] == "ML").sum())
            ar_da_w = int((cmp["Winner (Dir Acc)"] == "AR").sum())
            ml_ic_w = int((cmp["Winner (IC)"] == "ML").sum())
            ar_ic_w = int((cmp["Winner (IC)"] == "AR").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ML Wins (Dir Acc)", f"{ml_da_w} / {len(cmp)}")
            c2.metric("AR Wins (Dir Acc)", f"{ar_da_w} / {len(cmp)}")
            c3.metric("ML Wins (IC)", f"{ml_ic_w} / {len(cmp)}")
            c4.metric("AR Wins (IC)", f"{ar_ic_w} / {len(cmp)}")

            left, right = st.columns(2)
            with left:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="ML", x=list(common),
                                     y=cmp["ML Dir Acc"].values,
                                     marker_color="#2c7bb6"))
                fig.add_trace(go.Bar(name="AR", x=list(common),
                                     y=cmp["AR Dir Acc"].values,
                                     marker_color="#d7191c"))
                fig.update_layout(barmode="group",
                                  title="Directional Accuracy by Asset",
                                  yaxis_title="Dir Accuracy",
                                  yaxis_tickformat=".0%", height=400, **_LY)
                st.plotly_chart(fig, use_container_width=True)
            with right:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="ML", x=list(common),
                                     y=cmp["ML IC"].values,
                                     marker_color="#2c7bb6"))
                fig.add_trace(go.Bar(name="AR", x=list(common),
                                     y=cmp["AR IC"].values,
                                     marker_color="#d7191c"))
                fig.update_layout(barmode="group",
                                  title="Information Coefficient by Asset",
                                  yaxis_title="IC (Spearman)",
                                  yaxis_tickformat=".3f", height=400, **_LY)
                st.plotly_chart(fig, use_container_width=True)

            styled_cmp = cmp.style.format({
                "ML Dir Acc": "{:.1%}", "AR Dir Acc": "{:.1%}",
                "ML IC": "{:.4f}", "AR IC": "{:.4f}",
            })
            st.dataframe(styled_cmp, use_container_width=True)
    except Exception as e:
        st.caption(f"Comparison unavailable: {e}")


def render_regimes(D):
    """Render the *Market Regimes* tab: timeline, probabilities, tilt table."""
    regime_labels = D["regime_labels"]
    regime_stats = D["regime_stats"]
    regime_proba = D["regime_proba"]
    results = D["results"]

    primary_name = "MVO + ML + Regime"
    rets = results.get(primary_name, list(results.values())[0])

    aligned = regime_labels.reindex(rets.index).ffill().fillna(1).astype(int)
    st.plotly_chart(chart_regime_timeline(rets, aligned), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Regime Statistics")
        if regime_stats is not None and not regime_stats.empty:
            st.dataframe(regime_stats.style.format("{:.3f}"), use_container_width=True)
    with right:
        st.plotly_chart(chart_regime_proba(regime_proba), use_container_width=True)

    st.subheader("Allocation Tilts by Regime")
    n_reg = D.get("detector").n_regimes if D.get("detector") else 3
    if n_reg >= 4:
        tilt_data = {
            "Asset Class": ["Equity", "Bond", "Commodity", "Real Estate"],
            "Bull (x)": [1.30, 0.70, 1.00, 1.10],
            "Recovery (x)": [1.10, 0.90, 1.10, 1.00],
            "Slowdown (x)": [0.90, 1.10, 1.00, 0.90],
            "Bear / Crisis (x)": [0.60, 1.40, 1.30, 0.70],
        }
    else:
        tilt_data = {
            "Asset Class": ["Equity", "Bond", "Commodity", "Real Estate"],
            "Bull (x)": [1.30, 0.70, 1.00, 1.10],
            "Neutral (x)": [1.00, 1.00, 1.00, 1.00],
            "Bear / Crisis (x)": [0.60, 1.40, 1.30, 0.70],
        }
    tilts = pd.DataFrame(tilt_data).set_index("Asset Class")
    st.dataframe(tilts.style.format("{:.2f}").background_gradient(
        cmap="RdYlGn", axis=1), use_container_width=True)


def render_download(D):
    """Render the *Download* tab: CSV exports and PDF report generation."""
    st.subheader("Download Comparison Metrics")
    csv_comp = D["comparison"].to_csv()
    st.download_button("Download Strategy Comparison (CSV)", csv_comp,
                       "strategy_comparison.csv", "text/csv")

    st.subheader("Download Monthly Returns")
    csv_monthly = D["monthly"].to_csv()
    st.download_button("Download Monthly Returns (CSV)", csv_monthly,
                       "monthly_returns.csv", "text/csv")

    st.subheader("Download Final Weights")
    fw_df = pd.DataFrame(D["final_weights"]).fillna(0)
    csv_fw = fw_df.to_csv()
    st.download_button("Download Final Weights (CSV)", csv_fw,
                       "final_weights.csv", "text/csv")

    st.markdown("---")
    st.subheader("Generate PDF Research Report")
    st.caption("Generates the full PDF report with all charts and tables.")
    if st.button("Generate PDF Report"):
        with st.spinner("Generating matplotlib charts and PDF ..."):
            try:
                pdf_bytes = _generate_pdf(D)
                st.download_button("Download PDF Report", pdf_bytes,
                                   "portfolio_optimizer_report.pdf",
                                   "application/pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


def _build_cfg_ns(D):
    """Build a SimpleNamespace mimicking the ``config`` module for report generation."""
    return SimpleNamespace(
        TICKERS=D["tickers"],
        ASSETS=default_cfg.ASSETS,
        ASSET_CLASSES=default_cfg.ASSET_CLASSES,
        START_DATE=D["start"],
        END_DATE=D["end"],
        LOOKBACK_WINDOWS=default_cfg.LOOKBACK_WINDOWS,
        MOMENTUM_WINDOWS=default_cfg.MOMENTUM_WINDOWS,
        FORECAST_HORIZON=default_cfg.FORECAST_HORIZON,
        TRAIN_YEARS=default_cfg.TRAIN_YEARS,
        VALIDATION_SPLIT=default_cfg.VALIDATION_SPLIT,
        RANDOM_STATE=default_cfg.RANDOM_STATE,
        COVARIANCE_METHOD=default_cfg.COVARIANCE_METHOD,
        EWM_HALFLIFE=default_cfg.EWM_HALFLIFE,
        RISK_FREE_RATE=D["risk_free"],
        TARGET_VOLATILITY=default_cfg.TARGET_VOLATILITY,
        MAX_WEIGHT=default_cfg.MAX_WEIGHT,
        MIN_WEIGHT=default_cfg.MIN_WEIGHT,
        MAX_TURNOVER=default_cfg.MAX_TURNOVER,
        TRANSACTION_COST=default_cfg.TRANSACTION_COST,
        CVAR_ALPHA=default_cfg.CVAR_ALPHA,
        N_SCENARIOS=default_cfg.N_SCENARIOS,
        N_REGIMES=default_cfg.N_REGIMES,
        REBALANCE_FREQ=default_cfg.REBALANCE_FREQ,
    )


def _generate_pdf(D):
    """Generate all matplotlib charts into a temp dir, then build and return the PDF bytes."""
    tmpdir = Path(tempfile.mkdtemp())
    from visualization import plots as viz

    results = D["results"]
    monthly = D["monthly"]
    primary_name = "MVO + ML + Regime"
    primary_rets = results.get(primary_name, list(results.values())[0])
    benchmarks = {k: v for k, v in results.items() if k != primary_name}

    viz.plot_cumulative_returns(results, output_path=tmpdir / "01_cumulative_returns.png")
    viz.plot_allocation_over_time(D["weights_history"],
                                  output_path=tmpdir / "02_allocation_over_time.png")

    tickers = D["tickers"]
    est = CovarianceEstimator(default_cfg.COVARIANCE_METHOD).fit(monthly)
    mu = monthly.mean().values
    oc = OptimizationConfig(max_weight=default_cfg.MAX_WEIGHT, vol_target=10.0)
    frontier = PortfolioOptimizer(tickers, oc).efficient_frontier(mu, est.cov.values, n_points=50)
    ind_assets = pd.DataFrame({"vol": monthly.std() * np.sqrt(12),
                                "ret": monthly.mean() * 12}).rename_axis("ticker")
    viz.plot_efficient_frontier(frontier, individual_assets=ind_assets,
                                output_path=tmpdir / "03_efficient_frontier.png")
    viz.plot_drawdown({k: v for k, v in list(results.items())[:4]},
                      output_path=tmpdir / "04_drawdown.png")
    viz.plot_return_distribution(primary_rets,
                                 output_path=tmpdir / "05_return_distribution.png")
    viz.plot_rolling_sharpe({k: v for k, v in list(results.items())[:4]},
                            output_path=tmpdir / "06_rolling_sharpe.png")
    viz.plot_correlation_heatmap(monthly,
                                 output_path=tmpdir / "07_correlation_heatmap.png")

    regime_labels = D["regime_labels"]
    aligned = regime_labels.reindex(primary_rets.index).ffill().fillna(1).astype(int)
    if len(aligned) > 0:
        viz.plot_regime_timeline(primary_rets, aligned,
                                 output_path=tmpdir / "08_regime_timeline.png")
    viz.plot_monthly_heatmap(primary_rets,
                             output_path=tmpdir / "09_monthly_heatmap.png")

    bt_obj = D["bt_objects"].get(primary_name, list(D["bt_objects"].values())[0])
    viz.plot_full_dashboard(primary_rets, benchmarks, D["weights_history"],
                            regimes=bt_obj.regime_labels,
                            output_path=tmpdir / "10_full_dashboard.png")

    from reports.report_generator import generate_report
    cfg_ns = _build_cfg_ns(D)
    pdf_path = tmpdir / "report.pdf"
    generate_report(
        output_path=pdf_path, plots_dir=tmpdir,
        results=results, comparison_df=D["comparison"],
        weights_history=D["weights_history"],
        forecaster=D["forecaster"], ar_forecaster=D["ar_forecaster"],
        regime_stats=D["regime_stats"], snap_weights=D["snap_weights"],
        final_weights=D["final_weights"], cfg=cfg_ns,
    )
    return pdf_path.read_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar():
    """Render the sidebar configuration form and return the collected parameters."""
    st.sidebar.markdown("## Configuration")
    st.sidebar.info(
        "This app fetches live data from **Yahoo Finance**. "
        "Enter any ticker available on [finance.yahoo.com](https://finance.yahoo.com). "
        "Examples: `AAPL`, `NIFTYBEES.NS`, `TCS.NS`, `SPY`"
    )

    with st.sidebar.form("cfg_form"):
        st.markdown("**Asset Universe**")
        presets = list(default_cfg.ASSETS.keys())
        preset_tickers = st.multiselect(
            "Quick-pick presets", presets, default=presets,
            help="Pre-configured US ETFs — deselect any you don't want",
        )
        custom_raw = st.text_input(
            "Add custom tickers (comma-separated)",
            placeholder="e.g. NIFTYBEES.NS, TCS.NS, RELIANCE.NS",
            help="Any Yahoo Finance ticker. Indian NSE tickers end with .NS, BSE with .BO",
        )

        st.markdown("**Date Range**")
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", value=date(2013, 1, 1))
        end = c2.date_input("End", value=date(2023, 12, 31))

        st.markdown("**Risk Parameters**")
        risk_free = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.5) / 100
        target_vol = st.slider("Target Volatility (%)", 3.0, 30.0, 10.0, 1.0) / 100
        max_weight = st.slider("Max Single-Asset Weight (%)", 10.0, 100.0, 35.0, 5.0) / 100
        max_turnover = st.slider("Max Turnover / Rebalance (%)", 10.0, 100.0, 50.0, 5.0) / 100
        tx_bps = st.slider("Transaction Cost (bps)", 1, 50, 10, 1)

        st.markdown("**Model Settings**")
        cov_method = st.selectbox("Covariance Method",
                                  ["ledoit_wolf", "sample", "ewm"], index=0)
        n_regimes = st.selectbox("Number of Regimes", [2, 3, 4], index=1)

        submitted = st.form_submit_button("\U0001F680 Run Pipeline",
                                          type="primary", use_container_width=True)

    custom_tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    all_tickers = list(dict.fromkeys(preset_tickers + custom_tickers))

    return dict(submitted=submitted, tickers=all_tickers,
                start=str(start), end=str(end),
                risk_free=risk_free, target_vol=target_vol,
                max_weight=max_weight, max_turnover=max_turnover,
                tx_cost=tx_bps / 10_000, cov_method=cov_method,
                n_regimes=n_regimes)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Application entry point — sidebar, ticker validation, pipeline execution, and tab rendering."""
    st.title("ML-Enhanced Quantitative Portfolio Optimizer")
    sb = build_sidebar()

    if sb["submitted"]:
        tickers = sb["tickers"]
        if len(tickers) < 2:
            st.error("Select at least **2 tickers** to build a portfolio.")
            return

        if not _HAS_YF:
            st.error(
                "`yfinance` is not installed. Install it with `pip install yfinance` "
                "and restart the app."
            )
            return

        with st.spinner("Validating tickers against Yahoo Finance ..."):
            validity = validate_tickers(tuple(tickers), sb["start"], sb["end"])

        invalid = [t for t, ok in validity.items() if not ok]
        if invalid:
            st.error(
                f"**{len(invalid)} ticker(s) not found on Yahoo Finance:** "
                f"`{'`, `'.join(invalid)}`"
            )
            st.warning(
                "All tickers must return at least 10 trading days of data from Yahoo Finance. "
                "Please check spelling, date range, and exchange suffix "
                "(e.g. `.NS` for NSE India, `.BO` for BSE India, `.L` for London)."
            )
            with st.expander("Validation details"):
                for t, ok in validity.items():
                    icon = "\u2705" if ok else "\u274C"
                    st.write(f"{icon}  **{t}** — {'available' if ok else 'NOT FOUND'}")
            return

        st.success(f"All {len(tickers)} tickers validated on Yahoo Finance.")
        bar = st.progress(0, "Initializing ...")
        try:
            D = run_pipeline(
                tickers, sb["start"], sb["end"], sb["risk_free"],
                sb["target_vol"], sb["max_weight"], sb["max_turnover"],
                sb["tx_cost"], sb["cov_method"], sb["n_regimes"], bar,
            )
        except Exception as e:
            bar.empty()
            st.error(f"Pipeline failed: {e}")
            return
        bar.empty()
        st.session_state["D"] = D
        st.session_state["run_cfg"] = {k: v for k, v in sb.items() if k != "submitted"}

    if "D" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Pipeline** to begin.")
        st.markdown("""
### What this app does
- Loads **live price data** from [Yahoo Finance](https://finance.yahoo.com) for any tickers you choose
- Engineers 90+ features (momentum, volatility, mean-reversion, macro)
- Trains an ML ensemble (Ridge, Random Forest, XGBoost) for return forecasting
- Fits AR / GARCH time-series models per asset with BIC-based order auto-selection
- Detects market regimes (Bull / Neutral / Bear) via Gaussian Mixture Model
- Runs 6 portfolio optimisation strategies with walk-forward backtesting
- Computes comprehensive performance and risk analytics

### Supported tickers
Use any ticker from Yahoo Finance. Examples:
- **US:** `SPY`, `QQQ`, `AAPL`, `MSFT`, `TLT`, `GLD`
- **India (NSE):** `NIFTYBEES.NS`, `TCS.NS`, `RELIANCE.NS`, `GOLDBEES.NS`, `BANKBEES.NS`
- **India (BSE):** `TCS.BO`, `RELIANCE.BO`
- **UK:** `VUKE.L`, `ISF.L`
- **Europe:** `DAX`, `CAC.PA`
        """)
        return

    D = st.session_state["D"]
    if "run_cfg" in st.session_state:
        rc = st.session_state["run_cfg"]
        st.caption(f"Results for **{', '.join(rc['tickers'])}** | "
                   f"{rc['start']} to {rc['end']} | "
                   f"Risk-free {rc['risk_free']:.1%} | "
                   f"Vol target {rc['target_vol']:.0%}")

    tabs = st.tabs(["Overview", "Strategy Comparison", "Portfolio Weights",
                     "Risk Analytics", "ML Models", "Market Regimes", "Download"])
    with tabs[0]:
        render_overview(D)
    with tabs[1]:
        render_comparison(D)
    with tabs[2]:
        render_weights(D)
    with tabs[3]:
        render_risk(D)
    with tabs[4]:
        render_ml(D)
    with tabs[5]:
        render_regimes(D)
    with tabs[6]:
        render_download(D)


if __name__ == "__main__":
    main()
