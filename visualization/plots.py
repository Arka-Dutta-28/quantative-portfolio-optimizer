"""
visualization/plots.py
───────────────────────
All visualisations for the portfolio optimizer project.

Plots generated:
  1.  Cumulative returns comparison
  2.  Portfolio allocation over time (stacked area)
  3.  Efficient frontier
  4.  Drawdown curves
  5.  Return distribution (histogram + KDE)
  6.  Rolling Sharpe ratio
  7.  Rolling volatility
  8.  Asset correlation heatmap
  9.  Regime classification timeline
  10. Risk contribution breakdown
  11. Monthly returns heatmap
  12. Performance metrics comparison table
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# ── Style constants ────────────────────────────────────────────────────────────
PALETTE = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61", "#7b3294",
           "#d01c8b", "#4dac26", "#b8e186"]
REGIME_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}
REGIME_NAMES  = {0: "Bull", 1: "Neutral", 2: "Bear/Crisis"}
REGIME_NAMES_4 = {0: "Bull", 1: "Recovery", 2: "Slowdown", 3: "Bear/Crisis"}

plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#a0a0a0",
    "ytick.color":       "#a0a0a0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#2a2d3d",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4d",
    "font.family":       "DejaVu Sans",
})


def _save_and_close(fig: plt.Figure, path: str | Path) -> None:
    """Save a matplotlib figure to *path* and release memory."""
    fig.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cumulative Returns
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_returns(
    strategies: dict[str, pd.Series],
    title: str = "Cumulative Portfolio Returns",
    output_path: str | Path | None = None,
    cfg=None,
) -> plt.Figure:
    """Growth-of-$1 chart for every strategy in *strategies*."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, rets) in enumerate(strategies.items()):
        cum = (1 + rets).cumprod()
        lw  = 2.5 if i == 0 else 1.5
        ls  = "-"  if i == 0 else "--"
        ax.plot(cum.index, cum.values, label=name,
                color=PALETTE[i % len(PALETTE)], lw=lw, ls=ls)

    ax.axhline(1.0, color="#555", lw=0.8, ls=":")
    full_title = title
    if cfg is not None:
        full_title = f"{title}\n{cfg.START_DATE[:4]}–{cfg.END_DATE[:4]}  |  {len(cfg.TICKERS)} assets  |  Cov: {cfg.COVARIANCE_METHOD}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Portfolio Value (normalised to 1)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Portfolio Allocation Over Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_allocation_over_time(
    weights: pd.DataFrame,
    title: str = "Portfolio Allocation Over Time",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked-area chart of portfolio weight evolution over time."""
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = PALETTE[:len(weights.columns)]
    ax.stackplot(weights.index, weights.T.values, labels=weights.columns,
                 colors=colors, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Weight", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    individual_assets: pd.DataFrame | None = None,
    highlight_portfolios: dict | None = None,
    title: str = "Efficient Frontier",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of the mean-variance efficient frontier coloured by Sharpe."""
    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(
        frontier_df["vol"] * 100,
        frontier_df["ret"] * 100,
        c=frontier_df["sharpe"],
        cmap="plasma",
        s=30, alpha=0.9, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", fontsize=10)

    # Individual assets
    if individual_assets is not None:
        ax.scatter(
            individual_assets["vol"] * 100,
            individual_assets["ret"] * 100,
            c="white", s=80, zorder=5, edgecolors="#555", marker="D",
        )
        for _, row in individual_assets.iterrows():
            ax.annotate(
                row.name,
                (row["vol"] * 100 + 0.1, row["ret"] * 100),
                fontsize=8, color="#cccccc",
            )

    # Highlight specific portfolios
    if highlight_portfolios:
        for label, (vol, ret, color) in highlight_portfolios.items():
            ax.scatter(vol * 100, ret * 100, c=color, s=150, zorder=6,
                       edgecolors="white", marker="*")
            ax.annotate(label, (vol * 100 + 0.1, ret * 100 + 0.2),
                        fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Annual Volatility (%)", fontsize=11)
    ax.set_ylabel("Annual Return (%)", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Drawdown Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_drawdown(
    strategies: dict[str, pd.Series],
    title: str = "Portfolio Drawdown",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Underwater / drawdown chart for multiple strategies."""
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (name, rets) in enumerate(strategies.items()):
        cum = (1 + rets).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        lw  = 2.5 if i == 0 else 1.5
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.25 if i == 0 else 0.1,
                        color=PALETTE[i % len(PALETTE)])
        ax.plot(dd.index, dd.values, label=name,
                color=PALETTE[i % len(PALETTE)], lw=lw)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Return Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_return_distribution(
    returns: pd.Series,
    title: str = "Monthly Return Distribution",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram with Normal overlay and VaR/CVaR vertical markers."""
    from scipy.stats import norm, skew, kurtosis

    fig, ax = plt.subplots(figsize=(10, 6))

    rets = returns.dropna()
    ax.hist(rets * 100, bins=40, density=True, alpha=0.65,
            color=PALETTE[0], edgecolor="#1a1d27", label="Empirical")

    # Normal overlay
    mu_val, sig_val = rets.mean() * 100, rets.std() * 100
    xs = np.linspace(mu_val - 4 * sig_val, mu_val + 4 * sig_val, 300)
    ax.plot(xs, norm.pdf(xs, mu_val, sig_val), color="#f39c12",
            lw=2, label="Normal fit")

    # VaR / CVaR lines
    var95 = np.percentile(rets, 5) * 100
    cvar95 = rets[rets <= rets.quantile(0.05)].mean() * 100
    ax.axvline(var95, color="#e74c3c", lw=1.5, ls="--", label=f"VaR 95%: {var95:.2f}%")
    ax.axvline(cvar95, color="#c0392b", lw=1.5, ls=":",  label=f"CVaR 95%: {cvar95:.2f}%")

    stats_text = (
        f"Mean: {mu_val:.2f}%\nStd: {sig_val:.2f}%\n"
        f"Skew: {skew(rets):.2f}\nKurt: {kurtosis(rets):.2f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3d", alpha=0.8))

    ax.set_xlabel("Monthly Return (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rolling Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_sharpe(
    strategies: dict[str, pd.Series],
    window: int = 12,
    risk_free_monthly: float = 0.04 / 12,
    title: str = "Rolling 12-Month Sharpe Ratio",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Rolling annualised Sharpe ratio for each strategy."""
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (name, rets) in enumerate(strategies.items()):
        excess = rets - risk_free_monthly
        rolling_sharpe = (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(12)
        )
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                label=name, color=PALETTE[i % len(PALETTE)],
                lw=2 if i == 0 else 1.5)

    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.axhline(1, color="#555", lw=0.8, ls="--")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Annotated heatmap of pairwise asset return correlations."""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = LinearSegmentedColormap.from_list(
        "corr", ["#d7191c", "#f7f7f7", "#2c7bb6"], N=256
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ticks = range(len(corr))
    ax.set_xticks(ticks); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "black" if abs(val) > 0.5 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Regime Timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_timeline(
    returns: pd.Series,
    regimes: pd.Series,
    title: str = "Market Regime Classification",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Cumulative return line with regime-coloured background shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Cumulative returns
    cum = (1 + returns).cumprod()
    ax1.plot(cum.index, cum.values, color=PALETTE[0], lw=2)

    # Shade regime periods
    active_regimes = sorted(regimes.unique())
    names = REGIME_NAMES_4 if max(active_regimes) >= 3 else REGIME_NAMES
    for regime in active_regimes:
        color = REGIME_COLORS.get(regime, "#888")
        mask = regimes == regime
        dates = regimes.index[mask]
        for d in dates:
            if d in cum.index:
                ax1.axvspan(d - pd.offsets.MonthBegin(1), d,
                            alpha=0.2, color=color, lw=0)

    patches = [mpatches.Patch(color=REGIME_COLORS.get(k, "#888"),
               label=names.get(k, f"Regime {k}"), alpha=0.6)
               for k in active_regimes]
    ax1.legend(handles=patches, fontsize=9, loc="upper left")
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.3)

    # Regime bars
    regime_colors_series = regimes.map(REGIME_COLORS).fillna("#888")
    ax2.bar(regimes.index, [1] * len(regimes), width=25,
            color=regime_colors_series.values, alpha=0.8)
    ax2.set_ylabel("Regime", fontsize=10)
    ax2.set_yticks([])
    ax2.grid(False)

    fig.tight_layout()
    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Monthly Returns Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Year × Month heatmap of portfolio returns (green = positive, red = negative)."""
    df = returns.to_frame("ret").copy()
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.45)))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)

    cmap = LinearSegmentedColormap.from_list("ret", ["#c0392b", "#1a1d27", "#27ae60"])
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02,
                 format=mticker.PercentFormatter(xmax=1, decimals=1))

    ax.set_xticks(range(12)); ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=7.5, color="white")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()

    if output_path:
        _save_and_close(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. Comprehensive Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_dashboard(
    portfolio_returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    weights_history: pd.DataFrame,
    regimes: pd.Series | None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Five-panel summary dashboard: returns, drawdown, allocation, distribution, rolling Sharpe."""
    all_strats = {"Optimizer": portfolio_returns, **benchmark_returns}

    fig = plt.figure(figsize=(18, 16))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # Cumulative returns
    for i, (name, rets) in enumerate(all_strats.items()):
        cum = (1 + rets).cumprod()
        ax1.plot(cum.index, cum.values, label=name,
                 color=PALETTE[i], lw=2.5 if i == 0 else 1.5,
                 ls="-" if i == 0 else "--")
    ax1.axhline(1.0, color="#555", lw=0.8)
    ax1.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Drawdown
    for i, (name, rets) in enumerate(all_strats.items()):
        cum = (1 + rets).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.plot(dd.index, dd.values * 100, label=name,
                 color=PALETTE[i], lw=2 if i == 0 else 1.5)
    ax2.set_title("Drawdown (%)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Allocation over time
    colors = PALETTE[:len(weights_history.columns)]
    ax3.stackplot(weights_history.index, weights_history.T.values,
                  labels=weights_history.columns, colors=colors, alpha=0.85)
    ax3.set_title("Portfolio Allocation", fontsize=12, fontweight="bold")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax3.legend(loc="upper left", fontsize=7, ncol=2); ax3.grid(True, alpha=0.3)

    # Return distribution
    rets = portfolio_returns.dropna()
    ax4.hist(rets * 100, bins=35, color=PALETTE[0], alpha=0.7, density=True)
    from scipy.stats import norm
    xs = np.linspace(rets.min() * 100, rets.max() * 100, 200)
    ax4.plot(xs, norm.pdf(xs, rets.mean() * 100, rets.std() * 100),
             color="#f39c12", lw=2)
    ax4.axvline(np.percentile(rets, 5) * 100, color="#e74c3c", lw=1.5, ls="--",
                label="VaR 95%")
    ax4.set_title("Return Distribution", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # Rolling Sharpe
    excess = portfolio_returns - 0.04 / 12
    rs = excess.rolling(12).mean() / excess.rolling(12).std() * np.sqrt(12)
    ax5.plot(rs.index, rs.values, color=PALETTE[0], lw=2)
    ax5.axhline(0, color="#888", lw=0.8)
    ax5.axhline(1, color="#555", lw=0.8, ls="--")
    ax5.fill_between(rs.index, rs.values, 0,
                     where=(rs.values > 0), alpha=0.2, color=PALETTE[1])
    ax5.fill_between(rs.index, rs.values, 0,
                     where=(rs.values < 0), alpha=0.2, color="#e74c3c")
    ax5.set_title("Rolling 12m Sharpe Ratio", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        "ML-Enhanced Quantitative Portfolio Optimizer — Performance Dashboard",
        fontsize=16, fontweight="bold", y=1.01,
    )
    if output_path:
        _save_and_close(fig, output_path)
    return fig
