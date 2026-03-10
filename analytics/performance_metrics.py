"""
analytics/performance_metrics.py
──────────────────────────────────
Comprehensive risk-adjusted performance analytics.

Metrics computed:
  Returns   : cumulative, annualised, CAGR, monthly
  Risk      : volatility, VaR (historical / parametric), CVaR / ES
  Drawdown  : max drawdown, drawdown duration, recovery time
  Ratios    : Sharpe, Sortino, Calmar, Information Ratio vs benchmark
  Other     : beta, alpha, tracking error, hit rate
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Computes a full suite of performance metrics for a return series.

    Parameters
    ----------
    returns      : pd.Series of monthly portfolio returns
    benchmark    : pd.Series of monthly benchmark returns (optional)
    risk_free    : annual risk-free rate (default 4%)
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        risk_free: float = 0.04,
    ):
        self.returns   = returns.dropna()
        self.benchmark = benchmark.dropna() if benchmark is not None else None
        self.rf_monthly = risk_free / 12
        self.rf_annual  = risk_free

    # ── Full Summary ──────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a comprehensive performance summary DataFrame."""
        m = {}

        # Return metrics
        m["Total Return"]         = self.total_return()
        m["CAGR"]                 = self.cagr()
        m["Best Month"]           = float(self.returns.max())
        m["Worst Month"]          = float(self.returns.min())
        m["% Positive Months"]    = float((self.returns > 0).mean())

        # Risk metrics
        m["Annual Volatility"]    = self.annual_volatility()
        m["VaR 95% (monthly)"]    = self.var(alpha=0.05)
        m["CVaR 95% (monthly)"]   = self.cvar(alpha=0.05)
        m["Max Drawdown"]         = self.max_drawdown()
        m["Avg Drawdown Duration"]= self.avg_drawdown_duration()

        # Risk-adjusted ratios
        m["Sharpe Ratio"]         = self.sharpe()
        m["Sortino Ratio"]        = self.sortino()
        m["Calmar Ratio"]         = self.calmar()

        # vs. benchmark
        if self.benchmark is not None:
            m["Information Ratio"]    = self.information_ratio()
            m["Tracking Error (ann)"] = self.tracking_error()
            m["Beta"]                 = self.beta()
            m["Alpha (ann)"]          = self.alpha()

        df = pd.DataFrame.from_dict(m, orient="index", columns=["Value"])
        df["Value"] = df["Value"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
        return df

    # ── Return Metrics ────────────────────────────────────────────────────────

    def total_return(self) -> float:
        return float((1 + self.returns).prod() - 1)

    def cagr(self) -> float:
        n_years = len(self.returns) / 12
        if n_years <= 0:
            return 0.0
        return float((1 + self.total_return()) ** (1 / n_years) - 1)

    def annual_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(12))

    def cumulative_returns(self) -> pd.Series:
        return (1 + self.returns).cumprod() - 1

    def drawdown_series(self) -> pd.Series:
        cum = (1 + self.returns).cumprod()
        rolling_max = cum.cummax()
        return (cum - rolling_max) / rolling_max

    # ── Risk Metrics ──────────────────────────────────────────────────────────

    def var(self, alpha: float = 0.05, method: str = "historical") -> float:
        """Value at Risk (monthly, positive = loss)."""
        if method == "historical":
            return float(-np.percentile(self.returns, alpha * 100))
        else:  # parametric normal
            mu  = self.returns.mean()
            sig = self.returns.std()
            return float(-(mu + sig * stats.norm.ppf(alpha)))

    def cvar(self, alpha: float = 0.05) -> float:
        """Conditional VaR / Expected Shortfall (monthly, positive = loss)."""
        threshold = np.percentile(self.returns, alpha * 100)
        tail = self.returns[self.returns <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def max_drawdown(self) -> float:
        return float(self.drawdown_series().min())

    def avg_drawdown_duration(self) -> float:
        """Average number of months spent in a drawdown."""
        dd = self.drawdown_series()
        in_dd = (dd < 0).astype(int)
        # Split into drawdown episodes
        episodes = []
        count = 0
        for v in in_dd:
            if v:
                count += 1
            elif count > 0:
                episodes.append(count)
                count = 0
        if count > 0:
            episodes.append(count)
        return float(np.mean(episodes)) if episodes else 0.0

    def drawdown_table(self, top_n: int = 5) -> pd.DataFrame:
        """Return top-N drawdown periods with depth, start, end, recovery."""
        dd = self.drawdown_series()
        cum = (1 + self.returns).cumprod()

        rows = []
        in_dd = False
        start = None
        trough_date = None
        trough_val  = 0.0

        for date, val in dd.items():
            if not in_dd and val < 0:
                in_dd = True
                start = date
                trough_date = date
                trough_val  = val
            elif in_dd and val < trough_val:
                trough_date = date
                trough_val  = val
            elif in_dd and val >= 0:
                rows.append({
                    "Start":    start,
                    "Trough":   trough_date,
                    "Recovery": date,
                    "Depth":    trough_val,
                    "Duration": (trough_date - start).days,
                })
                in_dd = False
                start = trough_date = None
                trough_val = 0.0

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.nsmallest(top_n, "Depth").reset_index(drop=True)

    # ── Risk-Adjusted Ratios ──────────────────────────────────────────────────

    def sharpe(self) -> float:
        excess = self.returns - self.rf_monthly
        if excess.std() < 1e-9:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(12))

    def sortino(self) -> float:
        excess = self.returns - self.rf_monthly
        downside = excess[excess < 0]
        downside_std = downside.std() * np.sqrt(12)
        if downside_std < 1e-9:
            return 0.0
        return float(excess.mean() * 12 / downside_std)

    def calmar(self) -> float:
        mdd = abs(self.max_drawdown())
        if mdd < 1e-9:
            return 0.0
        return float(self.cagr() / mdd)

    # ── Benchmark-Relative Metrics ────────────────────────────────────────────

    def information_ratio(self) -> float:
        if self.benchmark is None:
            return np.nan
        aligned = self._align_benchmark()
        active = aligned["portfolio"] - aligned["benchmark"]
        if active.std() < 1e-9:
            return 0.0
        return float(active.mean() / active.std() * np.sqrt(12))

    def tracking_error(self) -> float:
        if self.benchmark is None:
            return np.nan
        aligned = self._align_benchmark()
        return float((aligned["portfolio"] - aligned["benchmark"]).std() * np.sqrt(12))

    def beta(self) -> float:
        if self.benchmark is None:
            return np.nan
        aligned = self._align_benchmark()
        cov = np.cov(aligned["portfolio"], aligned["benchmark"])
        bm_var = cov[1, 1]
        return float(cov[0, 1] / bm_var) if bm_var > 1e-9 else 1.0

    def alpha(self) -> float:
        if self.benchmark is None:
            return np.nan
        beta = self.beta()
        return float(self.cagr() - self.rf_annual - beta * (
            PerformanceAnalytics(self.benchmark, risk_free=self.rf_annual).cagr()
            - self.rf_annual
        ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _align_benchmark(self) -> pd.DataFrame:
        bm = self.benchmark.reindex(self.returns.index).fillna(0)
        return pd.DataFrame({"portfolio": self.returns, "benchmark": bm})

    @staticmethod
    def compare(
        strategies: dict[str, pd.Series],
        benchmark: pd.Series | None = None,
        risk_free: float = 0.04,
    ) -> pd.DataFrame:
        """Compare multiple strategies side-by-side."""
        rows = {}
        for name, rets in strategies.items():
            pa = PerformanceAnalytics(rets, benchmark, risk_free)
            rows[name] = {
                "CAGR":          pa.cagr(),
                "Vol":           pa.annual_volatility(),
                "Sharpe":        pa.sharpe(),
                "Sortino":       pa.sortino(),
                "Calmar":        pa.calmar(),
                "Max Drawdown":  pa.max_drawdown(),
                "CVaR 95%":      pa.cvar(),
                "Total Return":  pa.total_return(),
            }
        return pd.DataFrame(rows).T.round(4)


if __name__ == "__main__":
    np.random.seed(42)
    monthly_rets = pd.Series(np.random.normal(0.008, 0.04, 120))
    bench = pd.Series(np.random.normal(0.006, 0.035, 120))

    pa = PerformanceAnalytics(monthly_rets, bench)
    print(pa.summary())
    print("\n── Top Drawdowns ──")
    print(pa.drawdown_table())
