"""
data/data_loader.py
───────────────────
Loads historical price data for the asset universe.

Real usage  → downloads from Yahoo Finance via yfinance.
Offline use → generates realistic synthetic price series via GBM + regime
              switching so the full pipeline runs without a network connection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data parameters (GBM parameters per asset, calibrated roughly
# to real long-run statistics, 2013-2023)
# ─────────────────────────────────────────────────────────────────────────────
_ASSET_PARAMS: dict[str, dict] = {
    # ticker : (annual_mu, annual_sigma, starting_price)
    "SPY": {"mu": 0.13, "sigma": 0.16, "S0": 140.0},
    "QQQ": {"mu": 0.17, "sigma": 0.20, "S0": 70.0},
    "TLT": {"mu": 0.02, "sigma": 0.12, "S0": 115.0},
    "IEF": {"mu": 0.02, "sigma": 0.06, "S0": 100.0},
    "GLD": {"mu": 0.04, "sigma": 0.14, "S0": 155.0},
    "VNQ": {"mu": 0.10, "sigma": 0.18, "S0": 68.0},
    "EEM": {"mu": 0.07, "sigma": 0.22, "S0": 43.0},
    "HYG": {"mu": 0.05, "sigma": 0.08, "S0": 90.0},
}

# Correlation matrix (rough long-run estimates)
_CORR = np.array([
    # SPY   QQQ   TLT   IEF   GLD   VNQ   EEM   HYG
    [ 1.00, 0.90,-0.30,-0.25, 0.05, 0.65, 0.75, 0.55],  # SPY
    [ 0.90, 1.00,-0.25,-0.20, 0.02, 0.55, 0.68, 0.48],  # QQQ
    [-0.30,-0.25, 1.00, 0.90, 0.15,-0.15,-0.25,-0.05],  # TLT
    [-0.25,-0.20, 0.90, 1.00, 0.10,-0.10,-0.20,-0.02],  # IEF
    [ 0.05, 0.02, 0.15, 0.10, 1.00, 0.10, 0.10, 0.05],  # GLD
    [ 0.65, 0.55,-0.15,-0.10, 0.10, 1.00, 0.55, 0.50],  # VNQ
    [ 0.75, 0.68,-0.25,-0.20, 0.10, 0.55, 1.00, 0.60],  # EEM
    [ 0.55, 0.48,-0.05,-0.02, 0.05, 0.50, 0.60, 1.00],  # HYG
])


def _generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate daily synthetic prices using correlated GBM with three
    volatility regimes (low / medium / high) to create realistic drawdown
    and recovery patterns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)
    n_assets = len(tickers)

    params = [_ASSET_PARAMS[t] for t in tickers]
    mus    = np.array([p["mu"]    for p in params])
    sigs   = np.array([p["sigma"] for p in params])
    S0     = np.array([p["S0"]    for p in params])

    # Build covariance matrix from correlation + vols
    idx = [list(_ASSET_PARAMS.keys()).index(t) for t in tickers]
    corr_sub = _CORR[np.ix_(idx, idx)]
    vol_diag  = np.diag(sigs)
    cov_annual = vol_diag @ corr_sub @ vol_diag
    cov_daily  = cov_annual / 252

    # Cholesky decomposition for correlated draws
    L = np.linalg.cholesky(cov_daily + 1e-8 * np.eye(n_assets))

    # Regime switching (3 regimes: calm, volatile, crisis)
    regime_probs = np.array([0.60, 0.25, 0.15])
    regime_vol_multipliers = np.array([0.7, 1.2, 2.5])
    regime_mu_adjustments  = np.array([0.0, -0.02/252, -0.15/252])

    # Draw regime sequence (Markov-like)
    transitions = np.array([
        [0.97, 0.02, 0.01],
        [0.10, 0.85, 0.05],
        [0.15, 0.20, 0.65],
    ])
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = rng.choice(3, p=regime_probs)
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=transitions[regimes[t - 1]])

    # Simulate log-returns
    z = rng.standard_normal((n_days, n_assets))
    eps = z @ L.T  # correlated innovations

    mu_daily = mus / 252
    log_returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        vm = regime_vol_multipliers[regimes[t]]
        ma = regime_mu_adjustments[regimes[t]]
        drift = mu_daily + ma - 0.5 * (sigs / np.sqrt(252)) ** 2
        log_returns[t] = drift + vm * eps[t]

    # Cumulative prices
    log_prices = np.cumsum(log_returns, axis=0)
    prices = S0 * np.exp(log_prices)

    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads or generates price data, computes returns and volumes.

    Parameters
    ----------
    tickers   : list of asset tickers
    start     : start date string (YYYY-MM-DD)
    end       : end date string   (YYYY-MM-DD)
    use_cache : if True, saves / loads CSV from data/cache/
    """

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
        strict_yfinance: bool = False,
    ):
        self.tickers   = tickers
        self.start     = start
        self.end       = end
        self.use_cache = use_cache
        self.strict_yfinance = strict_yfinance
        self._cache_dir = Path(__file__).parent / "cache"
        self._cache_dir.mkdir(exist_ok=True)

        self.prices:  pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None

    # ── Core Methods ──────────────────────────────────────────────────────────

    def load(self) -> "DataLoader":
        """Load data (from cache, Yahoo Finance, or synthetic generator)."""
        cache_file = self._cache_dir / f"prices_{'_'.join(self.tickers)}_{self.start}_{self.end}.csv"

        if self.use_cache and cache_file.exists():
            logger.info("Loading prices from cache: %s", cache_file)
            self.prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            self.prices = self._download_or_generate()
            if self.use_cache:
                self.prices.to_csv(cache_file)
                logger.info("Prices cached to %s", cache_file)

        self._validate()
        self._compute_returns()
        logger.info(
            "Data loaded: %d assets × %d days (%s → %s)",
            len(self.tickers),
            len(self.prices),
            self.prices.index[0].date(),
            self.prices.index[-1].date(),
        )
        return self

    def get_monthly_returns(self) -> pd.DataFrame:
        """Resample daily prices to month-end and compute monthly returns."""
        monthly_prices = self.prices.resample("ME").last()
        return monthly_prices.pct_change().dropna()

    def get_monthly_prices(self) -> pd.DataFrame:
        """Resample daily prices to month-end (no return computation)."""
        return self.prices.resample("ME").last()

    def get_macro_features(self) -> pd.DataFrame:
        """
        Returns synthetic macro features (VIX proxy, yield spread, CPI trend).
        In production these would be fetched from FRED.
        """
        monthly = self.get_monthly_returns()
        idx = monthly.index

        rng = np.random.default_rng(999)
        n = len(idx)

        # VIX proxy: rolling realised vol of SPY (annualised %)
        if "SPY" in monthly.columns:
            vix = monthly["SPY"].rolling(3).std() * np.sqrt(12) * 100
        else:
            vix = pd.Series(15 + rng.standard_normal(n) * 5, index=idx)

        # 10Y-2Y yield spread (synthetic mean-reverting)
        spread = pd.Series(
            1.5 + np.cumsum(rng.standard_normal(n) * 0.05), index=idx
        ).clip(-1.5, 3.5)

        # CPI trend (random walk, slowly drifting)
        cpi = pd.Series(
            2.0 + np.cumsum(rng.standard_normal(n) * 0.08), index=idx
        ).clip(0, 9)

        macro = pd.DataFrame(
            {"vix": vix, "yield_spread": spread, "cpi_trend": cpi},
            index=idx,
        )
        return macro.bfill().ffill()

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _download_or_generate(self) -> pd.DataFrame:
        """Try yfinance download; fall back to synthetic GBM if unavailable."""
        try:
            import yfinance as yf  # optional dependency
            logger.info("Downloading data from Yahoo Finance …")
            raw = yf.download(
                self.tickers,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                if self.strict_yfinance:
                    raise ValueError(
                        f"yfinance returned no data for {self.tickers}. "
                        f"Check tickers and date range."
                    )
                return _generate_synthetic_prices(self.tickers, self.start, self.end)
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(self.tickers[0])
            missing = set(self.tickers) - set(close.columns)
            if missing:
                logger.warning("Missing tickers from yfinance: %s", missing)
                if self.strict_yfinance:
                    raise ValueError(
                        f"yfinance returned no data for: {missing}. "
                        f"Check that these tickers exist on Yahoo Finance."
                    )
            return close[self.tickers].dropna(how="all")
        except ValueError:
            raise
        except Exception as exc:
            if self.strict_yfinance:
                raise RuntimeError(
                    f"yfinance download failed: {exc}. "
                    f"Ensure yfinance is installed and you have internet access."
                ) from exc
            logger.warning(
                "yfinance download failed (%s). Falling back to synthetic data.", exc
            )
            return _generate_synthetic_prices(self.tickers, self.start, self.end)

    def _validate(self) -> None:
        """Check for missing tickers and forward-fill small NaN gaps."""
        missing = set(self.tickers) - set(self.prices.columns)
        if missing:
            raise ValueError(f"Missing tickers in price data: {missing}")
        pct_missing = self.prices.isna().mean()
        high_missing = pct_missing[pct_missing > 0.05]
        if not high_missing.empty:
            logger.warning("High NaN rate:\n%s", high_missing)
        # Forward-fill small gaps
        self.prices = self.prices.ffill().bfill()

    def _compute_returns(self) -> None:
        self.returns = self.prices.pct_change().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config import TICKERS, START_DATE, END_DATE

    loader = DataLoader(TICKERS, START_DATE, END_DATE)
    loader.load()

    print("\n── Monthly Returns (first 5 rows) ──")
    print(loader.get_monthly_returns().head())
    print("\n── Macro Features (first 5 rows) ──")
    print(loader.get_macro_features().head())
