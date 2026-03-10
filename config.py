"""
config.py
─────────
Global configuration for the ML-Enhanced Quantitative Portfolio Optimizer.

All pipeline parameters are centralised here.  Edit this file and re-run
``main.py`` — every chart, table, and the PDF report will reflect your changes.

Sections
--------
  Asset Universe         Tickers and asset-class mappings
  Data Settings          Date range, frequency
  Feature Engineering    Lookback / momentum / volatility windows
  ML Model Settings      Forecast horizon, train/val split, random seed
  Risk Model             Covariance estimation method and parameters
  Optimization           Constraints (weights, vol cap, turnover, tx cost)
  Backtesting            Rebalance frequency, transaction cost
  Regime Detection       Number of GMM regimes
  Benchmarks             Named benchmark strategies
  Output Paths           Directories for charts and reports
"""

# ── Asset Universe ────────────────────────────────────────────────────────────
ASSETS = {
    "SPY":  "S&P 500 ETF (US Equities)",
    "QQQ":  "NASDAQ ETF (Tech Equities)",
    "TLT":  "20+ Year Treasury ETF (Long Bonds)",
    "IEF":  "7-10 Year Treasury ETF (Mid Bonds)",
    "GLD":  "Gold ETF (Commodity)",
    "VNQ":  "Real Estate ETF (REIT)",
    "EEM":  "Emerging Markets ETF",
    "HYG":  "High-Yield Corporate Bond ETF",
}

ASSET_CLASSES = {
    "SPY": "Equity",
    "QQQ": "Equity",
    "TLT": "Bond",
    "IEF": "Bond",
    "GLD": "Commodity",
    "VNQ": "Real Estate",
    "EEM": "Equity",
    "HYG": "Bond",
}

TICKERS = list(ASSETS.keys())

# ── Data Settings ─────────────────────────────────────────────────────────────
START_DATE = "2013-01-01"
END_DATE   = "2023-12-31"
FREQUENCY  = "monthly"        # 'daily' | 'monthly'

# ── Feature Engineering ───────────────────────────────────────────────────────
LOOKBACK_WINDOWS = [1, 3, 6, 12]   # months
VOLATILITY_WINDOW = 12             # months for rolling vol
MOMENTUM_WINDOWS  = [3, 6, 12]     # months

# ── ML Model Settings ─────────────────────────────────────────────────────────
FORECAST_HORIZON   = 1      # months ahead
TRAIN_YEARS        = 5      # initial training window
VALIDATION_SPLIT   = 0.2
RANDOM_STATE       = 42

# ── Risk Model ────────────────────────────────────────────────────────────────
COVARIANCE_METHOD  = "ledoit_wolf"   # 'sample' | 'ewm' | 'ledoit_wolf'
EWM_HALFLIFE       = 12             # months

# ── Optimization Settings ─────────────────────────────────────────────────────
RISK_FREE_RATE     = 0.04           # annual
TARGET_VOLATILITY  = 0.10           # annual portfolio vol cap
MAX_WEIGHT         = 0.35           # max single asset weight
MIN_WEIGHT         = 0.00           # long-only
MAX_TURNOVER       = 0.50           # max % portfolio turned over per rebalance
CVAR_ALPHA         = 0.05           # CVaR confidence level (5% tail)
N_SCENARIOS        = 1000           # Monte Carlo scenarios for CVaR

# ── Backtesting ───────────────────────────────────────────────────────────────
REBALANCE_FREQ     = "M"            # pandas offset alias
TRANSACTION_COST   = 0.001          # 10 bps per trade

# ── Regime Detection ─────────────────────────────────────────────────────────
N_REGIMES          = 3              # bull / neutral / bear
REGIME_FEATURES    = ["market_return", "volatility", "momentum_6m", "yield_spread"]

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS = {
    "Equal Weight":  "equal_weight",
    "Risk Parity":   "risk_parity",
    "S&P 500 (SPY)": "spy_only",
}

# ── Output Paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = "outputs"
REPORT_DIR  = "reports"
