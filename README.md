# ML-Enhanced Quantitative Portfolio Optimizer

**Institutional-grade Strategic Asset Allocation engine** combining machine learning, AR/GARCH time-series models, regime detection, and constrained portfolio optimisation — with an interactive Streamlit dashboard and auto-generated PDF research reports.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quantative-portfolio-optimizer.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

> **Live Demo** — [quantative-portfolio-optimizer.streamlit.app](https://quantative-portfolio-optimizer.streamlit.app/)

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Module Summary](#module-summary)
- [Asset Universe](#asset-universe)
- [Optimization Strategies](#optimization-strategies)
- [Constraints](#constraints)
- [Regime Detection](#regime-detection)
- [AR / GARCH Time-Series Models](#ar--garch-time-series-models)
- [ML vs AR Model Usage](#ml-vs-ar-model-usage)
- [Performance Metrics](#performance-metrics)
- [Visualisations](#visualisations)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

---

## Architecture

```
Market Data  →  Feature Engineering  →  ML Return Forecast
     |                                          ↓
     └──→  AR / GARCH Forecasting  ──→  Ensemble Expected Returns
                                                ↓
Regime Detection (GMM)  ──────────→  Covariance Estimation
     ↓                                          ↓
Portfolio Optimization Engine  (MVO / Min-Vol / Max-Sharpe / CVaR / Risk Parity)
     ↓
Walk-Forward Backtester  →  Performance Analytics  →  Visualisations + PDF Report
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/quant_portfolio_optimizer.git
cd quant_portfolio_optimizer

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (CLI — uses synthetic data if no internet)
python main.py

# Launch the interactive web dashboard
streamlit run webapp.py
```

---

## Project Structure

```
quant_portfolio_optimizer/
├── config.py                           # Global parameters
├── main.py                             # CLI pipeline orchestrator
├── webapp.py                           # Streamlit web dashboard
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── data_loader.py                  # Yahoo Finance / synthetic loader
│   └── cache/                          # Auto-generated price CSV cache
│
├── features/
│   └── feature_engineering.py          # 90+ predictive features
│
├── models/
│   ├── return_forecaster.py            # ML ensemble (Ridge, RF, XGBoost)
│   ├── ar_forecaster.py                # AR/GARCH with BIC auto-selection
│   └── regime_detector.py              # GMM regime detection
│
├── risk/
│   └── covariance_estimator.py         # 5 covariance methods
│
├── optimization/
│   └── portfolio_optimizer.py          # 6 strategy implementations
│
├── backtesting/
│   └── backtester.py                   # Walk-forward engine
│
├── analytics/
│   └── performance_metrics.py          # Full risk-adjusted analytics suite
│
├── visualization/
│   └── plots.py                        # 10 matplotlib charts
│
├── reports/
│   └── report_generator.py            # Multi-section PDF report
│
└── outputs/                            # Generated charts (git-ignored)
```

---

## Module Summary

| Module | File | Description |
|--------|------|-------------|
| **Config** | `config.py` | Centralised pipeline parameters |
| **Data** | `data/data_loader.py` | Yahoo Finance loader with synthetic GBM fallback |
| **Features** | `features/feature_engineering.py` | Momentum, volatility, mean-reversion, macro features |
| **ML Model** | `models/return_forecaster.py` | Ridge + Random Forest + XGBoost ensemble |
| **AR/GARCH** | `models/ar_forecaster.py` | AR, ARMA, ARCH, GARCH, EGARCH with BIC auto-selection |
| **Regime** | `models/regime_detector.py` | GMM-based market regime detector (2–4 regimes) |
| **Risk** | `risk/covariance_estimator.py` | Sample / EWM / Ledoit-Wolf / OAS covariance |
| **Optimizer** | `optimization/portfolio_optimizer.py` | MVO / Min-Vol / Max-Sharpe / CVaR / Risk Parity |
| **Backtest** | `backtesting/backtester.py` | Walk-forward engine with transaction costs |
| **Analytics** | `analytics/performance_metrics.py` | Sharpe, Sortino, Calmar, VaR, CVaR, drawdown |
| **Viz** | `visualization/plots.py` | 10 publication-quality matplotlib plots |
| **Reports** | `reports/report_generator.py` | Multi-section PDF research report (ReportLab) |
| **Web App** | `webapp.py` | Interactive Streamlit dashboard with Plotly charts |

---

## Asset Universe

| Ticker | Asset | Class |
|--------|-------|-------|
| SPY | S&P 500 ETF | Equity |
| QQQ | NASDAQ ETF | Equity |
| TLT | 20yr Treasury ETF | Bond |
| IEF | 7-10yr Treasury ETF | Bond |
| GLD | Gold ETF | Commodity |
| VNQ | Real Estate ETF (REIT) | Real Estate |
| EEM | Emerging Markets ETF | Equity |
| HYG | High-Yield Bond ETF | Bond |

---

## Optimization Strategies

| Strategy | Description |
|----------|-------------|
| **MVO + ML + Regime** | Full pipeline: ML forecasts + regime tilt + Markowitz |
| **Min-Vol** | Minimize portfolio volatility |
| **Max-Sharpe** | Maximize risk-adjusted return |
| **CVaR** | Minimize tail risk (Expected Shortfall) |
| **Risk Parity** | Equal risk contribution per asset |
| **Equal Weight** | 1/N benchmark |

---

## Constraints

| Constraint | Default |
|------------|---------|
| Long-only | w ≥ 0 |
| Max single-asset weight | 35 % |
| Portfolio volatility cap | 10 % annualised |
| Max turnover per rebalance | 60 % |
| Transaction costs | 10 bps per unit traded |

---

## Regime Detection

The GMM identifies three regimes from financial time series:

| Regime | Characteristics | Strategy Tilt |
|--------|----------------|---------------|
| **Bull** | Positive momentum, low vol | +Equity, −Bonds |
| **Neutral** | Transition | Balanced |
| **Bear / Crisis** | Negative momentum, high vol | −Equity, +Bonds/Gold |

---

## AR / GARCH Time-Series Models

Five model families are fitted per asset with **BIC-based automatic order selection**:

| Family | Candidate Orders | Purpose |
|--------|-----------------|---------|
| **AR** | p ∈ {1, 2, 3, 4, 5} | Mean return forecast |
| **ARMA** | (p,q) ∈ {(1,1), (2,1), (1,2), (2,2)} | Mean return with MA correction |
| **ARCH** | q ∈ {1, 2} | Volatility clustering |
| **GARCH** | (p,q) ∈ {(1,1), (2,1)} | Persistent volatility dynamics |
| **EGARCH** | (p,q) ∈ {(1,1), (2,1)} | Asymmetric leverage effects |

For each asset × family, all candidate orders are fitted and the **lowest BIC** model is selected. The five selected models are combined into an inverse-RMSE-weighted ensemble.

---

## ML vs AR Model Usage

The pipeline trains **both** the ML ensemble and the AR/GARCH family and evaluates them head-to-head on directional accuracy and information coefficient. Currently, **only the ML ensemble forecasts drive portfolio decisions** — the AR/GARCH models are fitted, evaluated, and reported for transparency but do not influence weights.

**Rationale:** Across all asset universes and date ranges tested during development, the ML ensemble consistently matched or outperformed the AR/GARCH models on both directional accuracy and rank-correlation (IC). The AR models therefore serve as a benchmark and diagnostic tool rather than a competing signal source.

---

## Performance Metrics

| Category | Metrics |
|----------|---------|
| **Return** | CAGR, cumulative return, monthly return heatmap |
| **Risk** | Volatility, VaR 95 %, CVaR 95 %, Max Drawdown |
| **Ratios** | Sharpe, Sortino, Calmar, Information Ratio |
| **Relative** | Beta, Alpha, Tracking Error vs. S&P 500 |

---

## Visualisations

| # | Chart | Description |
|---|-------|-------------|
| 1 | `01_cumulative_returns.png` | Strategy comparison |
| 2 | `02_allocation_over_time.png` | Stacked weight chart |
| 3 | `03_efficient_frontier.png` | Sharpe-coloured frontier |
| 4 | `04_drawdown.png` | Drawdown curves |
| 5 | `05_return_distribution.png` | Histogram + VaR / CVaR |
| 6 | `06_rolling_sharpe.png` | Rolling 12 m Sharpe |
| 7 | `07_correlation_heatmap.png` | Asset correlations |
| 8 | `08_regime_timeline.png` | GMM regime classification |
| 9 | `09_monthly_heatmap.png` | Monthly return calendar |
| 10 | `10_full_dashboard.png` | Comprehensive summary |

---

## Dependencies

Core libraries (see `requirements.txt` for pinned versions):

| Package | Role |
|---------|------|
| `numpy`, `pandas`, `scipy` | Numerical computing and data handling |
| `scikit-learn` | Ridge, Random Forest, preprocessing |
| `xgboost` | Gradient-boosted tree ensemble |
| `cvxpy` | Convex portfolio optimisation (falls back to scipy) |
| `yfinance` | Market data from Yahoo Finance |
| `streamlit`, `plotly` | Interactive web dashboard |
| `matplotlib` | Static publication-quality charts |
| `reportlab` | PDF report generation |

---

## Future Work

- **Adaptive model blending** — weight ML and AR forecasts per asset based on rolling out-of-sample IC, so the portfolio automatically shifts toward whichever model is performing better in the current regime.
- **Winner-takes-all routing** — for each asset, route the expected-return input to the optimiser from whichever model family had higher directional accuracy over the most recent validation window.
- **Volatility-aware sizing** — use GARCH/EGARCH conditional volatility estimates to dynamically scale position sizes or tighten the vol-target constraint.
- **Additional model families** — incorporate LSTM / Transformer sequence models as a third forecasting pillar alongside ML and AR.
- **Live trading integration** — connect to a broker API for paper-trading with real-time rebalancing signals.
