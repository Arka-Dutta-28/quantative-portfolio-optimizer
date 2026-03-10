# ML-Enhanced Quantitative Portfolio Optimizer
### Institutional Asset Allocation System

---

## Overview

A production-style **Strategic Asset Allocation (SAA) engine** that integrates machine learning, autoregressive/GARCH time-series models, robust risk modelling, and constrained portfolio optimisation — mirroring the quantitative systems used by institutional asset managers.

Includes an **interactive Streamlit web dashboard** and an auto-generated **PDF research report**.

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

- Long-only (w ≥ 0)
- Max single-asset weight: 35%
- Portfolio volatility cap: 10% annual
- Max turnover per rebalance: 60%
- Transaction costs: 10 bps per unit traded

---

## Regime Detection

The GMM identifies three regimes from financial time series:

| Regime | Characteristics | Strategy Tilt |
|--------|----------------|---------------|
| **Bull** | Positive momentum, low vol | +Equity, -Bonds |
| **Neutral** | Transition | Balanced |
| **Bear/Crisis** | Negative momentum, high vol | -Equity, +Bonds/Gold |

---

## Performance Metrics

- **Return**: CAGR, cumulative return, monthly return heatmap
- **Risk**: Volatility, VaR 95%, CVaR 95%, Max Drawdown
- **Ratios**: Sharpe, Sortino, Calmar, Information Ratio
- **Relative**: Beta, Alpha, Tracking Error vs. S&P 500

---

## Visualisations Generated

1. `01_cumulative_returns.png` — Strategy comparison
2. `02_allocation_over_time.png` — Stacked weight chart
3. `03_efficient_frontier.png` — Sharpe-coloured frontier
4. `04_drawdown.png` — Drawdown curves
5. `05_return_distribution.png` — Histogram + VaR/CVaR
6. `06_rolling_sharpe.png` — Rolling 12m Sharpe
7. `07_correlation_heatmap.png` — Asset correlations
8. `08_regime_timeline.png` — GMM regime classification
9. `09_monthly_heatmap.png` — Monthly return calendar
10. `10_full_dashboard.png` — Comprehensive summary

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

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (CLI — uses synthetic data if no internet)
python main.py

# Launch the interactive web dashboard
streamlit run webapp.py
```

---

## Dependencies

See `requirements.txt`. Core libraries:

```
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
scipy >= 1.11
matplotlib >= 3.7
cvxpy >= 1.3          # optional — falls back to scipy
xgboost >= 2.0        # optional — falls back to RF + Ridge
yfinance >= 0.2       # optional — falls back to synthetic data
streamlit >= 1.30     # for the web dashboard
plotly >= 5.18        # interactive charts in the dashboard
reportlab >= 4.0      # PDF report generation
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
│   └── report_generator.py             # Multi-section PDF report
│
└── outputs/                            # Generated charts (git-ignored)
```

---

## ML vs AR Model Usage

The pipeline trains **both** the ML ensemble and the AR/GARCH family and compares them head-to-head on directional accuracy and information coefficient. Currently, **only the ML ensemble forecasts drive portfolio decisions** (expected returns fed to the optimiser and backtester). The AR/GARCH models are fitted, evaluated, and reported for transparency but do not influence weights.

**Rationale:** Across all asset universes and date ranges tested during development, the ML ensemble consistently matched or outperformed the AR/GARCH models on both directional accuracy and rank-correlation (IC). The AR models therefore serve as a benchmark and diagnostic tool rather than a competing signal source.

---

## Future Work

- **Adaptive model blending** — weight ML and AR forecasts per asset based on rolling out-of-sample IC, so the portfolio automatically shifts toward whichever model is performing better in the current regime.
- **Winner-takes-all routing** — for each asset, route the expected-return input to the optimiser from whichever model family (ML or AR) had higher directional accuracy over the most recent validation window.
- **Volatility-aware sizing** — use GARCH/EGARCH conditional volatility estimates from the AR module to dynamically scale position sizes or tighten the vol-target constraint.
- **Additional model families** — incorporate LSTM / Transformer sequence models as a third forecasting pillar alongside ML and AR.
- **Live trading integration** — connect to a broker API for paper-trading with real-time rebalancing signals.

---

## Project Context

This system demonstrates the core components of a **quant investment research platform**:

- **Strategic Asset Allocation** framework with multiple optimisation strategies
- **Machine Learning** ensemble models applied to financial time series
- **Autoregressive / GARCH** time-series models with data-driven model selection
- **Robust optimisation** with real-world constraints (vol cap, turnover, long-only)
- **Walk-forward backtesting** with transaction costs
- **Risk analytics** used by institutional investment teams
- **Interactive dashboard** for live exploration of results
- **Professional PDF report** generation for stakeholder delivery
