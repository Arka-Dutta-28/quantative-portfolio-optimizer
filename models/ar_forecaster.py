"""
models/ar_forecaster.py
───────────────────────
Pure NumPy / SciPy implementations of classical time-series models for
one-step-ahead return forecasting, with automatic order selection via BIC.

Model families (each searched over candidate orders per asset)
──────────────────────────────────────────────────────────────
  AR(p)           p in {1,2,3,4,5}         — OLS fit
  ARMA(p,q)       (p,q) in {(1,1),(2,1),(1,2),(2,2)}  — conditional MLE
  ARCH(q)         q in {1,2}               — conditional MLE
  GARCH(qa,qg)    (qa,qg) in {(1,1),(2,1)} — conditional MLE
  EGARCH(qa,qg)   (qa,qg) in {(1,1),(2,1)} — conditional MLE

For each asset, the best order within each family is chosen by BIC
(Bayesian Information Criterion). The 5 family winners are then
ensembled via inverse-validation-MSE weighting.
"""

from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_TINY = 1e-8
_MAX_ITER = 300


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute validation metrics: RMSE, directional accuracy, and Spearman IC."""
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    da   = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    ic   = 0.0
    if len(y_pred) > 3:
        r, _ = spearmanr(y_pred, y_true)
        ic   = float(r) if not np.isnan(r) else 0.0
    return {"val_rmse": round(rmse, 6), "dir_acc": round(da, 4), "ic": round(ic, 4)}


def _val_split(y: np.ndarray, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Split series into 80/20 train/validation for held-out evaluation."""
    n = len(y)
    cut = max(int(n * ratio), 2)
    return y[:cut], y[cut:]


def _gaussian_ll(residuals: np.ndarray) -> float:
    """
    Log-likelihood under i.i.d. Gaussian assumption.

    LL = -n/2 * (1 + log(2*pi) + log(sigma^2))
    where sigma^2 = mean(residuals^2).  Used by OLS-fitted AR models.
    """
    n = len(residuals)
    if n < 2:
        return -1e10
    sigma2 = np.mean(residuals ** 2) + _TINY
    return -0.5 * n * (1.0 + np.log(2.0 * np.pi) + np.log(sigma2))


# ─────────────────────────────────────────────────────────────────────────────
# AR(p) — Autoregressive Model (OLS)
# ─────────────────────────────────────────────────────────────────────────────

class AR:
    """
    Autoregressive model of order p, fitted by Ordinary Least Squares.

        y_t = mu + phi_1 * y_{t-1} + phi_2 * y_{t-2} + ... + phi_p * y_{t-p} + eps_t

    BIC candidate search range: p in {1, 2, 3, 4, 5}  (5 candidates).

    Parameters
    ----------
    p : int
        Number of autoregressive lags (the "order").
        During auto-selection the bundle tries p = 1 through 5 and picks the
        one with lowest BIC.

    Attributes after fit
    --------------------
    _coef   : array of shape (p+1,) — [mu, phi_1, ..., phi_p]
    _ll     : float — Gaussian log-likelihood on the full data (for BIC)
    _n_obs  : int   — number of fitted residuals = len(y) - p
    """

    def __init__(self, p: int = 2):
        self.p = p
        self._coef: np.ndarray | None = None
        self.last_metrics: dict = {}
        self.name = f"AR({p})"
        self._ll: float | None = None
        self._n_obs: int = 0

    @property
    def n_params(self) -> int:
        """k = p + 1 (intercept mu + p lag coefficients)."""
        return self.p + 1

    @property
    def aic(self) -> float:
        """AIC = 2k - 2*LL.  Lower is better."""
        if self._ll is None:
            return np.inf
        return 2 * self.n_params - 2 * self._ll

    @property
    def bic(self) -> float:
        """BIC = k*ln(n) - 2*LL.  Penalises complexity more than AIC."""
        if self._ll is None or self._n_obs < 1:
            return np.inf
        return self.n_params * np.log(self._n_obs) - 2 * self._ll

    def _fit_ols(self, y: np.ndarray):
        """Fit coefficients [mu, phi_1..phi_p] via OLS on y."""
        n = len(y)
        if n <= self.p + 1:
            self._coef = np.zeros(self.p + 1)
            return
        X = np.column_stack(
            [np.ones(n - self.p)] +
            [y[self.p - k - 1: n - k - 1] for k in range(self.p)]
        )
        yy = y[self.p:]
        coef, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        self._coef = coef

    def _compute_ll(self, y: np.ndarray):
        """Store Gaussian LL and n_obs from OLS residuals (used for AIC/BIC)."""
        if self._coef is None or len(y) <= self.p:
            self._ll = -1e10
            self._n_obs = 0
            return
        X = np.column_stack(
            [np.ones(len(y) - self.p)] +
            [y[self.p - k - 1: len(y) - k - 1] for k in range(self.p)]
        )
        residuals = y[self.p:] - X @ self._coef
        self._n_obs = len(residuals)
        self._ll = _gaussian_ll(residuals)

    def _predict_insample(self, y: np.ndarray, start: int) -> np.ndarray:
        n = len(y)
        preds = []
        for t in range(start, n):
            if t < self.p:
                preds.append(float(np.mean(y[:t])) if t > 0 else 0.0)
            else:
                lags = y[t - self.p: t][::-1]
                preds.append(float(self._coef[0] + self._coef[1:] @ lags))
        return np.array(preds)

    def fit(self, y: np.ndarray) -> "AR":
        """
        Three-stage fit:
          1. Fit OLS on 80% train split.
          2. Evaluate on 20% validation -> store last_metrics (RMSE, dir_acc, IC).
          3. Refit OLS on full series -> store LL & n_obs for BIC computation.
        """
        y = np.asarray(y, dtype=float)
        y_tr, y_val = _val_split(y)
        self._fit_ols(y_tr)
        y_pred_val = self._predict_insample(y, start=len(y_tr))
        self.last_metrics = _rolling_metrics(y_val, y_pred_val)
        self._fit_ols(y)
        self._compute_ll(y)
        return self

    def predict_next(self, y: np.ndarray) -> float:
        """One-step-ahead forecast using the p most recent observations."""
        if self._coef is None:
            return 0.0
        y = np.asarray(y, dtype=float)
        lags = y[-self.p:][::-1] if len(y) >= self.p else np.pad(y, (self.p - len(y), 0))[::-1]
        return float(self._coef[0] + self._coef[1:] @ lags)

    def predict_array(self, y: np.ndarray) -> np.ndarray:
        return self._predict_insample(y, start=self.p)

    def conditional_vol(self, y: np.ndarray) -> float:
        return float(np.std(y[-12:]) if len(y) >= 12 else np.std(y))


# ─────────────────────────────────────────────────────────────────────────────
# ARMA(p,q) — Autoregressive Moving Average (conditional MLE)
# ─────────────────────────────────────────────────────────────────────────────

class ARMA:
    """
    Autoregressive Moving-Average model of order (p, q), fitted by conditional MLE.

        y_t = mu + phi_1*y_{t-1} + ... + phi_p*y_{t-p}
                 + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q} + eps_t

    The MA terms allow the model to absorb transient shocks that pure AR misses.

    BIC candidate search range: (p, q) in {(1,1), (2,1), (1,2), (2,2)}
    — 4 candidates, covering AR lags 1-2 and MA lags 1-2.

    Parameters
    ----------
    p : int   — number of AR lags (max tested: 2)
    q : int   — number of MA lags (max tested: 2)

    Fitted parameter vector: [mu, phi_1..phi_p, theta_1..theta_q, log_sigma]
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self._params: np.ndarray | None = None
        self.last_metrics: dict = {}
        self.name = f"ARMA({p},{q})"
        self._ll: float | None = None
        self._n_obs: int = 0

    @property
    def n_params(self) -> int:
        """k = 1 (mu) + p (AR) + q (MA) + 1 (sigma) = p + q + 2."""
        return 1 + self.p + self.q + 1

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self._ll if self._ll is not None else np.inf

    @property
    def bic(self) -> float:
        if self._ll is None or self._n_obs < 1:
            return np.inf
        return self.n_params * np.log(self._n_obs) - 2 * self._ll

    def _negll(self, params: np.ndarray, y: np.ndarray) -> float:
        """Conditional negative log-likelihood with constant variance."""
        mu = params[0]
        phis = params[1: 1 + self.p]
        thetas = params[1 + self.p: 1 + self.p + self.q]
        log_sig = params[-1]
        sig2 = np.exp(2 * log_sig) + _TINY
        n = len(y)
        eps = np.zeros(n)
        ll = 0.0
        start = max(self.p, 1)
        for t in range(start, n):
            ar = sum(phis[j] * y[t - 1 - j] for j in range(self.p) if t - 1 - j >= 0)
            ma = sum(thetas[j] * eps[t - 1 - j] for j in range(self.q) if t - 1 - j >= 0)
            eps[t] = y[t] - mu - ar - ma
            ll -= 0.5 * (np.log(2 * np.pi * sig2) + eps[t] ** 2 / sig2)
        return -ll

    def _fit_on(self, y: np.ndarray):
        """Minimise neg-LL via L-BFGS-B with bounded phi/theta in (-0.99, 0.99)."""
        mu0 = float(np.mean(y))
        sig0 = float(np.std(y))
        x0 = np.array([mu0] + [0.1] * self.p + [0.1] * self.q +
                       [np.log(max(sig0, _TINY))])
        bnds = ([(-1.0, 1.0)] +
                [(-0.99, 0.99)] * self.p +
                [(-0.99, 0.99)] * self.q +
                [(-10.0, 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(self._negll, x0, args=(y,), method="L-BFGS-B",
                           bounds=bnds, options={"maxiter": _MAX_ITER, "ftol": 1e-8})
        self._params = res.x

    def _predict_array(self, y: np.ndarray, start: int) -> np.ndarray:
        mu = self._params[0]
        phis = self._params[1: 1 + self.p]
        thetas = self._params[1 + self.p: 1 + self.p + self.q]
        n = len(y)
        eps = np.zeros(n)
        s0 = max(self.p, 1)
        for t in range(s0, min(start, n)):
            ar = sum(phis[j] * y[t - 1 - j] for j in range(self.p) if t - 1 - j >= 0)
            ma = sum(thetas[j] * eps[t - 1 - j] for j in range(self.q) if t - 1 - j >= 0)
            eps[t] = y[t] - mu - ar - ma
        preds = []
        for t in range(start, n):
            ar = sum(phis[j] * y[t - 1 - j] for j in range(self.p) if t - 1 - j >= 0)
            ma = sum(thetas[j] * eps[t - 1 - j] for j in range(self.q) if t - 1 - j >= 0)
            pred = mu + ar + ma
            eps[t] = y[t] - pred
            preds.append(pred)
        return np.array(preds)

    def fit(self, y: np.ndarray) -> "ARMA":
        """
        Three-stage fit (same pattern as AR):
          1. Fit MLE on 80% train -> 2. Evaluate on 20% val -> 3. Refit on full.
        After stage 3, LL and n_obs stored for BIC.
        """
        y = np.asarray(y, dtype=float)
        y_tr, y_val = _val_split(y)
        self._fit_on(y_tr)
        pred_val = self._predict_array(y, start=len(y_tr))
        self.last_metrics = _rolling_metrics(y_val, pred_val)
        self._fit_on(y)
        self._ll = -self._negll(self._params, y)
        self._n_obs = len(y) - max(self.p, 1)
        return self

    def predict_next(self, y: np.ndarray) -> float:
        """One-step-ahead forecast; reconstructs residual history for MA terms."""
        if self._params is None:
            return 0.0
        mu = self._params[0]
        phis = self._params[1: 1 + self.p]
        thetas = self._params[1 + self.p: 1 + self.p + self.q]
        y = np.asarray(y, dtype=float)
        n = len(y)
        eps = np.zeros(n)
        s0 = max(self.p, 1)
        for t in range(s0, n):
            ar = sum(phis[j] * y[t - 1 - j] for j in range(self.p) if t - 1 - j >= 0)
            ma = sum(thetas[j] * eps[t - 1 - j] for j in range(self.q) if t - 1 - j >= 0)
            eps[t] = y[t] - mu - ar - ma
        ar_n = sum(phis[j] * y[-(1 + j)] for j in range(self.p) if len(y) > j)
        ma_n = sum(thetas[j] * eps[-(1 + j)] for j in range(self.q) if len(eps) > j)
        return float(mu + ar_n + ma_n)

    def predict_array(self, y: np.ndarray) -> np.ndarray:
        return self._predict_array(y, start=max(self.p, 1))

    def conditional_vol(self, y: np.ndarray) -> float:
        if self._params is None:
            return float(np.std(y))
        return float(np.exp(self._params[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# ARCH(q) — Autoregressive Conditional Heteroskedasticity
# ─────────────────────────────────────────────────────────────────────────────

class ARCH:
    """
    ARCH(q) — Engle (1982).  Conditional heteroskedasticity model.

    Mean equation:     y_t = mu + phi * y_{t-1} + eps_t
    Variance equation: sigma_t^2 = omega + alpha_1*eps_{t-1}^2 + ... + alpha_q*eps_{t-q}^2

    Volatility responds only to the magnitude of past shocks (no persistence
    term unlike GARCH).  Fitted by conditional MLE.

    BIC candidate search range: q in {1, 2}  (2 candidates).

    Parameters
    ----------
    q : int — number of lagged squared-residual terms (max tested: 2)

    Fitted parameter vector: [mu, phi, omega, alpha_1, ..., alpha_q]
    """

    def __init__(self, q: int = 1):
        self.q = q
        self._params: np.ndarray | None = None
        self.last_metrics: dict = {}
        self.name = f"ARCH({q})"
        self._ll: float | None = None
        self._n_obs: int = 0

    @property
    def n_params(self) -> int:
        """k = 2 (mu, phi) + 1 (omega) + q (alphas) = q + 3."""
        return 2 + 1 + self.q

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self._ll if self._ll is not None else np.inf

    @property
    def bic(self) -> float:
        if self._ll is None or self._n_obs < 1:
            return np.inf
        return self.n_params * np.log(self._n_obs) - 2 * self._ll

    def _compute_vars(self, y, mu, phi, omega, alphas):
        n = len(y)
        eps = np.zeros(n)
        sig = np.full(n, np.std(y) ** 2 + _TINY)
        eps[0] = y[0] - mu
        for t in range(1, n):
            eps[t] = y[t] - mu - phi * y[t - 1]
            arch_part = sum(alphas[j] * eps[t - 1 - j] ** 2
                           for j in range(self.q) if t - 1 - j >= 0)
            sig[t] = max(omega + arch_part, _TINY)
        return eps, sig

    def _negll(self, params, y):
        mu, phi = params[0], params[1]
        omega = abs(params[2]) + _TINY
        alphas = np.abs(params[3: 3 + self.q])
        eps, sig2 = self._compute_vars(y, mu, phi, omega, alphas)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sig2[1:]) + eps[1:] ** 2 / sig2[1:])
        return -ll

    def _fit_on(self, y):
        mu0 = float(np.mean(y))
        om0 = float(np.var(y) * 0.5) + _TINY
        x0 = np.concatenate([[mu0, 0.1, om0], np.full(self.q, 0.1)])
        bnds = [(-0.5, 0.5), (-0.99, 0.99),
                (1e-7, float(np.var(y)) * 5)] + [(0.0, 0.98)] * self.q
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(self._negll, x0, args=(y,), method="L-BFGS-B",
                           bounds=bnds, options={"maxiter": _MAX_ITER})
        self._params = res.x

    def _predict_mean_array(self, y, start):
        mu, phi = self._params[0], self._params[1]
        return np.array([mu + phi * y[t - 1] for t in range(start, len(y))])

    def fit(self, y: np.ndarray) -> "ARCH":
        y = np.asarray(y, dtype=float)
        y_tr, y_val = _val_split(y)
        self._fit_on(y_tr)
        pred_val = self._predict_mean_array(y, start=len(y_tr))
        self.last_metrics = _rolling_metrics(y_val, pred_val)
        self._fit_on(y)
        self._ll = -self._negll(self._params, y)
        self._n_obs = len(y) - 1
        return self

    def predict_next(self, y):
        if self._params is None:
            return 0.0
        return float(self._params[0] + self._params[1] * y[-1])

    def predict_array(self, y):
        return self._predict_mean_array(y, start=1)

    def conditional_vol(self, y):
        if self._params is None:
            return float(np.std(y))
        mu, phi = self._params[0], self._params[1]
        omega = abs(self._params[2]) + _TINY
        alphas = np.abs(self._params[3: 3 + self.q])
        eps, sig2 = self._compute_vars(y, mu, phi, omega, alphas)
        next_sig2 = omega + sum(alphas[j] * eps[-1 - j] ** 2
                                for j in range(self.q) if len(eps) > j)
        return float(np.sqrt(max(next_sig2, _TINY)))


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(n_arch, n_garch) — Generalised ARCH
# ─────────────────────────────────────────────────────────────────────────────

class GARCH:
    """
    Generalised ARCH — Bollerslev (1986).

    Mean equation:
        y_t = mu + phi * y_{t-1} + eps_t

    Variance equation:
        sigma_t^2 = omega
                  + alpha_1*eps_{t-1}^2 + ... + alpha_{n_arch}*eps_{t-n_arch}^2
                  + beta_1*sigma_{t-1}^2 + ... + beta_{n_garch}*sigma_{t-n_garch}^2

    Stationarity: sum(alpha_i) + sum(beta_j) < 1.
    The beta terms add volatility persistence — once vol rises, it stays elevated.
    Fitted by conditional MLE with stationarity enforced via penalty.

    BIC candidate search range: (n_arch, n_garch) in {(1,1), (2,1)}
    — 2 candidates.  (1,1) is the industry standard; (2,1) adds a second
    ARCH lag to capture faster volatility response to recent shocks.

    Parameters
    ----------
    n_arch  : int — number of lagged eps^2 terms (max tested: 2)
    n_garch : int — number of lagged sigma^2 terms (max tested: 1)

    Fitted parameter vector: [mu, phi, omega, alpha_1..n_arch, beta_1..n_garch]
    """

    def __init__(self, n_arch: int = 1, n_garch: int = 1):
        self.n_arch = n_arch
        self.n_garch = n_garch
        self._params: np.ndarray | None = None
        self.last_metrics: dict = {}
        self.name = f"GARCH({n_arch},{n_garch})"
        self._ll: float | None = None
        self._n_obs: int = 0

    @property
    def n_params(self) -> int:
        """k = 2 (mu, phi) + 1 (omega) + n_arch (alphas) + n_garch (betas)."""
        return 2 + 1 + self.n_arch + self.n_garch

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self._ll if self._ll is not None else np.inf

    @property
    def bic(self) -> float:
        if self._ll is None or self._n_obs < 1:
            return np.inf
        return self.n_params * np.log(self._n_obs) - 2 * self._ll

    def _unpack(self, params):
        mu = params[0]
        phi = params[1]
        omega = abs(params[2]) + _TINY
        alphas = np.abs(params[3: 3 + self.n_arch])
        betas = np.abs(params[3 + self.n_arch: 3 + self.n_arch + self.n_garch])
        return mu, phi, omega, alphas, betas

    def _compute_vars(self, y, mu, phi, omega, alphas, betas):
        n = len(y)
        eps = np.zeros(n)
        sig2 = np.full(n, np.var(y) + _TINY)
        eps[0] = y[0] - mu
        for t in range(1, n):
            eps[t] = y[t] - mu - phi * y[t - 1]
            a_sum = sum(alphas[j] * eps[max(t - 1 - j, 0)] ** 2
                        for j in range(self.n_arch))
            g_sum = sum(betas[j] * sig2[max(t - 1 - j, 0)]
                        for j in range(self.n_garch))
            sig2[t] = max(omega + a_sum + g_sum, _TINY)
        return eps, sig2

    def _negll(self, params, y):
        mu, phi, omega, alphas, betas = self._unpack(params)
        if np.sum(alphas) + np.sum(betas) >= 1.0:
            return 1e10
        eps, sig2 = self._compute_vars(y, mu, phi, omega, alphas, betas)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sig2[1:]) + eps[1:] ** 2 / sig2[1:])
        return -ll if np.isfinite(ll) else 1e10

    def _fit_on(self, y):
        v0 = float(np.var(y))
        x0 = np.concatenate([
            [float(np.mean(y)), 0.05, v0 * 0.1],
            np.full(self.n_arch, 0.10),
            np.full(self.n_garch, 0.80 / max(self.n_garch, 1)),
        ])
        bnds = ([(-0.5, 0.5), (-0.99, 0.99), (1e-8, v0 * 10)] +
                [(1e-6, 0.50)] * self.n_arch +
                [(1e-6, 0.98)] * self.n_garch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(self._negll, x0, args=(y,), method="L-BFGS-B",
                           bounds=bnds, options={"maxiter": _MAX_ITER, "ftol": 1e-9})
        self._params = res.x

    def _mean_array(self, y, start):
        mu, phi = self._params[0], self._params[1]
        return np.array([mu + phi * y[t - 1] for t in range(start, len(y))])

    def fit(self, y: np.ndarray) -> "GARCH":
        y = np.asarray(y, dtype=float)
        y_tr, y_val = _val_split(y)
        self._fit_on(y_tr)
        pred_val = self._mean_array(y, start=len(y_tr))
        self.last_metrics = _rolling_metrics(y_val, pred_val)
        self._fit_on(y)
        self._ll = -self._negll(self._params, y)
        self._n_obs = len(y) - 1
        return self

    def predict_next(self, y):
        if self._params is None:
            return 0.0
        return float(self._params[0] + self._params[1] * y[-1])

    def predict_array(self, y):
        return self._mean_array(y, start=1)

    def conditional_vol(self, y):
        if self._params is None:
            return float(np.std(y))
        mu, phi, omega, alphas, betas = self._unpack(self._params)
        eps, sig2 = self._compute_vars(y, mu, phi, omega, alphas, betas)
        a_sum = sum(alphas[j] * eps[max(len(eps) - 1 - j, 0)] ** 2
                    for j in range(self.n_arch))
        g_sum = sum(betas[j] * sig2[max(len(sig2) - 1 - j, 0)]
                    for j in range(self.n_garch))
        return float(np.sqrt(max(omega + a_sum + g_sum, _TINY)))


# ─────────────────────────────────────────────────────────────────────────────
# EGARCH(n_arch, n_garch) — Exponential GARCH (Nelson 1991)
# ─────────────────────────────────────────────────────────────────────────────

class EGARCH:
    """
    Exponential GARCH — Nelson (1991).  Models log-variance, allowing leverage.

    Mean equation:
        y_t = mu + phi * y_{t-1} + eps_t

    Variance equation (log-scale):
        log(sigma_t^2) = omega
                       + sum_i alpha_i * (|z_{t-i}| - E|z|)   (magnitude)
                       + sum_i gamma_i * z_{t-i}               (leverage / sign)
                       + sum_j beta_j  * log(sigma_{t-j}^2)    (persistence)

    where z_t = eps_t / sigma_t is the standardised residual.

    The gamma terms capture the leverage effect: negative shocks (gamma < 0)
    raise volatility more than positive shocks of equal size.  Because the
    model operates in log-space, sigma^2 is always positive without needing
    non-negativity constraints on parameters.

    Stationarity: sum(|beta_j|) < 1, enforced via penalty in the neg-LL.

    BIC candidate search range: (n_arch, n_garch) in {(1,1), (2,1)}
    — 2 candidates.  (2,1) adds a second news-impact lag.

    Parameters
    ----------
    n_arch  : int — number of news-impact lags (alpha + gamma pairs; max tested: 2)
    n_garch : int — number of log-variance persistence lags (max tested: 1)

    Fitted parameter vector:
        [mu, phi, omega, alpha_1..n_arch, gamma_1..n_arch, beta_1..n_garch]
    """

    _EZ = float(np.sqrt(2.0 / np.pi))  # E[|z|] for standard normal

    def __init__(self, n_arch: int = 1, n_garch: int = 1):
        self.n_arch = n_arch
        self.n_garch = n_garch
        self._params: np.ndarray | None = None
        self.last_metrics: dict = {}
        self.name = f"EGARCH({n_arch},{n_garch})"
        self._ll: float | None = None
        self._n_obs: int = 0

    @property
    def n_params(self) -> int:
        """k = 2 (mu, phi) + 1 (omega) + 2*n_arch (alpha+gamma pairs) + n_garch (betas)."""
        return 2 + 1 + 2 * self.n_arch + self.n_garch

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self._ll if self._ll is not None else np.inf

    @property
    def bic(self) -> float:
        if self._ll is None or self._n_obs < 1:
            return np.inf
        return self.n_params * np.log(self._n_obs) - 2 * self._ll

    def _unpack(self, params):
        mu = params[0]
        phi = params[1]
        omega = params[2]
        alphas = params[3: 3 + self.n_arch]
        gammas = params[3 + self.n_arch: 3 + 2 * self.n_arch]
        betas = params[3 + 2 * self.n_arch: 3 + 2 * self.n_arch + self.n_garch]
        return mu, phi, omega, alphas, gammas, betas

    def _compute_vars(self, y, mu, phi, omega, alphas, gammas, betas):
        n = len(y)
        eps = np.zeros(n)
        log_h = np.full(n, np.log(np.var(y) + _TINY))
        eps[0] = y[0] - mu
        for t in range(1, n):
            eps[t] = y[t] - mu - phi * y[t - 1]
            news = 0.0
            for j in range(self.n_arch):
                idx = max(t - 1 - j, 0)
                sig_j = float(np.exp(0.5 * log_h[idx]))
                z_j = eps[idx] / max(sig_j, _TINY)
                news += alphas[j] * (abs(z_j) - self._EZ) + gammas[j] * z_j
            pers = sum(betas[j] * log_h[max(t - 1 - j, 0)]
                       for j in range(self.n_garch))
            log_h[t] = np.clip(omega + news + pers, -30.0, 30.0)
        return eps, np.exp(log_h)

    def _negll(self, params, y):
        mu, phi, omega, alphas, gammas, betas = self._unpack(params)
        if np.sum(np.abs(betas)) >= 1.0:
            return 1e10
        eps, sig2 = self._compute_vars(y, mu, phi, omega, alphas, gammas, betas)
        ll = -0.5 * np.sum(
            np.log(2 * np.pi * sig2[1:] + _TINY) + eps[1:] ** 2 / (sig2[1:] + _TINY)
        )
        return -ll if np.isfinite(ll) else 1e10

    def _fit_on(self, y):
        log_v = np.log(np.var(y) + _TINY)
        x0 = np.concatenate([
            [float(np.mean(y)), 0.05, log_v * 0.05],
            np.full(self.n_arch, 0.15),    # alphas
            np.full(self.n_arch, -0.10),   # gammas (leverage)
            np.full(self.n_garch, 0.75 / max(self.n_garch, 1)),
        ])
        bnds = ([(-0.5, 0.5), (-0.99, 0.99), (-30.0, 5.0)] +
                [(-2.0, 2.0)] * self.n_arch +    # alphas
                [(-2.0, 2.0)] * self.n_arch +    # gammas
                [(-0.99, 0.99)] * self.n_garch)   # betas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(self._negll, x0, args=(y,), method="L-BFGS-B",
                           bounds=bnds, options={"maxiter": _MAX_ITER, "ftol": 1e-9})
        self._params = res.x

    def _mean_array(self, y, start):
        mu, phi = self._params[0], self._params[1]
        return np.array([mu + phi * y[t - 1] for t in range(start, len(y))])

    def fit(self, y: np.ndarray) -> "EGARCH":
        y = np.asarray(y, dtype=float)
        y_tr, y_val = _val_split(y)
        self._fit_on(y_tr)
        pred_val = self._mean_array(y, start=len(y_tr))
        self.last_metrics = _rolling_metrics(y_val, pred_val)
        self._fit_on(y)
        self._ll = -self._negll(self._params, y)
        self._n_obs = len(y) - 1
        return self

    def predict_next(self, y):
        if self._params is None:
            return 0.0
        return float(self._params[0] + self._params[1] * y[-1])

    def predict_array(self, y):
        return self._mean_array(y, start=1)

    def conditional_vol(self, y):
        if self._params is None:
            return float(np.std(y))
        mu, phi, omega, alphas, gammas, betas = self._unpack(self._params)
        eps, sig2 = self._compute_vars(y, mu, phi, omega, alphas, gammas, betas)
        n = len(eps)
        news = 0.0
        for j in range(self.n_arch):
            idx = max(n - 1 - j, 0)
            sig_j = float(np.sqrt(sig2[idx] + _TINY))
            z_j = eps[idx] / max(sig_j, _TINY)
            news += alphas[j] * (abs(z_j) - self._EZ) + gammas[j] * z_j
        pers = sum(betas[j] * np.log(sig2[max(n - 1 - j, 0)] + _TINY)
                   for j in range(self.n_garch))
        log_h_next = np.clip(omega + news + pers, -30.0, 30.0)
        return float(np.sqrt(np.exp(log_h_next) + _TINY))


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pools for BIC-based model selection
#
# For each of the 5 model families, we define a list of candidate
# parameterisations.  During auto-selection (ARForecastBundle.fit), every
# candidate in every family is fitted on the asset's return series and the
# one with the lowest BIC wins for that family.
#
# Total candidates per asset: 5 + 4 + 2 + 2 + 2 = 15 model fits.
# At ~0.5-2s per fit, total ≈ 8-30s per asset (acceptable).
#
# Family        Candidates                         Count
# ──────────    ─────────────────────────────────   ─────
# AR            p = 1, 2, 3, 4, 5                    5
# ARMA          (p,q) = (1,1), (2,1), (1,2), (2,2)  4
# ARCH          q = 1, 2                             2
# GARCH         (qa,qg) = (1,1), (2,1)               2
# EGARCH        (qa,qg) = (1,1), (2,1)               2
# ─────────────────────────────────────────────────────────────────────────────

_FAMILIES = {
    "AR":    (AR,    [dict(p=p) for p in range(1, 6)]),
    "ARMA":  (ARMA,  [dict(p=p, q=q) for p, q in [(1, 1), (2, 1), (1, 2), (2, 2)]]),
    "ARCH":  (ARCH,  [dict(q=q) for q in [1, 2]]),
    "GARCH": (GARCH, [dict(n_arch=a, n_garch=g) for a, g in [(1, 1), (2, 1)]]),
    "EGARCH":(EGARCH,[dict(n_arch=a, n_garch=g) for a, g in [(1, 1), (2, 1)]]),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-Asset AR Ensemble with BIC-based auto-selection
# ─────────────────────────────────────────────────────────────────────────────

class ARForecastBundle:
    """
    Per-asset ensemble of 5 BIC-selected time-series models.

    Pipeline (executed in .fit(y)):
    ───────────────────────────────
    For each of the 5 model families (AR, ARMA, ARCH, GARCH, EGARCH):

      a) Instantiate every candidate order defined in _FAMILIES.
         Example for AR:  AR(1), AR(2), AR(3), AR(4), AR(5).

      b) Call candidate.fit(y) on each.  Internally this:
           - Splits y into 80% train / 20% validation
           - Fits parameters on the training portion
           - Evaluates RMSE, dir_acc, IC on the validation portion
           - Refits on the full y
           - Computes log-likelihood -> AIC and BIC from the full-data fit

      c) Compare BIC values across all candidates in the family.
         Select the one with the LOWEST BIC (best complexity-fit trade-off).

    After all 5 families are processed, we have 5 winning models.

    Ensemble weighting:
      weight_m = (1 / val_MSE_m) / sum_j(1 / val_MSE_j)
    Models with lower validation error get higher weight.

    Stored metadata:
      self.selection_info[family] = {
          "selected": "AR(3)",              # winning model name
          "bic":      -245.6,               # its BIC
          "n_params": 4,                    # its parameter count
          "alternatives": [                 # all candidates tried
              {"name": "AR(1)", "bic": -240.1, "aic": -242.3},
              {"name": "AR(2)", "bic": -243.5, "aic": -246.0},
              ...
          ],
      }
    """

    def __init__(self, ar_order: int = 2):
        self._ar_order = ar_order
        self._models: dict[str, object] = {}
        self._weights: dict[str, float] = {}
        self._y_full: np.ndarray | None = None
        self.per_model_metrics: dict[str, dict] = {}
        self.selection_info: dict[str, dict] = {}

    def fit(self, y: np.ndarray) -> "ARForecastBundle":
        """Fit all ~15 candidates, select best per family by BIC, build ensemble."""
        y = np.asarray(y, dtype=float)
        self._y_full = y
        mse_scores: dict[str, float] = {}

        for family, (cls, candidates) in _FAMILIES.items():
            best_model = None
            best_bic = np.inf
            alternatives = []

            for kwargs in candidates:
                model = cls(**kwargs)
                try:
                    model.fit(y)
                    bic_val = model.bic
                    alternatives.append({"name": model.name, "bic": round(bic_val, 2),
                                          "aic": round(model.aic, 2)})
                    if bic_val < best_bic:
                        best_model = model
                        best_bic = bic_val
                except Exception as e:
                    logger.debug("Candidate %s failed for family %s: %s",
                                 kwargs, family, e)
                    alternatives.append({"name": f"{family}({kwargs})", "bic": np.inf,
                                          "aic": np.inf})

            if best_model is not None:
                name = best_model.name
                self._models[name] = best_model
                mse_scores[name] = max(
                    best_model.last_metrics.get("val_rmse", 1.0) ** 2, _TINY
                )
                self.per_model_metrics[name] = best_model.last_metrics.copy()
                self.selection_info[family] = {
                    "selected": name,
                    "bic": round(best_bic, 2),
                    "n_params": best_model.n_params,
                    "alternatives": alternatives,
                }
            else:
                logger.warning("No model succeeded for family %s", family)

        inv = {k: 1.0 / (v + _TINY) for k, v in mse_scores.items()}
        total = sum(inv.values()) or 1.0
        self._weights = {k: v / total for k, v in inv.items()}
        return self

    def predict_next(self, y: np.ndarray) -> float:
        """Weighted-average one-step-ahead forecast from the 5 winning models."""
        total = 0.0
        for name, model in self._models.items():
            try:
                total += self._weights.get(name, 0.0) * model.predict_next(y)
            except Exception:
                pass
        return float(total)

    def predict_array(self, y: np.ndarray) -> np.ndarray:
        """Weighted-average in-sample predictions (used for ensemble metrics)."""
        preds = None
        for name, model in self._models.items():
            try:
                arr = model.predict_array(y)
                w = self._weights.get(name, 0.0)
                if preds is None:
                    preds = np.zeros(len(arr))
                n = min(len(preds), len(arr))
                preds[:n] += w * arr[:n]
            except Exception:
                pass
        return preds if preds is not None else np.zeros(len(y) - 1)

    def conditional_vol(self, y: np.ndarray) -> dict[str, float]:
        """Per-model conditional volatility forecasts (used as risk signals)."""
        vols = {}
        for name, model in self._models.items():
            try:
                vols[name] = model.conditional_vol(y)
            except Exception:
                vols[name] = float(np.std(y))
        return vols

    @property
    def weights(self) -> dict[str, float]:
        return self._weights


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset AR Forecaster (mirrors ReturnForecaster API)
# ─────────────────────────────────────────────────────────────────────────────

class ARForecaster:
    """
    Multi-asset AR forecaster — mirrors the ReturnForecaster API.

    For each asset column in the provided monthly returns DataFrame:
      1. Extract the asset's return series as a 1-D numpy array.
      2. Create an ARForecastBundle and call .fit(y).
         This runs ~15 candidate model fits and selects the best per family
         by BIC, then builds an inverse-MSE weighted ensemble.
      3. Store the bundle and compute ensemble-level metrics (MSE, RMSE,
         directional accuracy, Spearman IC) on the full in-sample predictions.

    Key methods
    -----------
    fit(monthly_returns)      : Train all bundles.
    predict_next(monthly_returns) : One-step-ahead forecast per asset.
    metrics_summary()         : DataFrame of ensemble accuracy per asset.
    per_model_summary()       : DataFrame of per-model metrics & weights.
    selection_summary()       : DataFrame showing BIC-selected order per
                                asset per family (the main new addition).
    conditional_vols(monthly_returns) : Per-model conditional vol forecasts.
    """

    def __init__(self, assets: list[str], ar_order: int = 2):
        import pandas as pd
        self.assets = assets
        self.ar_order = ar_order
        self._bundles: dict[str, ARForecastBundle] = {}
        self._ensemble_metrics: dict[str, dict] = {}

    def fit(self, monthly_returns) -> "ARForecaster":
        """
        Fit one ARForecastBundle per asset (BIC auto-selection + ensemble).

        Requires at least 20 monthly observations per asset; assets with
        fewer are skipped with a warning.
        """
        import pandas as pd
        logger.info("Fitting AR models (BIC auto-selection) for %d assets ...",
                     len(self.assets))
        for asset in self.assets:
            if asset not in monthly_returns.columns:
                continue
            y = monthly_returns[asset].dropna().values
            if len(y) < 20:
                logger.warning("AR: too few observations for %s (%d)", asset, len(y))
                continue
            bundle = ARForecastBundle(ar_order=self.ar_order)
            bundle.fit(y)
            self._bundles[asset] = bundle

            preds = bundle.predict_array(y)
            y_true = y[len(y) - len(preds):]
            mse = float(np.mean((y_true - preds) ** 2))
            da = float(np.mean(np.sign(preds) == np.sign(y_true)))
            ic, _ = spearmanr(preds, y_true)
            self._ensemble_metrics[asset] = {
                "mse": round(mse, 6),
                "rmse": round(float(np.sqrt(mse)), 6),
                "dir_accuracy": round(da, 4),
                "ic": round(float(ic) if not np.isnan(ic) else 0.0, 4),
            }
            selected = [bundle.selection_info[f]["selected"]
                        for f in _FAMILIES if f in bundle.selection_info]
            logger.info("  %s -> selected: %s", asset, ", ".join(selected))
        return self

    def predict_next(self, monthly_returns) -> "pd.Series":
        """One-step-ahead ensemble forecast per asset -> pd.Series."""
        import pandas as pd
        preds = {}
        for asset, bundle in self._bundles.items():
            y = monthly_returns[asset].dropna().values
            preds[asset] = bundle.predict_next(y)
        return pd.Series(preds)

    def metrics_summary(self) -> "pd.DataFrame":
        """Ensemble-level MSE, RMSE, dir_accuracy, IC per asset."""
        import pandas as pd
        return pd.DataFrame(self._ensemble_metrics).T

    def per_model_summary(self) -> "pd.DataFrame":
        """Per-model (the 5 BIC winners) val_rmse, dir_acc, IC, weight per asset."""
        import pandas as pd
        rows = {}
        for asset, bundle in self._bundles.items():
            row = {}
            for mn, m in bundle.per_model_metrics.items():
                for k, v in m.items():
                    row[f"{mn}_{k}"] = v
            for mn, w in bundle.weights.items():
                row[f"{mn}_weight"] = round(w, 4)
            rows[asset] = row
        return pd.DataFrame(rows).T

    def selection_summary(self) -> "pd.DataFrame":
        """
        Per-asset, per-family: which order was selected and its BIC.

        Returns a DataFrame with columns like:
            AR_selected, AR_bic, AR_n_params,
            ARMA_selected, ARMA_bic, ARMA_n_params,
            ARCH_selected, ARCH_bic, ARCH_n_params,
            GARCH_selected, GARCH_bic, GARCH_n_params,
            EGARCH_selected, EGARCH_bic, EGARCH_n_params
        Rows are indexed by asset name.
        """
        import pandas as pd
        rows = {}
        for asset, bundle in self._bundles.items():
            row = {}
            for family, info in bundle.selection_info.items():
                row[f"{family}_selected"] = info["selected"]
                row[f"{family}_bic"] = info["bic"]
                row[f"{family}_n_params"] = info["n_params"]
            rows[asset] = row
        return pd.DataFrame(rows).T

    def conditional_vols(self, monthly_returns) -> "pd.DataFrame":
        """Per-model conditional volatility forecasts for each asset."""
        import pandas as pd
        result = {}
        for asset, bundle in self._bundles.items():
            y = monthly_returns[asset].dropna().values
            result[asset] = bundle.conditional_vol(y)
        return pd.DataFrame(result).T
