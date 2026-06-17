"""Maximum likelihood estimation of short rate models.

This module provides :class:`MLECalibrator`, which fits the Vasicek and
Cox-Ingersoll-Ross (CIR) one-factor short rate models to an observed time
series of the short rate via maximum likelihood.

For the Vasicek model the conditional transition density is Gaussian in closed
form, so the (negative) log-likelihood is a sum of Gaussian log-densities. For
the CIR model the exact transition law is a scaled non-central chi-squared
distribution, evaluated through :func:`scipy.stats.ncx2.logpdf`.

Information criteria (AIC, BIC) and asymptotic standard errors (from the
inverse Hessian of the log-likelihood) accompany each fit.

References
----------
Vasicek, O. (1977). *Journal of Financial Economics*, 5(2), 177-188.
Cox, J., Ingersoll, J., Ross, S. (1985). *Econometrica*, 53(2), 385-407.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import optimize, stats

from models import CIRParams, VasicekParams

# Numerical guards -----------------------------------------------------------
_MIN_KAPPA = 1e-6
_MIN_SIGMA = 1e-8
_MIN_THETA = 1e-8


def _numerical_hessian(
    func, x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Central finite-difference Hessian of a scalar function.

    Parameters
    ----------
    func : callable
        Scalar function of a 1-D parameter vector.
    x : np.ndarray
        Point at which to evaluate the Hessian.
    eps : float, optional
        Base relative step size.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)`` Hessian matrix.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n))
    steps = eps * np.maximum(np.abs(x), 1.0)

    f0 = func(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy()
            if i == j:
                xi[i] = x[i] + steps[i]
                fp = func(xi)
                xi[i] = x[i] - steps[i]
                fm = func(xi)
                hess[i, i] = (fp - 2.0 * f0 + fm) / steps[i] ** 2
            else:
                xpp = x.copy(); xpp[i] += steps[i]; xpp[j] += steps[j]
                xpm = x.copy(); xpm[i] += steps[i]; xpm[j] -= steps[j]
                xmp = x.copy(); xmp[i] -= steps[i]; xmp[j] += steps[j]
                xmm = x.copy(); xmm[i] -= steps[i]; xmm[j] -= steps[j]
                val = (
                    func(xpp) - func(xpm) - func(xmp) + func(xmm)
                ) / (4.0 * steps[i] * steps[j])
                hess[i, j] = val
                hess[j, i] = val
    return hess


def _std_errors_from_hessian(hess: np.ndarray) -> np.ndarray:
    """Asymptotic standard errors from the Hessian of the negative log-lik.

    The inverse of the observed information matrix (the Hessian of the negative
    log-likelihood) gives the asymptotic covariance. Non-finite or negative
    variances are returned as ``nan``.
    """
    try:
        cov = np.linalg.inv(hess)
        var = np.diag(cov)
        se = np.where(var > 0, np.sqrt(np.abs(var)), np.nan)
    except np.linalg.LinAlgError:
        se = np.full(hess.shape[0], np.nan)
    return se


def _vasicek_neg_loglik(params: np.ndarray, r: np.ndarray, dt: float) -> float:
    """Negative Gaussian conditional log-likelihood of the Vasicek model.

    Parameters
    ----------
    params : np.ndarray
        ``[kappa, theta, sigma]``.
    r : np.ndarray
        Observed short rate series.
    dt : float
        Time step between observations (years).

    Returns
    -------
    float
        Negative log-likelihood (``+inf`` for invalid parameters).
    """
    kappa, theta, sigma = float(params[0]), float(params[1]), float(params[2])
    if kappa <= _MIN_KAPPA or sigma <= _MIN_SIGMA:
        return np.inf

    r_prev = r[:-1]
    r_next = r[1:]
    e = np.exp(-kappa * dt)
    mean = r_prev * e + theta * (1.0 - e)
    var = (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * dt))
    if var <= 0 or not np.isfinite(var):
        return np.inf

    ll = -0.5 * (
        np.log(2.0 * np.pi * var) + (r_next - mean) ** 2 / var
    )
    total = float(np.sum(ll))
    if not np.isfinite(total):
        return np.inf
    return -total


def _cir_neg_loglik(params: np.ndarray, r: np.ndarray, dt: float) -> float:
    """Negative log-likelihood of the CIR model via non-central chi-squared.

    Using the exact transition law, ``2 c r(t) ~ ncx2(df, ncp)`` with
    ``c = 2 kappa / (sigma^2 (1 - e^{-kappa dt}))``,
    ``df = 4 kappa theta / sigma^2`` and ``ncp = 2 c r(s) e^{-kappa dt}``.
    The log-density of ``r(t)`` is ``log(2c) + ncx2.logpdf(2 c r(t), df, ncp)``.

    Parameters
    ----------
    params : np.ndarray
        ``[kappa, theta, sigma]``.
    r : np.ndarray
        Observed short rate series (assumed strictly positive).
    dt : float
        Time step between observations (years).

    Returns
    -------
    float
        Negative log-likelihood (``+inf`` for invalid parameters).
    """
    kappa, theta, sigma = float(params[0]), float(params[1]), float(params[2])
    if kappa <= _MIN_KAPPA or theta <= _MIN_THETA or sigma <= _MIN_SIGMA:
        return np.inf

    r_prev = r[:-1]
    r_next = r[1:]
    if np.any(r_next <= 0.0):
        return np.inf

    e = np.exp(-kappa * dt)
    c = 2.0 * kappa / (sigma**2 * (1.0 - e))
    if not np.isfinite(c) or c <= 0:
        return np.inf
    df = 4.0 * kappa * theta / sigma**2
    ncp = 2.0 * c * r_prev * e

    logpdf = np.log(2.0 * c) + stats.ncx2.logpdf(2.0 * c * r_next, df, ncp)
    total = float(np.sum(logpdf))
    if not np.isfinite(total):
        return np.inf
    return -total


class MLECalibrator:
    """Maximum likelihood calibrator for the Vasicek and CIR models.

    Examples
    --------
    >>> from models import VasicekModel, VasicekParams
    >>> sim = VasicekModel(VasicekParams(0.5, 0.05, 0.02)).simulate_exact(
    ...     0.04, 20.0, 1, 5040, seed=0)
    >>> cal = MLECalibrator()
    >>> res = cal.calibrate_vasicek(sim.paths[0], dt=1.0 / 252)
    >>> isinstance(res["params"], VasicekParams)
    True
    """

    # ----------------------------------------------------------- Vasicek
    def calibrate_vasicek(self, rate_series: np.ndarray, dt: float) -> Dict:
        """Calibrate the Vasicek model by exact Gaussian MLE.

        Parameters
        ----------
        rate_series : np.ndarray
            Observed short rate time series.
        dt : float
            Time step between observations (years).

        Returns
        -------
        dict
            Keys: ``params`` (:class:`VasicekParams`), ``std_errors`` (dict),
            ``log_likelihood``, ``aic``, ``bic``, ``success``, ``n_obs``.
        """
        r = np.asarray(rate_series, dtype=float).ravel()
        n = r.size - 1
        if n < 3:
            raise ValueError("Need at least 4 observations to calibrate.")

        x0 = self._vasicek_initial_guess(r, dt)
        bounds = [(_MIN_KAPPA, None), (None, None), (_MIN_SIGMA, None)]

        res = optimize.minimize(
            _vasicek_neg_loglik,
            x0,
            args=(r, dt),
            method="L-BFGS-B",
            bounds=bounds,
        )
        kappa, theta, sigma = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        log_lik = -float(res.fun)

        hess = _numerical_hessian(
            lambda p: _vasicek_neg_loglik(p, r, dt), res.x
        )
        se = _std_errors_from_hessian(hess)

        k = 3
        aic = -2.0 * log_lik + 2.0 * k
        bic = -2.0 * log_lik + k * np.log(n)

        return {
            "params": VasicekParams(
                kappa=max(kappa, _MIN_KAPPA),
                theta=theta,
                sigma=max(sigma, _MIN_SIGMA),
            ),
            "std_errors": {
                "kappa": float(se[0]),
                "theta": float(se[1]),
                "sigma": float(se[2]),
            },
            "log_likelihood": log_lik,
            "aic": float(aic),
            "bic": float(bic),
            "success": bool(res.success),
            "n_obs": int(n),
        }

    @staticmethod
    def _vasicek_initial_guess(r: np.ndarray, dt: float) -> np.ndarray:
        """Moment / OLS based starting values for the optimiser."""
        r_prev = r[:-1]
        r_next = r[1:]
        # OLS of r_next on r_prev: r_next = a + b r_prev + eps.
        X = np.column_stack([np.ones_like(r_prev), r_prev])
        coef, *_ = np.linalg.lstsq(X, r_next, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        b = min(max(b, 1e-6), 1.0 - 1e-6)
        kappa = -np.log(b) / dt
        theta = a / (1.0 - b)
        resid = r_next - (a + b * r_prev)
        var_resid = float(np.var(resid))
        denom = (1.0 - b**2) / (2.0 * kappa)
        sigma = np.sqrt(max(var_resid / max(denom, 1e-12), 1e-8))
        return np.array([
            max(kappa, 10.0 * _MIN_KAPPA),
            theta,
            max(sigma, 10.0 * _MIN_SIGMA),
        ])

    # --------------------------------------------------------------- CIR
    def calibrate_cir(self, rate_series: np.ndarray, dt: float) -> Dict:
        """Calibrate the CIR model by exact non-central chi-squared MLE.

        Parameters
        ----------
        rate_series : np.ndarray
            Observed short rate time series (should be strictly positive).
        dt : float
            Time step between observations (years).

        Returns
        -------
        dict
            Keys: ``params`` (:class:`CIRParams`), ``std_errors`` (dict),
            ``log_likelihood``, ``aic``, ``bic``, ``feller`` (dict),
            ``success``, ``n_obs``.
        """
        r = np.asarray(rate_series, dtype=float).ravel()
        # CIR requires positive rates; clip tiny non-positive values.
        r = np.where(r <= 0.0, _MIN_THETA, r)
        n = r.size - 1
        if n < 3:
            raise ValueError("Need at least 4 observations to calibrate.")

        x0 = self._cir_initial_guess(r, dt)
        bounds = [
            (_MIN_KAPPA, None),
            (_MIN_THETA, None),
            (_MIN_SIGMA, None),
        ]

        res = optimize.minimize(
            _cir_neg_loglik,
            x0,
            args=(r, dt),
            method="L-BFGS-B",
            bounds=bounds,
        )
        kappa, theta, sigma = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        log_lik = -float(res.fun)

        hess = _numerical_hessian(lambda p: _cir_neg_loglik(p, r, dt), res.x)
        se = _std_errors_from_hessian(hess)

        k = 3
        aic = -2.0 * log_lik + 2.0 * k
        bic = -2.0 * log_lik + k * np.log(n)

        lhs = 2.0 * kappa * theta
        rhs = sigma**2
        feller = {
            "holds": bool(lhs > rhs),
            "lhs": float(lhs),
            "rhs": float(rhs),
            "ratio": float(lhs / rhs) if rhs > 0 else float("inf"),
        }

        return {
            "params": CIRParams(
                kappa=max(kappa, _MIN_KAPPA),
                theta=max(theta, _MIN_THETA),
                sigma=max(sigma, _MIN_SIGMA),
            ),
            "std_errors": {
                "kappa": float(se[0]),
                "theta": float(se[1]),
                "sigma": float(se[2]),
            },
            "log_likelihood": log_lik,
            "aic": float(aic),
            "bic": float(bic),
            "feller": feller,
            "success": bool(res.success),
            "n_obs": int(n),
        }

    @staticmethod
    def _cir_initial_guess(r: np.ndarray, dt: float) -> np.ndarray:
        """Pearson-Sun style starting values for the CIR optimiser."""
        r_prev = r[:-1]
        r_next = r[1:]
        dr = r_next - r_prev
        sqrt_r = np.sqrt(np.maximum(r_prev, 1e-10))
        # Regress dr / sqrt(r) on 1/sqrt(r) and sqrt(r).
        y = dr / sqrt_r
        X = np.column_stack([1.0 / sqrt_r, sqrt_r])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b1, b2 = float(coef[0]), float(coef[1])
        kappa = -b2 / dt
        if kappa <= 0:
            kappa = 0.5
            theta = float(np.mean(r))
        else:
            theta = (b1 / dt) / kappa
        if theta <= 0:
            theta = float(np.mean(r))
        resid = y - X @ coef
        sigma = float(np.std(resid) / np.sqrt(dt))
        return np.array([
            max(kappa, 10.0 * _MIN_KAPPA),
            max(theta, 10.0 * _MIN_THETA),
            max(sigma, 10.0 * _MIN_SIGMA),
        ])
