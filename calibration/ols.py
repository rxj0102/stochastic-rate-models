"""Ordinary least squares calibration of short rate models.

This module provides :class:`OLSCalibrator`, which fits the Vasicek and CIR
models by linear regression of the discretised SDE. These estimators are fast
and provide good starting values for maximum likelihood; they are biased in
small samples but consistent as ``dt -> 0``.

Vasicek
    The Euler discretisation gives
    ``r_{t+dt} - r_t = kappa theta dt - kappa dt r_t + sigma sqrt(dt) eps``.
    Regressing ``Delta r`` on ``r_t`` yields ``Delta r = alpha + beta r_t`` so
    ``kappa = -beta / dt``, ``theta = alpha / (kappa dt) = -alpha / beta`` and
    ``sigma = std(resid) / sqrt(dt)``.

CIR (Pearson-Sun)
    Dividing the discretised SDE by ``sqrt(r_t)`` stabilises the variance:
    ``Delta r / sqrt(r_t) = b1 (1 / sqrt(r_t)) + b2 sqrt(r_t) + error`` with
    ``b1 = kappa theta dt`` and ``b2 = -kappa dt``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from models import CIRParams, VasicekParams

_MIN_KAPPA = 1e-6
_MIN_SIGMA = 1e-8
_MIN_THETA = 1e-8


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Ordinary least squares via the normal equations / lstsq.

    Returns
    -------
    coef : np.ndarray
        Estimated coefficients.
    resid : np.ndarray
        Residuals ``y - X coef``.
    r_squared : float
        Coefficient of determination.
    se : np.ndarray
        Standard errors of the coefficients.
    """
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n, p = X.shape
    dof = max(n - p, 1)
    sigma2 = ss_res / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return coef, resid, r_squared, se


class OLSCalibrator:
    """Least squares calibrator for the Vasicek and CIR models."""

    # ----------------------------------------------------------- Vasicek
    def calibrate_vasicek(self, rate_series: np.ndarray, dt: float) -> Dict:
        """Calibrate the Vasicek model by regressing ``Delta r`` on ``r_t``.

        Parameters
        ----------
        rate_series : np.ndarray
            Observed short rate series.
        dt : float
            Time step between observations (years).

        Returns
        -------
        dict
            Keys: ``params`` (:class:`VasicekParams`), ``r_squared``,
            ``residuals``, ``std_errors`` (dict), ``n_obs``.
        """
        r = np.asarray(rate_series, dtype=float).ravel()
        if r.size < 3:
            raise ValueError("Need at least 3 observations to calibrate.")

        r_prev = r[:-1]
        dr = r[1:] - r_prev
        X = np.column_stack([np.ones_like(r_prev), r_prev])
        coef, resid, r_squared, se = _ols(X, dr)
        alpha, beta = float(coef[0]), float(coef[1])

        # beta = -kappa dt < 0 for a mean-reverting series.
        kappa = -beta / dt
        kappa = max(kappa, _MIN_KAPPA)
        # theta = alpha / (kappa dt) = -alpha / beta.
        theta = alpha / (kappa * dt)
        sigma = float(np.std(resid, ddof=2) / np.sqrt(dt))
        sigma = max(sigma, _MIN_SIGMA)

        # Delta method standard errors for kappa, theta.
        se_kappa = float(se[1] / dt)
        # theta = -alpha/beta; ignore covariance for a simple approximation.
        se_theta = float(
            np.sqrt((se[0] / beta) ** 2 + (alpha * se[1] / beta**2) ** 2)
            if beta != 0
            else np.nan
        )

        return {
            "params": VasicekParams(kappa=kappa, theta=theta, sigma=sigma),
            "r_squared": float(r_squared),
            "residuals": resid,
            "std_errors": {
                "kappa": se_kappa,
                "theta": se_theta,
                "sigma": float("nan"),
            },
            "n_obs": int(r_prev.size),
        }

    # --------------------------------------------------------------- CIR
    def calibrate_cir(self, rate_series: np.ndarray, dt: float) -> Dict:
        """Calibrate the CIR model by the Pearson-Sun weighted regression.

        Parameters
        ----------
        rate_series : np.ndarray
            Observed short rate series (should be strictly positive).
        dt : float
            Time step between observations (years).

        Returns
        -------
        dict
            Keys: ``params`` (:class:`CIRParams`), ``r_squared``,
            ``residuals``, ``std_errors`` (dict), ``n_obs``.
        """
        r = np.asarray(rate_series, dtype=float).ravel()
        r = np.where(r <= 0.0, _MIN_THETA, r)
        if r.size < 3:
            raise ValueError("Need at least 3 observations to calibrate.")

        r_prev = r[:-1]
        dr = r[1:] - r_prev
        sqrt_r = np.sqrt(np.maximum(r_prev, 1e-12))

        # Delta r / sqrt(r) = b1 (1/sqrt(r)) + b2 sqrt(r) + error.
        y = dr / sqrt_r
        X = np.column_stack([1.0 / sqrt_r, sqrt_r])
        coef, resid, r_squared, se = _ols(X, y)
        b1, b2 = float(coef[0]), float(coef[1])

        # b1 = kappa theta dt, b2 = -kappa dt.
        kappa = -b2 / dt
        kappa = max(kappa, _MIN_KAPPA)
        theta = (b1 / dt) / kappa
        if theta <= 0:
            theta = float(np.mean(r))
        theta = max(theta, _MIN_THETA)

        # The weighted residual has homoscedastic variance sigma^2 dt.
        sigma = float(np.std(resid, ddof=2) / np.sqrt(dt))
        sigma = max(sigma, _MIN_SIGMA)

        se_kappa = float(se[1] / dt)

        return {
            "params": CIRParams(kappa=kappa, theta=theta, sigma=sigma),
            "r_squared": float(r_squared),
            "residuals": resid,
            "std_errors": {
                "kappa": se_kappa,
                "theta": float("nan"),
                "sigma": float("nan"),
            },
            "n_obs": int(r_prev.size),
        }
