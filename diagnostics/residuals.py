"""Residual diagnostics for fitted short rate models.

Given an observed short rate series and a fitted model, the standardised
one-step-ahead residuals should be approximately independent standard normal
variates when the model is correctly specified. This module computes those
residuals and the usual battery of misspecification tests: normality
(Jarque-Bera, Shapiro-Wilk, skew, excess kurtosis), serial correlation
(Ljung-Box, autocorrelation function) and QQ-plot coordinates.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import stats

from models import CIRModel, ShortRateModel, VasicekModel

#: Numerical guard for variances under a square root.
_EPS: float = 1e-12


class ResidualDiagnostics:
    """One-step residual diagnostics for short rate models."""

    # -------------------------------------------------- residual builder
    def compute_residuals(
        self, rate_series: np.ndarray, model: ShortRateModel, dt: float
    ) -> np.ndarray:
        """Standardised one-step-ahead residuals of a series under a model.

        Parameters
        ----------
        rate_series : np.ndarray
            One-dimensional short rate series sampled at interval ``dt``.
        model : ShortRateModel
            Fitted model providing ``conditional_mean`` and
            ``conditional_variance`` (Vasicek and CIR). Other models fall back on
            an Euler one-step approximation using ``drift``/``diffusion``.
        dt : float
            Sampling interval in years.

        Returns
        -------
        np.ndarray
            Standardised residuals
            ``(r_{t+1} - E[r_{t+1} | r_t]) / sqrt(Var[r_{t+1} | r_t])`` of length
            ``len(rate_series) - 1``. Under a correctly specified model these are
            approximately i.i.d. standard normal.
        """
        series = np.asarray(rate_series, dtype=float).ravel()
        r_t = series[:-1]
        r_next = series[1:]

        if isinstance(model, VasicekModel):
            k, th = model.params.kappa, model.params.theta
            e = np.exp(-k * dt)
            cond_mean = r_t * e + th * (1.0 - e)
            cond_var = np.full_like(r_t, model.conditional_variance(dt))
        elif isinstance(model, CIRModel):
            k, th, s = model.params.kappa, model.params.theta, model.params.sigma
            e = np.exp(-k * dt)
            cond_mean = r_t * e + th * (1.0 - e)
            cond_var = (
                r_t * (s**2 / k) * (e - np.exp(-2.0 * k * dt))
                + th * (s**2 / (2.0 * k)) * (1.0 - e) ** 2
            )
        else:
            # Generic Euler one-step moments.
            drift = np.asarray(model.drift(r_t, 0.0), dtype=float)
            diff = np.asarray(model.diffusion(r_t, 0.0), dtype=float)
            cond_mean = r_t + drift * dt
            cond_var = (diff**2) * dt

        std = np.sqrt(np.maximum(cond_var, _EPS))
        return (r_next - cond_mean) / std

    # ----------------------------------------------------- normality
    def normality_tests(self, residuals: np.ndarray) -> dict[str, float]:
        """Normality tests and shape statistics of the residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Standardised residuals.

        Returns
        -------
        dict
            ``jb_stat``, ``jb_pvalue`` (Jarque-Bera), ``shapiro_stat``,
            ``shapiro_pvalue`` (Shapiro-Wilk), ``skew`` and ``kurtosis`` (excess
            kurtosis, normal == 0).
        """
        res = np.asarray(residuals, dtype=float).ravel()
        res = res[np.isfinite(res)]

        jb_stat, jb_pvalue = np.nan, np.nan
        if res.size >= 2:
            jb = stats.jarque_bera(res)
            jb_stat, jb_pvalue = float(jb.statistic), float(jb.pvalue)

        shapiro_stat, shapiro_pvalue = np.nan, np.nan
        # Shapiro-Wilk requires 3 <= n <= 5000.
        if 3 <= res.size <= 5000:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sw = stats.shapiro(res)
            shapiro_stat, shapiro_pvalue = float(sw.statistic), float(sw.pvalue)

        return {
            "jb_stat": jb_stat,
            "jb_pvalue": jb_pvalue,
            "shapiro_stat": shapiro_stat,
            "shapiro_pvalue": shapiro_pvalue,
            "skew": float(stats.skew(res)) if res.size else float("nan"),
            "kurtosis": float(stats.kurtosis(res)) if res.size else float("nan"),
        }

    # ----------------------------------------------------- serial corr
    def ljung_box(self, residuals: np.ndarray, lags: int = 10) -> dict[str, float]:
        """Ljung-Box test for residual autocorrelation up to ``lags``.

        Parameters
        ----------
        residuals : np.ndarray
            Standardised residuals.
        lags : int, optional
            Maximum lag tested.

        Returns
        -------
        dict
            ``lb_stat`` and ``lb_pvalue`` at the maximum lag. A small p-value
            indicates remaining serial correlation (model misspecification).
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        res = np.asarray(residuals, dtype=float).ravel()
        res = res[np.isfinite(res)]
        max_lag = max(1, min(lags, res.size - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = acorr_ljungbox(res, lags=[max_lag], return_df=True)
        return {
            "lb_stat": float(table["lb_stat"].iloc[-1]),
            "lb_pvalue": float(table["lb_pvalue"].iloc[-1]),
        }

    # ----------------------------------------------------- acf / qq
    def acf(self, residuals: np.ndarray, nlags: int = 20) -> np.ndarray:
        """Sample autocorrelation function of the residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Standardised residuals.
        nlags : int, optional
            Number of lags returned (excluding lag 0).

        Returns
        -------
        np.ndarray
            Autocorrelations for lags ``0 .. nlags`` (lag 0 == 1).
        """
        res = np.asarray(residuals, dtype=float).ravel()
        res = res[np.isfinite(res)]
        n = res.size
        if n < 2:
            return np.array([1.0])
        nlags = min(nlags, n - 1)
        res = res - res.mean()
        denom = np.dot(res, res)
        if denom <= _EPS:
            return np.concatenate([[1.0], np.zeros(nlags)])
        out = np.empty(nlags + 1)
        for k in range(nlags + 1):
            out[k] = np.dot(res[: n - k], res[k:]) / denom
        return out

    def qq_data(self, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Theoretical and sample quantiles for a normal QQ plot.

        Parameters
        ----------
        residuals : np.ndarray
            Standardised residuals.

        Returns
        -------
        tuple of np.ndarray
            ``(theoretical_quantiles, sample_quantiles)`` both sorted ascending.
            Plotting one against the other yields a straight line when the
            residuals are normal.
        """
        res = np.asarray(residuals, dtype=float).ravel()
        res = res[np.isfinite(res)]
        sample = np.sort(res)
        n = sample.size
        if n == 0:
            return np.array([]), np.array([])
        # Median-rank (Filliben-style) plotting positions.
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical = stats.norm.ppf(probs)
        return theoretical, sample
