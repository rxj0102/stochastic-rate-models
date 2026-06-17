"""Stationarity and mean-reversion diagnostics for short rate series.

This module bundles the empirical tests practitioners use to validate that a
fitted short rate model is consistent with the observed dynamics of a rate
series: unit-root / stationarity tests (ADF, KPSS), an AR(1) half-life
estimate, and a comparison of the theoretical terminal variance implied by a
model against the empirical Monte Carlo variance.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from models import CIRModel, ShortRateModel, VasicekModel

#: Numerical guard for variance / time quantities.
_EPS: float = 1e-12


class RateStationarityTests:
    """Empirical stationarity and mean-reversion diagnostics.

    The class is stateless; every method takes the data (and, where relevant, a
    model) as an explicit argument so that it can be reused across series.
    """

    # ------------------------------------------------------------------ tests
    def test_mean_reversion(
        self, rate_series: np.ndarray
    ) -> dict[str, float]:
        """Run unit-root / stationarity tests and estimate the half-life.

        Parameters
        ----------
        rate_series : np.ndarray
            One-dimensional series of short rate observations.

        Returns
        -------
        dict
            Dictionary with keys ``adf_stat``, ``adf_pvalue``, ``kpss_stat``,
            ``kpss_pvalue`` and ``half_life``. A small ADF p-value and a large
            KPSS p-value both point to a stationary (mean-reverting) series.

        Notes
        -----
        The Augmented Dickey-Fuller test has the null hypothesis of a unit root
        (non-stationarity); the KPSS test has the complementary null of
        stationarity. The KPSS routine emits an ``InterpolationWarning`` when the
        statistic falls outside the tabulated range; this is suppressed.
        """
        series = np.asarray(rate_series, dtype=float).ravel()
        series = series[np.isfinite(series)]

        adf_stat, adf_pvalue = np.nan, np.nan
        kpss_stat, kpss_pvalue = np.nan, np.nan

        if series.size >= 8:
            try:
                adf_res = adfuller(series, autolag="AIC")
                adf_stat, adf_pvalue = float(adf_res[0]), float(adf_res[1])
            except (ValueError, np.linalg.LinAlgError):
                pass

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    kpss_res = kpss(series, regression="c", nlags="auto")
                    kpss_stat, kpss_pvalue = float(kpss_res[0]), float(kpss_res[1])
                except (ValueError, OverflowError):
                    pass

        return {
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "kpss_stat": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
            "half_life": self.half_life_estimate(series),
        }

    def half_life_estimate(self, rate_series: np.ndarray) -> float:
        """Estimate the mean-reversion half-life from an AR(1) fit.

        Parameters
        ----------
        rate_series : np.ndarray
            One-dimensional rate series.

        Returns
        -------
        float
            Estimated half-life in the units of the sampling interval (i.e. in
            "steps"). When the series is sampled at ``dt`` years per step the
            half-life in years is ``half_life * dt``. Returns ``inf`` when the
            series does not mean-revert (slope ``b >= 0``).

        Notes
        -----
        Regress the increment ``Delta r_t = r_t - r_{t-1}`` on the lagged level
        ``r_{t-1}`` as ``Delta r = a + b r_{t-1}``. Under the discretised OU
        recursion ``r_t = (1 + b) r_{t-1} + a + eps``, so the per-step decay
        factor is ``phi = 1 + b`` and the half-life is ``-ln 2 / ln(1 + b)``
        (equivalently ``-ln 2 / b`` to first order for small ``|b|``).
        """
        series = np.asarray(rate_series, dtype=float).ravel()
        series = series[np.isfinite(series)]
        if series.size < 3:
            return float("inf")

        r_lag = series[:-1]
        d_r = np.diff(series)
        # OLS slope/intercept of d_r on r_lag.
        x = np.column_stack([np.ones_like(r_lag), r_lag])
        try:
            coef, *_ = np.linalg.lstsq(x, d_r, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        b = float(coef[1])

        # b must be negative (mean reverting); 1 + b must lie in (0, 1).
        if b >= 0.0:
            return float("inf")
        phi = 1.0 + b
        if phi <= 0.0:
            # Oscillatory / overshoot; fall back on the linear approximation.
            return float(-np.log(2.0) / b)
        return float(-np.log(2.0) / np.log(phi))

    # ----------------------------------------------- variance comparison
    def theoretical_vs_empirical_variance(
        self,
        model: ShortRateModel,
        r0: float,
        T: float = 30.0,
        n_paths: int = 20_000,
        n_steps: int = 2_520,
        seed: int = 0,
    ) -> dict[str, float]:
        """Compare the model terminal variance to a Monte Carlo estimate.

        Parameters
        ----------
        model : ShortRateModel
            A model exposing a ``simulate`` method. Closed-form variances are
            available for Vasicek and CIR; other models fall back on the
            cross-sectional sample variance as the "theoretical" reference is
            unavailable.
        r0 : float
            Initial short rate.
        T : float, optional
            Simulation horizon in years.
        n_paths, n_steps : int, optional
            Monte Carlo resolution.
        seed : int, optional
            Random seed.

        Returns
        -------
        dict
            ``empirical`` (sample variance of ``r(T)``), ``theoretical``
            (closed-form ``Var[r(T) | r0]`` where known, else ``nan``) and
            ``ratio`` (empirical / theoretical).
        """
        sim = model.simulate(r0, T, n_paths, n_steps, seed)
        empirical = float(np.var(sim.terminal, ddof=1))

        theoretical = float("nan")
        if isinstance(model, VasicekModel):
            theoretical = float(model.conditional_variance(T))
        elif isinstance(model, CIRModel):
            theoretical = float(model.conditional_variance(r0, T))

        ratio = (
            empirical / theoretical
            if np.isfinite(theoretical) and abs(theoretical) > _EPS
            else float("nan")
        )
        return {
            "empirical": empirical,
            "theoretical": theoretical,
            "ratio": ratio,
        }
