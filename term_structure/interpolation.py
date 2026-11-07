"""Curve interpolation helpers.

Wraps SciPy interpolators with conventions that are appropriate for interest
rate curves: cubic splines for smooth yield interpolation, monotone PCHIP for
shape-preserving interpolation, and log-linear interpolation of discount
factors (the market-standard scheme that keeps implied forwards positive on a
piecewise-flat instantaneous forward assumption).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Floor applied to discount factors before taking logarithms.
MIN_DF: float = 1e-300


class CurveInterpolator:
    """Construct interpolating callables for term-structure curves."""

    # ------------------------------------------------------------- builders
    @staticmethod
    def cubic(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Natural cubic-spline interpolant.

        Parameters
        ----------
        x : np.ndarray
            Strictly increasing abscissae (e.g. maturities).
        y : np.ndarray
            Ordinates to interpolate (e.g. yields).

        Returns
        -------
        callable
            Function mapping query points to interpolated values.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        spline = CubicSpline(x, y, extrapolate=True)
        return lambda xq: spline(np.asarray(xq, dtype=float))

    @staticmethod
    def monotone(
        x: np.ndarray, y: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Shape-preserving monotone (PCHIP) interpolant.

        Parameters
        ----------
        x : np.ndarray
            Strictly increasing abscissae.
        y : np.ndarray
            Ordinates to interpolate.

        Returns
        -------
        callable
            PCHIP interpolating function.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pchip = PchipInterpolator(x, y, extrapolate=True)
        return lambda xq: pchip(np.asarray(xq, dtype=float))

    @staticmethod
    def log_linear_df(
        maturities: np.ndarray, dfs: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Log-linear discount-factor interpolant.

        Linear interpolation is performed on ``ln(DF)`` against maturity, which
        is equivalent to assuming piecewise-constant continuously compounded
        forward rates between nodes.

        Parameters
        ----------
        maturities : np.ndarray
            Strictly increasing maturities in years.
        dfs : np.ndarray
            Discount factors at ``maturities``.

        Returns
        -------
        callable
            Function returning interpolated discount factors.
        """
        maturities = np.asarray(maturities, dtype=float)
        dfs = np.maximum(np.asarray(dfs, dtype=float), MIN_DF)
        log_dfs = np.log(dfs)

        def interp(xq: np.ndarray) -> np.ndarray:
            xq = np.asarray(xq, dtype=float)
            return np.exp(np.interp(xq, maturities, log_dfs))

        return interp

    # ------------------------------------------------------------ dispatch
    @staticmethod
    def interpolate(
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        method: str = "cubic",
    ) -> np.ndarray:
        """Interpolate ``(x, y)`` at ``x_new`` using the requested method.

        Parameters
        ----------
        x : np.ndarray
            Abscissae.
        y : np.ndarray
            Ordinates.
        x_new : np.ndarray
            Query points.
        method : {'cubic', 'monotone', 'log_linear_df'}, optional
            Interpolation scheme. Defaults to ``'cubic'``.

        Returns
        -------
        np.ndarray
            Interpolated values at ``x_new``.

        Raises
        ------
        ValueError
            If ``method`` is unknown.
        """
        if method == "cubic":
            f = CurveInterpolator.cubic(x, y)
        elif method == "monotone":
            f = CurveInterpolator.monotone(x, y)
        elif method == "log_linear_df":
            f = CurveInterpolator.log_linear_df(x, y)
        else:
            raise ValueError(f"Unknown interpolation method: {method!r}")
        return np.asarray(f(x_new), dtype=float)
