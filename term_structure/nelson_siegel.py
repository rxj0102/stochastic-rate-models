"""Nelson-Siegel and Svensson parametric yield curve fitting.

This module implements the classic Nelson-Siegel (1987) three-factor curve and
its Svensson (1994) six-factor extension with a second curvature hump. Both are
fitted by minimising the root-mean-square pricing error against an observed
zero curve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Lower bound on the decay parameters ``tau`` to avoid division by zero.
MIN_TAU: float = 1e-6

#: Lower bound on maturities used when forming the ``T / tau`` ratio.
MIN_TIME: float = 1e-12

#: Candidate decay values seeded into the multi-start optimiser.
_TAU_SEEDS = (0.5, 1.0, 2.0, 5.0)


def _ns_loadings(maturities: np.ndarray, tau: float) -> tuple:
    """Return the slope and curvature loadings for decay ``tau``."""
    tau = max(float(tau), MIN_TAU)
    T = np.maximum(np.asarray(maturities, dtype=float), MIN_TIME)
    x = T / tau
    decay = (1.0 - np.exp(-x)) / x
    slope = decay
    curvature = decay - np.exp(-x)
    return slope, curvature


class NelsonSiegelFitter:
    """Fit and evaluate Nelson-Siegel / Svensson yield curves.

    Attributes
    ----------
    params_ : dict or None
        Fitted Nelson-Siegel parameters after :meth:`fit`.
    svensson_params_ : dict or None
        Fitted Svensson parameters after :meth:`fit_svensson`.
    """

    def __init__(self) -> None:
        self.params_: Optional[dict] = None
        self.svensson_params_: Optional[dict] = None

    # ----------------------------------------------------- nelson-siegel
    @staticmethod
    def _evaluate(maturities: np.ndarray, beta0, beta1, beta2, tau) -> np.ndarray:
        """Nelson-Siegel yield ``y(T)`` for the given parameters."""
        slope, curvature = _ns_loadings(maturities, tau)
        return beta0 + beta1 * slope + beta2 * curvature

    def fit(self, maturities: np.ndarray, yields: np.ndarray) -> dict:
        """Fit the three-factor Nelson-Siegel model.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.
        yields : np.ndarray
            Observed continuously compounded zero rates.

        Returns
        -------
        dict
            Fitted parameters ``{beta0, beta1, beta2, tau}``.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)

        def rmse(p: np.ndarray) -> float:
            beta0, beta1, beta2, tau = p
            pred = self._evaluate(maturities, beta0, beta1, beta2, tau)
            return float(np.sqrt(np.mean((pred - yields) ** 2)))

        long_end = float(yields[-1])
        short_end = float(yields[0])
        bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (MIN_TAU, 30.0)]

        best = None
        for tau0 in _TAU_SEEDS:
            x0 = np.array([long_end, short_end - long_end, 0.0, tau0])
            res = minimize(rmse, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res

        beta0, beta1, beta2, tau = best.x
        self.params_ = {
            "beta0": float(beta0),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "tau": float(max(tau, MIN_TAU)),
        }
        return self.params_

    def predict(self, maturities: np.ndarray) -> np.ndarray:
        """Evaluate the fitted Nelson-Siegel curve.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.

        Returns
        -------
        np.ndarray
            Fitted yields.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.params_ is None:
            raise RuntimeError("Call fit() before predict().")
        p = self.params_
        return self._evaluate(
            maturities, p["beta0"], p["beta1"], p["beta2"], p["tau"]
        )

    @staticmethod
    def factor_loadings(maturities: np.ndarray, tau: float) -> np.ndarray:
        """Nelson-Siegel factor loadings.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.
        tau : float
            Decay parameter.

        Returns
        -------
        np.ndarray
            Array of shape ``(3, N)`` holding the level (ones), slope and
            curvature loadings respectively.
        """
        maturities = np.asarray(maturities, dtype=float)
        level = np.ones_like(maturities)
        slope, curvature = _ns_loadings(maturities, tau)
        return np.vstack([level, slope, curvature])

    # ---------------------------------------------------------- svensson
    @staticmethod
    def _evaluate_svensson(
        maturities: np.ndarray, beta0, beta1, beta2, beta3, tau1, tau2
    ) -> np.ndarray:
        """Svensson yield ``y(T)`` for the given parameters."""
        slope, curv1 = _ns_loadings(maturities, tau1)
        _, curv2 = _ns_loadings(maturities, tau2)
        return beta0 + beta1 * slope + beta2 * curv1 + beta3 * curv2

    def fit_svensson(self, maturities: np.ndarray, yields: np.ndarray) -> dict:
        """Fit the six-factor Svensson model.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.
        yields : np.ndarray
            Observed continuously compounded zero rates.

        Returns
        -------
        dict
            Fitted parameters ``{beta0, beta1, beta2, beta3, tau1, tau2}``.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)

        def rmse(p: np.ndarray) -> float:
            beta0, beta1, beta2, beta3, tau1, tau2 = p
            pred = self._evaluate_svensson(
                maturities, beta0, beta1, beta2, beta3, tau1, tau2
            )
            return float(np.sqrt(np.mean((pred - yields) ** 2)))

        long_end = float(yields[-1])
        short_end = float(yields[0])
        bounds = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (MIN_TAU, 30.0),
            (MIN_TAU, 30.0),
        ]

        best = None
        for tau1_0 in _TAU_SEEDS:
            for tau2_0 in _TAU_SEEDS:
                if tau2_0 <= tau1_0:
                    continue
                x0 = np.array(
                    [long_end, short_end - long_end, 0.0, 0.0, tau1_0, tau2_0]
                )
                res = minimize(rmse, x0, method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res

        beta0, beta1, beta2, beta3, tau1, tau2 = best.x
        self.svensson_params_ = {
            "beta0": float(beta0),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "beta3": float(beta3),
            "tau1": float(max(tau1, MIN_TAU)),
            "tau2": float(max(tau2, MIN_TAU)),
        }
        return self.svensson_params_

    def predict_svensson(self, maturities: np.ndarray) -> np.ndarray:
        """Evaluate the fitted Svensson curve.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.

        Returns
        -------
        np.ndarray
            Fitted yields.

        Raises
        ------
        RuntimeError
            If :meth:`fit_svensson` has not been called.
        """
        if self.svensson_params_ is None:
            raise RuntimeError("Call fit_svensson() before predict_svensson().")
        p = self.svensson_params_
        return self._evaluate_svensson(
            maturities,
            p["beta0"],
            p["beta1"],
            p["beta2"],
            p["beta3"],
            p["tau1"],
            p["tau2"],
        )
