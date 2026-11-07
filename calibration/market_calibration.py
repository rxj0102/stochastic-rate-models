"""Market calibration of the Hull-White model to traded instruments.

This module provides :class:`MarketCalibrator`, which fits the Hull-White
one-factor model's free parameters ``(kappa, sigma)`` to market quotes. The
``theta(t)`` function is not free: it is implied by the initial yield curve via
:meth:`HullWhiteModel.set_initial_curve`, so only mean reversion and volatility
are optimised.

Two objectives are offered:

``calibrate_hw_to_caps``
    Least-squares fit of model cap prices to market cap prices.
``calibrate_hw_full``
    A weighted objective that also penalises zero-coupon bond mispricing,
    useful when curve and volatility instruments are calibrated jointly.

Notes
-----
Market quotes are passed as *prices* (not implied volatilities) to keep the
module self-contained. If you hold Black volatilities, convert them to cap
prices before calling.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import optimize

from models import HullWhiteModel, HullWhiteParams

_MIN_KAPPA = 1e-6
_MIN_SIGMA = 1e-8


def _build_hw(
    kappa: float,
    sigma: float,
    curve_mats: np.ndarray,
    curve_yields: np.ndarray,
) -> HullWhiteModel:
    """Construct a Hull-White model fitted to the given initial curve."""
    model = HullWhiteModel(
        HullWhiteParams(kappa=max(kappa, _MIN_KAPPA), sigma=max(sigma, _MIN_SIGMA))
    )
    model.set_initial_curve(curve_mats, curve_yields)
    return model


class MarketCalibrator:
    """Calibrate the Hull-White model to market cap (and bond) quotes."""

    # ------------------------------------------------------- caps only
    def calibrate_hw_to_caps(
        self,
        market_maturities: Sequence[float],
        market_cap_prices: Sequence[float],
        initial_curve_mats: Sequence[float],
        initial_curve_yields: Sequence[float],
        r0: Optional[float] = None,
        cap_rate: Optional[float] = None,
        notional: float = 1.0,
        freq: int = 4,
    ) -> Dict:
        """Fit ``(kappa, sigma)`` to a set of market cap prices.

        Parameters
        ----------
        market_maturities : sequence of float
            Cap maturities (years) corresponding to each market price.
        market_cap_prices : sequence of float
            Observed market cap prices.
        initial_curve_mats, initial_curve_yields : sequence of float
            The initial zero-coupon yield curve used to imply ``theta(t)``.
        r0 : float, optional
            Unused for pricing (the curve implies ``r0``); accepted for API
            symmetry and reported back in the result.
        cap_rate : float, optional
            Strike of each cap. If ``None``, an at-the-money strike equal to the
            curve yield interpolated at the cap maturity is used.
        notional : float, optional
            Cap notional.
        freq : int, optional
            Caplet reset frequency per year.

        Returns
        -------
        dict
            Keys: ``params`` (:class:`HullWhiteParams`), ``rmse``,
            ``model_prices``, ``market_prices``, ``success``, ``r0``.
        """
        mats = np.asarray(market_maturities, dtype=float)
        market = np.asarray(market_cap_prices, dtype=float)
        curve_mats = np.asarray(initial_curve_mats, dtype=float)
        curve_yields = np.asarray(initial_curve_yields, dtype=float)

        def strikes(model: HullWhiteModel) -> np.ndarray:
            if cap_rate is not None:
                return np.full(mats.shape, float(cap_rate))
            # ATM strike: interpolate the curve yield at each maturity.
            return np.interp(mats, curve_mats, curve_yields)

        def model_prices(kappa: float, sigma: float) -> np.ndarray:
            model = _build_hw(kappa, sigma, curve_mats, curve_yields)
            ks = strikes(model)
            return np.array([
                model.cap_price(float(ks[i]), float(mats[i]), notional, freq)
                for i in range(mats.size)
            ])

        def objective(x: np.ndarray) -> float:
            try:
                mp = model_prices(x[0], x[1])
            except Exception:
                return 1e12
            diff = mp - market
            if not np.all(np.isfinite(diff)):
                return 1e12
            return float(np.sum(diff**2))

        x0 = np.array([0.1, 0.01])
        bounds = [(1e-3, 5.0), (1e-4, 0.5)]
        res = optimize.minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds
        )

        kappa, sigma = float(res.x[0]), float(res.x[1])
        mp = model_prices(kappa, sigma)
        rmse = float(np.sqrt(np.mean((mp - market) ** 2)))

        fitted = _build_hw(kappa, sigma, curve_mats, curve_yields)
        return {
            "params": HullWhiteParams(
                kappa=max(kappa, _MIN_KAPPA), sigma=max(sigma, _MIN_SIGMA)
            ),
            "rmse": rmse,
            "model_prices": mp,
            "market_prices": market,
            "success": bool(res.success),
            "r0": float(fitted.r0) if r0 is None else float(r0),
        }

    # ------------------------------------------------------- full fit
    def calibrate_hw_full(
        self,
        zcb_prices: Sequence[float],
        swaption_or_cap_targets: Sequence[float],
        maturities: Sequence[float],
        initial_curve: tuple[Sequence[float], Sequence[float]],
        r0: Optional[float] = None,
        w1: float = 1.0,
        w2: float = 1.0,
        cap_rate: Optional[float] = None,
        notional: float = 1.0,
        freq: int = 4,
    ) -> Dict:
        """Jointly fit ``(kappa, sigma)`` to ZCB prices and cap targets.

        The objective is a weighted sum of two RMSEs::

            w1 * RMSE(zcb) + w2 * RMSE(cap)

        Parameters
        ----------
        zcb_prices : sequence of float
            Target zero-coupon bond prices at each maturity.
        swaption_or_cap_targets : sequence of float
            Target cap (or other derivative) prices at each maturity.
        maturities : sequence of float
            Maturities (years) shared by both target sets.
        initial_curve : tuple of (mats, yields)
            The initial yield curve implying ``theta(t)``.
        r0 : float, optional
            Reported back; the curve implies the pricing ``r0``.
        w1, w2 : float, optional
            Weights on the bond-price and derivative-price RMSE terms.
        cap_rate : float, optional
            Cap strike; if ``None`` an ATM strike from the curve is used.
        notional : float, optional
            Cap notional.
        freq : int, optional
            Caplet reset frequency per year.

        Returns
        -------
        dict
            Keys: ``params`` (:class:`HullWhiteParams`), ``fit_quality`` (dict
            with ``zcb_rmse``, ``cap_rmse``, ``objective``), ``model_zcb``,
            ``model_caps``, ``success``, ``r0``.
        """
        mats = np.asarray(maturities, dtype=float)
        zcb_target = np.asarray(zcb_prices, dtype=float)
        cap_target = np.asarray(swaption_or_cap_targets, dtype=float)
        curve_mats = np.asarray(initial_curve[0], dtype=float)
        curve_yields = np.asarray(initial_curve[1], dtype=float)

        def strikes() -> np.ndarray:
            if cap_rate is not None:
                return np.full(mats.shape, float(cap_rate))
            return np.interp(mats, curve_mats, curve_yields)

        def model_quotes(kappa: float, sigma: float):
            model = _build_hw(kappa, sigma, curve_mats, curve_yields)
            r0_loc = model.r0
            zcb = np.array([model.zcb_price(r0_loc, float(T)) for T in mats])
            ks = strikes()
            caps = np.array([
                model.cap_price(float(ks[i]), float(mats[i]), notional, freq)
                for i in range(mats.size)
            ])
            return zcb, caps

        def rmse(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.sqrt(np.mean((a - b) ** 2)))

        def objective(x: np.ndarray) -> float:
            try:
                zcb, caps = model_quotes(x[0], x[1])
            except Exception:
                return 1e12
            if not (np.all(np.isfinite(zcb)) and np.all(np.isfinite(caps))):
                return 1e12
            return w1 * rmse(zcb, zcb_target) + w2 * rmse(caps, cap_target)

        x0 = np.array([0.1, 0.01])
        bounds = [(1e-3, 5.0), (1e-4, 0.5)]
        res = optimize.minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds
        )

        kappa, sigma = float(res.x[0]), float(res.x[1])
        zcb, caps = model_quotes(kappa, sigma)
        zcb_rmse = rmse(zcb, zcb_target)
        cap_rmse = rmse(caps, cap_target)

        fitted = _build_hw(kappa, sigma, curve_mats, curve_yields)
        return {
            "params": HullWhiteParams(
                kappa=max(kappa, _MIN_KAPPA), sigma=max(sigma, _MIN_SIGMA)
            ),
            "fit_quality": {
                "zcb_rmse": zcb_rmse,
                "cap_rmse": cap_rmse,
                "objective": float(res.fun),
            },
            "model_zcb": zcb,
            "model_caps": caps,
            "success": bool(res.success),
            "r0": float(fitted.r0) if r0 is None else float(r0),
        }
