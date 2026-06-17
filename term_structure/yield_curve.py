"""Yield curve construction utilities.

This module provides :class:`YieldCurveBuilder`, a thin façade that turns either
a calibrated short rate model or a set of observed market instruments into a
tidy :class:`pandas.DataFrame` describing the zero (spot) curve together with
discount factors, instantaneous forwards and par swap rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Lower bound on maturities to avoid division by zero when annualising.
MIN_TIME: float = 1e-12

#: Identifier reported by the two-factor Gaussian model (G2++).
G2_MODEL_NAME: str = "G2++"


class YieldCurveBuilder:
    """Construct zero, discount and par curves from models or market data.

    The class is intentionally stateless; every method is a pure function of
    its inputs so the builder can be shared freely across the library.
    """

    # ------------------------------------------------------------------ model
    @staticmethod
    def from_model(
        model,
        r0: float,
        maturities: np.ndarray,
    ) -> pd.DataFrame:
        """Build a curve table from a calibrated short rate model.

        Parameters
        ----------
        model : ShortRateModel
            Any model exposing ``zcb_price``, ``yield_curve`` and
            ``forward_curve``. The two-factor G2++ model is handled specially
            because its ``yield_curve`` signature takes the initial factor
            values ``(x0, y0)`` rather than a single short rate.
        r0 : float
            Initial short rate (ignored by G2++, which is fixed at ``x0=y0=0``).
        maturities : np.ndarray
            Strictly positive maturities in years.

        Returns
        -------
        pandas.DataFrame
            Columns ``maturity``, ``yield``, ``price`` and ``forward``.
        """
        maturities = np.asarray(maturities, dtype=float)

        is_g2 = getattr(model, "name", "") == G2_MODEL_NAME
        if is_g2:
            yields = model.yield_curve(0.0, 0.0, maturities)
        else:
            yields = model.yield_curve(r0, maturities)

        prices = np.array([model.zcb_price(r0, float(T)) for T in maturities])
        forwards = np.asarray(model.forward_curve(r0, maturities), dtype=float)

        return pd.DataFrame(
            {
                "maturity": maturities,
                "yield": np.asarray(yields, dtype=float),
                "price": prices,
                "forward": forwards,
            }
        )

    # ----------------------------------------------------------------- market
    @staticmethod
    def from_market(
        maturities: np.ndarray,
        prices: np.ndarray,
        face: float = 1.0,
    ) -> pd.DataFrame:
        """Build a zero curve from observed zero coupon bond prices.

        Parameters
        ----------
        maturities : np.ndarray
            Bond maturities in years.
        prices : np.ndarray
            Observed clean ZCB prices (per ``face`` units of notional).
        face : float, optional
            Face value associated with ``prices``. Defaults to ``1.0``.

        Returns
        -------
        pandas.DataFrame
            Columns ``maturity``, ``yield``, ``price``, ``discount_factor``.

        Notes
        -----
        The continuously compounded zero rate is ``R = -ln(P / face) / T``.
        """
        maturities = np.asarray(maturities, dtype=float)
        prices = np.asarray(prices, dtype=float)

        yields = YieldCurveBuilder.spot_rates(maturities, prices, face=face)
        dfs = prices / face

        return pd.DataFrame(
            {
                "maturity": maturities,
                "yield": yields,
                "price": prices,
                "discount_factor": dfs,
            }
        )

    # ------------------------------------------------------------ conversions
    @staticmethod
    def spot_rates(
        maturities: np.ndarray,
        prices: np.ndarray,
        face: float = 1.0,
    ) -> np.ndarray:
        """Continuously compounded zero rates from ZCB prices.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.
        prices : np.ndarray
            ZCB prices per ``face`` units of notional.
        face : float, optional
            Face value. Defaults to ``1.0``.

        Returns
        -------
        np.ndarray
            Spot rates ``R(0, T) = -ln(P / face) / T``.
        """
        maturities = np.maximum(np.asarray(maturities, dtype=float), MIN_TIME)
        prices = np.asarray(prices, dtype=float)
        return -np.log(prices / face) / maturities

    @staticmethod
    def discount_factors(
        maturities: np.ndarray,
        yields: np.ndarray,
    ) -> np.ndarray:
        """Discount factors ``exp(-y * T)`` from a continuously compounded curve.

        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years.
        yields : np.ndarray
            Continuously compounded zero rates.

        Returns
        -------
        np.ndarray
            Discount factors ``P(0, T)``.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)
        return np.exp(-yields * maturities)

    @staticmethod
    def par_rates(
        maturities: np.ndarray,
        yields: np.ndarray,
    ) -> np.ndarray:
        """Par swap rates implied by a continuously compounded zero curve.

        Parameters
        ----------
        maturities : np.ndarray
            Payment-grid maturities in years (used directly as the swap
            payment schedule).
        yields : np.ndarray
            Continuously compounded zero rates at ``maturities``.

        Returns
        -------
        np.ndarray
            Par rate at each node,
            ``c_n = (1 - P(T_n)) / sum_{i<=n} delta_i P(T_i)``.

        Notes
        -----
        Accrual factors ``delta_i`` are taken as the consecutive differences of
        ``maturities`` (the first accrual uses ``T_0`` itself), so the grid need
        not be annual.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)
        dfs = YieldCurveBuilder.discount_factors(maturities, yields)

        deltas = np.diff(maturities, prepend=0.0)
        annuity = np.cumsum(deltas * dfs)
        annuity = np.where(annuity <= MIN_TIME, MIN_TIME, annuity)
        return (1.0 - dfs) / annuity
