"""Coupon bond pricing and fixed income risk analytics.

This module exposes :class:`CouponBondPricer`, which prices fixed-rate coupon
bonds off a short rate model's discount curve and computes the standard first
and second order risk measures (Macaulay duration, modified duration, DV01 and
convexity). Yield-to-maturity helpers based on Newton-Raphson are also
provided for converting between price and yield without a model.
"""

from __future__ import annotations

import numpy as np

from models import BondPrice, ShortRateModel

#: Default coupon frequency (semi-annual).
DEFAULT_FREQ: int = 2
#: Default face value.
DEFAULT_FACE: float = 100.0
#: One basis point, used for DV01.
ONE_BP: float = 1.0e-4
#: Newton-Raphson tolerance for yield solving.
YTM_TOL: float = 1.0e-12
#: Maximum Newton-Raphson iterations.
YTM_MAX_ITER: int = 100
#: Numerical guard for division.
EPS: float = 1.0e-14


def _cashflow_schedule(
    coupon_rate: float, maturity: float, freq: int, face: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, cashflows)`` for a fixed-rate coupon bond.

    The coupon per period is ``coupon_rate * face / freq`` paid at each period
    end; the final cashflow additionally redeems ``face``.
    """
    n = int(round(maturity * freq))
    n = max(n, 1)
    coupon = coupon_rate * face / freq
    times = np.array([(i + 1) / freq for i in range(n)], dtype=float)
    cfs = np.full(n, coupon, dtype=float)
    cfs[-1] += face
    return times, cfs


class CouponBondPricer:
    """Price coupon bonds and compute their risk metrics off a model curve."""

    def price(
        self,
        model: ShortRateModel,
        r0: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> BondPrice:
        """Price a coupon bond and populate a :class:`BondPrice` container.

        Returns
        -------
        BondPrice
            Holds ``price``, ``yield_`` (yield to maturity), Macaulay
            ``duration``, ``modified_duration``, ``convexity`` and ``dv01``.
        """
        times, cfs = _cashflow_schedule(coupon_rate, maturity, freq, face)
        dfs = np.array([model.zcb_price(r0, float(t)) for t in times])
        pvs = cfs * dfs
        price = float(pvs.sum())

        ytm = self.yield_to_maturity(price, coupon_rate, maturity, freq, face)

        weights = pvs / max(price, EPS)
        mac_dur = float(np.sum(weights * times))
        mod_dur = mac_dur / (1.0 + ytm / freq)
        # PV-weighted convexity: sum w_i t_i (t_i + 1/freq) / (1 + y/freq)^2.
        conv = float(
            np.sum(weights * times * (times + 1.0 / freq)) / (1.0 + ytm / freq) ** 2
        )
        dv01 = mod_dur * price * ONE_BP

        return BondPrice(
            price=price,
            yield_=ytm,
            duration=mac_dur,
            convexity=conv,
            model_name=model.name,
            modified_duration=mod_dur,
            dv01=dv01,
        )

    def price_from_yield(
        self,
        ytm: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> float:
        """Price a coupon bond from a flat yield to maturity (no model).

        Discounting uses the periodic compounding convention
        ``(1 + y/freq)^{-freq * t}``.
        """
        times, cfs = _cashflow_schedule(coupon_rate, maturity, freq, face)
        disc = (1.0 + ytm / freq) ** (-freq * times)
        return float(np.sum(cfs * disc))

    def yield_to_maturity(
        self,
        price: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> float:
        """Solve for the yield to maturity via Newton-Raphson.

        Returns
        -------
        float
            The periodic-compounded yield that reprices the bond to ``price``.
        """
        times, cfs = _cashflow_schedule(coupon_rate, maturity, freq, face)

        def pv_and_deriv(y: float) -> tuple[float, float]:
            base = 1.0 + y / freq
            disc = base ** (-freq * times)
            pv = np.sum(cfs * disc)
            # d/dy of (1+y/freq)^(-freq t) = -t * (1+y/freq)^(-freq t - 1).
            dpv = np.sum(cfs * (-times) * base ** (-freq * times - 1.0))
            return float(pv), float(dpv)

        # Initial guess: current yield, floored away from singularities.
        y = max(coupon_rate, 1.0e-4)
        for _ in range(YTM_MAX_ITER):
            pv, dpv = pv_and_deriv(y)
            diff = pv - price
            if abs(diff) < YTM_TOL:
                break
            if abs(dpv) < EPS:
                break
            y -= diff / dpv
            if y <= -freq + 1.0e-6:  # keep (1 + y/freq) > 0
                y = -freq + 1.0e-6
        return float(y)

    def ytm_from_price(
        self,
        price: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> float:
        """Alias of :meth:`yield_to_maturity`."""
        return self.yield_to_maturity(price, coupon_rate, maturity, freq, face)

    def duration(
        self,
        model: ShortRateModel,
        r0: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> float:
        """Macaulay duration using model discount factors for PV weights."""
        return self.price(model, r0, coupon_rate, maturity, freq, face).duration

    def modified_duration(
        self,
        model: ShortRateModel,
        r0: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> float:
        """Modified duration ``D_mac / (1 + y/freq)``."""
        bp = self.price(model, r0, coupon_rate, maturity, freq, face)
        return float(bp.modified_duration)

    def convexity(
        self,
        model: ShortRateModel,
        r0: float,
        coupon_rate: float,
        maturity: float,
        freq: int = DEFAULT_FREQ,
        face: float = DEFAULT_FACE,
    ) -> float:
        """PV-weighted convexity of the coupon bond."""
        return self.price(model, r0, coupon_rate, maturity, freq, face).convexity
