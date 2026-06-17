"""Profit-and-loss attribution for a bond under a small rate move.

This module exposes :class:`PnLAttribution`, which decomposes the change in a
bond's value over a small rate move (and an optional holding period) into
interpretable components: carry/roll-down (the passage of time), first-order
duration, second-order convexity, and an unexplained residual. The components
sum to the actual repriced P&L by construction (the residual closes the gap).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from models import ShortRateModel
from pricing import CouponBondPricer

from ._bond_utils import bond_field, price_bond

#: Numerical guard.
EPS: float = 1.0e-14

_COUPON = CouponBondPricer()


class PnLAttribution:
    """Decompose bond P&L into carry, duration, convexity and residual."""

    def attribute(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
        dr: float,
        dt: float = 0.0,
    ) -> dict:
        """Attribute the bond's P&L over a rate move ``dr`` and time ``dt``.

        The decomposition is a second-order Taylor expansion in the yield::

            dP ≈ carry          (time effect, holding dt at constant rates)
                 - P * D_mod * dr      (first-order duration)
                 + 0.5 * P * C * dr^2  (second-order convexity)
                 + residual            (closes to the actual repriced P&L)

        The duration and convexity sensitivities (``D_mod``, ``C``) are taken
        from :class:`CouponBondPricer` at the base point. ``carry`` is the
        value change from advancing the valuation date by ``dt`` years holding
        the short rate fixed (zero when ``dt = 0``). The ``residual`` is the
        actual repriced P&L minus the explained components and is small for
        small ``dr``.

        Parameters
        ----------
        model : ShortRateModel
            Discounting model.
        r0 : float
            Base short rate.
        bond : Mapping
            Bond specification.
        dr : float
            Parallel short-rate move applied for repricing.
        dt : float, optional
            Holding period in years for the carry/roll component. Default 0.

        Returns
        -------
        dict
            Keys ``carry``, ``duration``, ``convexity``, ``residual``,
            ``total`` (the actual repriced P&L) and ``explained`` (sum of the
            three structural components). ``carry + duration + convexity +
            residual == total`` to numerical precision.
        """
        coupon_rate = float(bond_field(bond, "coupon_rate", 0.0))
        maturity = float(bond_field(bond, "maturity", 1.0))
        freq = int(bond_field(bond, "freq", 2))
        face = float(bond_field(bond, "face", 100.0))
        btype = str(bond.get("type", "coupon")).lower()

        p0 = price_bond(model, r0, bond)

        # First/second order sensitivities at the base point. For a ZCB we
        # approximate via a tiny single coupon-free schedule; otherwise use the
        # coupon pricer's analytic duration/convexity.
        if btype == "zcb":
            bp = _COUPON.price(model, r0, 0.0, maturity, max(freq, 1), face)
        else:
            bp = _COUPON.price(model, r0, coupon_rate, maturity, freq, face)
        mod_dur = float(bp.modified_duration)
        convexity = float(bp.convexity)

        # Total repriced P&L (actual), including the time/carry shift.
        p_new = self._reprice(model, r0, bond, dr, dt)
        total = float(p_new - p0)

        # Carry / roll-down: pure time effect at unchanged rates.
        if dt > 0.0:
            p_carry = self._reprice(model, r0, bond, 0.0, dt)
            carry = float(p_carry - p0)
        else:
            carry = 0.0

        duration = float(-p0 * mod_dur * dr)
        convex = float(0.5 * p0 * convexity * dr**2)
        explained = carry + duration + convex
        residual = total - explained

        return {
            "carry": carry,
            "duration": duration,
            "convexity": convex,
            "residual": residual,
            "explained": explained,
            "total": total,
        }

    @staticmethod
    def _reprice(
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
        dr: float,
        dt: float,
    ) -> float:
        """Reprice the bond at shifted rate ``r0 + dr`` after ``dt`` years.

        Advancing the valuation date by ``dt`` is approximated by shortening the
        bond maturity by ``dt`` (cashflows that have rolled off are dropped from
        the schedule). This is a roll-down approximation suitable for small
        ``dt``; ``dt = 0`` recovers an exact parallel-shift reprice.
        """
        if dt <= 0.0:
            return price_bond(model, r0 + dr, bond)

        new_bond = dict(bond)
        new_mat = max(float(bond_field(bond, "maturity", 1.0)) - dt, EPS)
        new_bond["maturity"] = new_mat
        return price_bond(model, r0 + dr, new_bond)
