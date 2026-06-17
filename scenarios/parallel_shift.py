"""Parallel yield-curve shift scenarios.

A parallel shift moves every point on the zero curve by the same amount. This
is the workhorse rate scenario and underlies duration/DV01 risk: for a small
shift ``dy`` the P&L of a bond is approximately ``-modified_duration * P * dy``.
This module reprices the bond off the shifted curve to capture the full
(non-linear, convexity-inclusive) response and compares it to the linear
duration approximation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from models import ShortRateModel
from pricing import CouponBondPricer

from ._curve import ONE_BP, base_curve, price_bond_off_curve

#: Default set of parallel shifts (basis points) to evaluate.
DEFAULT_SHIFTS_BPS: tuple[int, ...] = (-200, -100, -50, -25, 25, 50, 100, 200)


class ParallelShiftScenario:
    """Apply parallel zero-curve shifts and reprice a bond off the curve."""

    def __init__(self, interp_method: str = "cubic") -> None:
        self._method = interp_method
        self._pricer = CouponBondPricer()

    # ------------------------------------------------------------------ run
    def run(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        shifts_bps: Sequence[float] = DEFAULT_SHIFTS_BPS,
        maturities: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Evaluate a bond under a set of parallel curve shifts.

        Parameters
        ----------
        model : ShortRateModel
            Model used to build the base zero curve.
        r0 : float
            Initial short rate.
        bond : mapping
            Bond specification (see :func:`scenarios._curve.price_bond_off_curve`).
        shifts_bps : sequence of float, optional
            Parallel shifts to apply, in basis points.
        maturities : np.ndarray, optional
            Maturity grid for the curve.

        Returns
        -------
        pandas.DataFrame
            One row per shift with columns ``shift_bps``, ``bond_price``,
            ``pnl``, ``pct_change``, ``duration_approx`` (the linear
            ``-mod_dur * P * dy`` estimate) and ``actual_minus_approx``.
        """
        mats, base_yields = base_curve(model, r0, maturities)
        base_price = price_bond_off_curve(mats, base_yields, bond, self._method)
        mod_dur = self._modified_duration(model, r0, bond, base_price)

        rows = []
        for s_bps in shifts_bps:
            dy = float(s_bps) * ONE_BP
            shifted = base_yields + dy
            price = price_bond_off_curve(mats, shifted, bond, self._method)
            pnl = price - base_price
            approx = -mod_dur * base_price * dy
            rows.append(
                {
                    "shift_bps": float(s_bps),
                    "bond_price": price,
                    "pnl": pnl,
                    "pct_change": pnl / base_price if base_price else np.nan,
                    "duration_approx": approx,
                    "actual_minus_approx": pnl - approx,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------ dv01 validation
    def dv01_validation(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Validate model DV01 against a 1bp curve-repricing P&L.

        A +1bp parallel shift should produce a P&L close to ``-DV01``. Returns
        the two quantities, their absolute and relative differences.

        Returns
        -------
        dict
            Keys ``dv01``, ``repricing_pnl_1bp``, ``neg_dv01``, ``abs_error``,
            ``rel_error`` and ``base_price``.
        """
        mats, base_yields = base_curve(model, r0, maturities)
        base_price = price_bond_off_curve(mats, base_yields, bond, self._method)

        shifted = base_yields + ONE_BP
        price_up = price_bond_off_curve(mats, shifted, bond, self._method)
        pnl_1bp = price_up - base_price

        dv01 = self._dv01(model, r0, bond)
        neg_dv01 = -dv01
        abs_err = abs(pnl_1bp - neg_dv01)
        rel_err = abs_err / abs(neg_dv01) if neg_dv01 else np.nan
        return {
            "base_price": base_price,
            "dv01": dv01,
            "neg_dv01": neg_dv01,
            "repricing_pnl_1bp": pnl_1bp,
            "abs_error": abs_err,
            "rel_error": rel_err,
        }

    # --------------------------------------------------------- model risk
    def _bond_price_metric(
        self, model: ShortRateModel, r0: float, bond: Mapping[str, Any]
    ):
        """Price the bond via the model pricer to recover duration/DV01.

        For a zero coupon bond we treat it as a zero-coupon instrument
        (``coupon_rate=0``) within the coupon pricer.
        """
        coupon_rate = (
            0.0 if str(bond.get("type", "coupon")).lower() == "zcb"
            else float(bond.get("coupon_rate", 0.0))
        )
        return self._pricer.price(
            model,
            r0,
            coupon_rate,
            float(bond["maturity"]),
            int(bond.get("freq", 2)),
            float(bond.get("face", 100.0)),
        )

    def _modified_duration(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        base_price: float,
    ) -> float:
        """Modified duration from the model pricer (guarded for zero price)."""
        bp = self._bond_price_metric(model, r0, bond)
        return float(bp.modified_duration) if bp.modified_duration is not None else 0.0

    def _dv01(
        self, model: ShortRateModel, r0: float, bond: Mapping[str, Any]
    ) -> float:
        """DV01 from the model pricer."""
        bp = self._bond_price_metric(model, r0, bond)
        return float(bp.dv01) if bp.dv01 is not None else 0.0
