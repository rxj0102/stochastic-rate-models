"""Fixed income portfolio aggregation.

:class:`FixedIncomePortfolio` holds a collection of zero coupon and coupon bond
positions and aggregates their value and risk (duration, convexity) off a
single short rate model, using :class:`ZCBPricer` and :class:`CouponBondPricer`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from models import ShortRateModel

from .coupon_bond import CouponBondPricer
from .zcb import ZCBPricer

#: Supported instrument types.
TYPE_ZCB: str = "zcb"
TYPE_COUPON: str = "coupon"
#: Numerical guard.
EPS: float = 1.0e-14


class FixedIncomePortfolio:
    """A weighted portfolio of zero coupon and coupon bonds."""

    def __init__(self) -> None:
        self._instruments: list[dict[str, Any]] = []
        self._zcb = ZCBPricer()
        self._coupon = CouponBondPricer()

    @property
    def instruments(self) -> list[dict[str, Any]]:
        """The list of held instrument dictionaries."""
        return self._instruments

    def add_bond(
        self,
        instrument_type: str,
        params: dict[str, Any],
        notional: float = 1.0,
    ) -> "FixedIncomePortfolio":
        """Add a bond position to the portfolio.

        Parameters
        ----------
        instrument_type : str
            Either ``'zcb'`` or ``'coupon'``.
        params : dict
            For a ZCB: ``{'maturity': T, 'face': F}`` (``face`` optional,
            default 1.0). For a coupon bond: ``{'coupon_rate', 'maturity',
            'freq', 'face'}`` (``freq``/``face`` optional).
        notional : float, optional
            Number of units held (a multiplicative weight on the position).

        Returns
        -------
        FixedIncomePortfolio
            ``self`` (for chaining).
        """
        itype = instrument_type.lower()
        if itype not in (TYPE_ZCB, TYPE_COUPON):
            raise ValueError(f"unknown instrument type '{instrument_type}'")
        self._instruments.append(
            {"type": itype, "params": dict(params), "notional": float(notional)}
        )
        return self

    # ------------------------------------------------------------------ pricing
    def _instrument_value(
        self, inst: dict[str, Any], model: ShortRateModel, r0: float
    ) -> float:
        """Present value of a single instrument (per unit notional)."""
        p = inst["params"]
        if inst["type"] == TYPE_ZCB:
            face = p.get("face", 1.0)
            return face * self._zcb.price_analytical(model, r0, p["maturity"])
        return self._coupon.price(
            model,
            r0,
            p["coupon_rate"],
            p["maturity"],
            p.get("freq", 2),
            p.get("face", 100.0),
        ).price

    def total_value(self, model: ShortRateModel, r0: float) -> float:
        """Total present value of the portfolio."""
        return float(
            sum(
                inst["notional"] * self._instrument_value(inst, model, r0)
                for inst in self._instruments
            )
        )

    def _instrument_duration(
        self, inst: dict[str, Any], model: ShortRateModel, r0: float
    ) -> float:
        """Macaulay duration of a single instrument."""
        p = inst["params"]
        if inst["type"] == TYPE_ZCB:
            return float(p["maturity"])  # ZCB duration equals its maturity.
        return self._coupon.duration(
            model,
            r0,
            p["coupon_rate"],
            p["maturity"],
            p.get("freq", 2),
            p.get("face", 100.0),
        )

    def _instrument_convexity(
        self, inst: dict[str, Any], model: ShortRateModel, r0: float
    ) -> float:
        """Convexity of a single instrument."""
        p = inst["params"]
        if inst["type"] == TYPE_ZCB:
            T = float(p["maturity"])
            return T * T  # continuously compounded ZCB convexity.
        return self._coupon.convexity(
            model,
            r0,
            p["coupon_rate"],
            p["maturity"],
            p.get("freq", 2),
            p.get("face", 100.0),
        )

    def portfolio_duration(self, model: ShortRateModel, r0: float) -> float:
        """Value-weighted (Macaulay) duration of the portfolio."""
        total = 0.0
        weighted = 0.0
        for inst in self._instruments:
            val = inst["notional"] * self._instrument_value(inst, model, r0)
            weighted += val * self._instrument_duration(inst, model, r0)
            total += val
        return float(weighted / max(total, EPS))

    def portfolio_convexity(self, model: ShortRateModel, r0: float) -> float:
        """Value-weighted convexity of the portfolio."""
        total = 0.0
        weighted = 0.0
        for inst in self._instruments:
            val = inst["notional"] * self._instrument_value(inst, model, r0)
            weighted += val * self._instrument_convexity(inst, model, r0)
            total += val
        return float(weighted / max(total, EPS))

    def cashflow_table(self) -> pd.DataFrame:
        """Tabulate every scheduled cashflow across all instruments.

        Returns
        -------
        pandas.DataFrame
            Columns ``instrument`` (index), ``type``, ``time`` and ``cashflow``
            (already scaled by the position notional).
        """
        rows = []
        for idx, inst in enumerate(self._instruments):
            p = inst["params"]
            notional = inst["notional"]
            if inst["type"] == TYPE_ZCB:
                face = p.get("face", 1.0)
                rows.append(
                    {
                        "instrument": idx,
                        "type": TYPE_ZCB,
                        "time": float(p["maturity"]),
                        "cashflow": notional * face,
                    }
                )
            else:
                freq = p.get("freq", 2)
                face = p.get("face", 100.0)
                coupon_rate = p["coupon_rate"]
                maturity = p["maturity"]
                n = max(int(round(maturity * freq)), 1)
                coupon = coupon_rate * face / freq
                for i in range(n):
                    t = (i + 1) / freq
                    cf = coupon + (face if i == n - 1 else 0.0)
                    rows.append(
                        {
                            "instrument": idx,
                            "type": TYPE_COUPON,
                            "time": float(t),
                            "cashflow": notional * cf,
                        }
                    )
        return pd.DataFrame(rows)
