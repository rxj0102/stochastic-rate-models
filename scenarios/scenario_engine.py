"""Unified scenario engine.

:class:`ScenarioEngine` provides a single entry point that dispatches a
*scenario specification* to the appropriate concrete scenario
(parallel / twist / historical / custom) and evaluates it on a portfolio. It
also exposes helpers for plotting (before/after curves) and waterfall analysis
(P&L by bond).

Scenario specification
----------------------
``scenario_spec`` is a mapping with a ``type`` key::

    {"type": "parallel",   "shift_bps": 100}
    {"type": "twist",      "kind": "steepener"/"flattener"/"bull_steepener"/...,
                            "short_shift_bps": .., "long_shift_bps": ..}
    {"type": "historical", "name": "2008 Crisis"}
    {"type": "custom",     "tenor_shifts_bps": {2: -300, 10: -200}}
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from models import ShortRateModel

from ._curve import ONE_BP, base_curve, price_bond_off_curve, shift_from_tenor_dict
from .parallel_shift import ParallelShiftScenario
from .stress_test import RateStressTester
from .twist import TwistScenario


class ScenarioEngine:
    """Dispatch scenario specs to concrete scenarios and aggregate P&L."""

    def __init__(self, interp_method: str = "cubic") -> None:
        self._method = interp_method
        self.parallel = ParallelShiftScenario(interp_method=interp_method)
        self.twist = TwistScenario(interp_method=interp_method)
        self.stress = RateStressTester(interp_method=interp_method)

    # -------------------------------------------------- shift construction
    def _shift_vector(
        self,
        mats: np.ndarray,
        scenario_spec: Mapping[str, Any],
    ) -> np.ndarray:
        """Translate a scenario spec into a per-maturity yield shift (decimal)."""
        stype = str(scenario_spec.get("type", "parallel")).lower()

        if stype == "parallel":
            dy = float(scenario_spec.get("shift_bps", 0.0)) * ONE_BP
            return np.full_like(mats, dy)

        if stype == "twist":
            kind = str(scenario_spec.get("kind", "steepener")).lower()
            defaults = _TWIST_DEFAULTS.get(kind, (0.0, 0.0))
            short_bps = float(scenario_spec.get("short_shift_bps", defaults[0]))
            long_bps = float(scenario_spec.get("long_shift_bps", defaults[1]))
            short_tenor = float(scenario_spec.get("short_tenor", self.twist.short_tenor))
            long_tenor = float(scenario_spec.get("long_tenor", self.twist.long_tenor))
            return self.twist._twist_shift(
                mats, short_bps, long_bps, short_tenor, long_tenor
            )

        if stype == "historical":
            name = scenario_spec["name"]
            if name not in self.stress.scenarios:
                raise KeyError(f"unknown historical scenario {name!r}")
            return shift_from_tenor_dict(
                mats, self.stress.scenarios[name], method=self._method
            )

        if stype == "custom":
            return shift_from_tenor_dict(
                mats, scenario_spec["tenor_shifts_bps"], method=self._method
            )

        raise ValueError(f"unknown scenario type {stype!r}")

    # ------------------------------------------------------------------ run
    def run(
        self,
        model: ShortRateModel,
        r0: float,
        portfolio: Sequence[Mapping[str, Any]],
        scenario_spec: Mapping[str, Any],
        maturities: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Run a single scenario over a portfolio and return per-bond P&L.

        Returns
        -------
        pandas.DataFrame
            One row per bond with ``base_price``, ``scenario_price``, ``pnl``,
            ``pct_change`` and ``weight``, plus a final ``TOTAL`` row carrying
            the weighted aggregate.
        """
        mats, base_yields = base_curve(model, r0, maturities)
        shift = self._shift_vector(mats, scenario_spec)
        shifted_yields = base_yields + shift

        rows = []
        total_base = total_scn = 0.0
        for i, b in enumerate(portfolio):
            w = float(b.get("weight", b.get("notional", 1.0)))
            base_p = price_bond_off_curve(mats, base_yields, b, self._method)
            scn_p = price_bond_off_curve(mats, shifted_yields, b, self._method)
            rows.append(
                {
                    "bond": str(b.get("name", f"bond_{i}")),
                    "weight": w,
                    "base_price": base_p,
                    "scenario_price": scn_p,
                    "pnl": w * (scn_p - base_p),
                    "pct_change": (scn_p - base_p) / base_p if base_p else np.nan,
                }
            )
            total_base += w * base_p
            total_scn += w * scn_p

        df = pd.DataFrame(rows)
        total_row = {
            "bond": "TOTAL",
            "weight": np.nan,
            "base_price": total_base,
            "scenario_price": total_scn,
            "pnl": total_scn - total_base,
            "pct_change": (
                (total_scn - total_base) / total_base if total_base else np.nan
            ),
        }
        return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    # ----------------------------------------------------- plotting helper
    def before_after_curve(
        self,
        model: ShortRateModel,
        r0: float,
        scenario_spec: Mapping[str, Any],
        maturities: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(maturities, base_yields, shifted_yields)`` for plotting."""
        mats, base_yields = base_curve(model, r0, maturities)
        shifted_yields = base_yields + self._shift_vector(mats, scenario_spec)
        return mats, base_yields, shifted_yields

    # ---------------------------------------------------- waterfall helper
    def pnl_by_bond(
        self,
        model: ShortRateModel,
        r0: float,
        portfolio: Sequence[Mapping[str, Any]],
        scenario_spec: Mapping[str, Any],
        maturities: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """Weighted P&L per bond (for a waterfall), with a ``TOTAL`` entry."""
        df = self.run(model, r0, portfolio, scenario_spec, maturities)
        return df.set_index("bond")["pnl"]


#: Default short/long anchor shifts (bps) for named twist kinds.
_TWIST_DEFAULTS: dict[str, tuple[float, float]] = {
    "steepener": (-50.0, 50.0),
    "flattener": (50.0, -50.0),
    "bull_steepener": (-75.0, -25.0),
    "bear_steepener": (25.0, 75.0),
    "bull_flattener": (-25.0, -75.0),
    "bear_flattener": (75.0, 25.0),
}
