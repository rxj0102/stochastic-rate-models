"""Yield-curve scenario analysis for the stochastic rate model library.

Scenarios operate on the zero (spot) yield curve rather than directly on the
short rate, which keeps them model agnostic. A scenario maps a base zero curve
(built from any :class:`~models.ShortRateModel`) to a shifted curve, and bonds
are repriced by discounting their cashflows off the shifted curve::

    P&L = price(shifted_curve) - price(base_curve)

The package provides parallel shifts, twists (steepeners/flatteners),
pre-built historical stress tests, and a unified :class:`ScenarioEngine`.
"""

from ._curve import (
    DEFAULT_MATURITIES,
    base_curve,
    interp_yields,
    price_bond_off_curve,
    shift_from_tenor_dict,
)
from .parallel_shift import ParallelShiftScenario
from .scenario_engine import ScenarioEngine
from .stress_test import SCENARIOS, RateStressTester
from .twist import TwistScenario

__all__ = [
    "ParallelShiftScenario",
    "TwistScenario",
    "RateStressTester",
    "SCENARIOS",
    "ScenarioEngine",
    "price_bond_off_curve",
    "base_curve",
    "interp_yields",
    "shift_from_tenor_dict",
    "DEFAULT_MATURITIES",
]
