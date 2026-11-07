"""Historical rate stress scenarios.

This module ships a small library of stylised historical rate shocks expressed
as per-key-tenor shifts (basis points) at the tenors ``[0.25, 1, 2, 5, 10, 30]``
years. A scenario is applied by spline-interpolating its key-tenor shifts onto
the full maturity grid and repricing the bond/portfolio off the shifted curve.

The shocks are deliberate, rounded caricatures of well-known episodes intended
for what-if analysis, not point-in-time historical reconstructions.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from models import ShortRateModel

from ._curve import base_curve, price_bond_off_curve, shift_from_tenor_dict

#: Key tenors (years) at which historical shocks are anchored.
KEY_TENORS: tuple[float, ...] = (0.25, 1.0, 2.0, 5.0, 10.0, 30.0)

#: Pre-built historical scenarios as ``{tenor_years: shift_bps}`` dictionaries.
SCENARIOS: dict[str, dict[float, float]] = {
    # Flight-to-quality rally: front end collapses, belly/long rally less.
    "2008 Crisis": {0.25: -350.0, 1.0: -325.0, 2.0: -300.0, 5.0: -250.0,
                    10.0: -200.0, 30.0: -150.0},
    # Long-end driven sell-off on Fed taper signalling.
    "2013 Taper Tantrum": {0.25: 5.0, 1.0: 15.0, 2.0: 40.0, 5.0: 90.0,
                           10.0: 130.0, 30.0: 120.0},
    # Aggressive hiking cycle: very large front-end shock, bear flattening.
    "2022 Rate Hike Cycle": {0.25: 500.0, 1.0: 480.0, 2.0: 450.0, 5.0: 350.0,
                             10.0: 250.0, 30.0: 200.0},
    # Pandemic dash-for-cash: sharp broad rally, front end leads.
    "COVID March 2020": {0.25: -150.0, 1.0: -150.0, 2.0: -150.0, 5.0: -135.0,
                         10.0: -120.0, 30.0: -100.0},
    # 1994 bond massacre: broad parallel-ish bear shift.
    "1994 Bond Massacre": {0.25: 200.0, 1.0: 215.0, 2.0: 230.0, 5.0: 245.0,
                           10.0: 250.0, 30.0: 240.0},
}


class RateStressTester:
    """Apply pre-built or custom historical rate stresses to bonds/portfolios."""

    def __init__(self, interp_method: str = "cubic") -> None:
        self._method = interp_method
        self.scenarios = SCENARIOS

    # --------------------------------------------------------- named stress
    def apply_scenario(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        scenario_name: str,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Apply a named historical scenario to a single bond.

        Parameters
        ----------
        scenario_name : str
            Key into :data:`SCENARIOS`.

        Returns
        -------
        dict
            ``price``, ``pnl``, ``pct`` plus the shifted curve for plotting.
        """
        if scenario_name not in self.scenarios:
            raise KeyError(
                f"unknown scenario {scenario_name!r}; "
                f"available: {sorted(self.scenarios)}"
            )
        return self.custom_scenario(
            model, r0, bond, self.scenarios[scenario_name], maturities,
            _label=scenario_name,
        )

    # -------------------------------------------------------- custom stress
    def custom_scenario(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        tenor_shifts_bps: Mapping[float, float],
        maturities: Optional[np.ndarray] = None,
        _label: str = "custom",
    ) -> dict:
        """Apply a user-defined per-tenor shock (spline-interpolated to the grid).

        Parameters
        ----------
        tenor_shifts_bps : mapping of float to float
            Shift in basis points keyed by tenor (years).

        Returns
        -------
        dict
            ``scenario``, ``base_price``, ``price``, ``pnl``, ``pct`` and the
            ``maturities``/``base_yields``/``shifted_yields`` curves.
        """
        mats, base_yields = base_curve(model, r0, maturities)
        base_price = price_bond_off_curve(mats, base_yields, bond, self._method)
        shift = shift_from_tenor_dict(mats, tenor_shifts_bps, method=self._method)
        shifted_yields = base_yields + shift
        price = price_bond_off_curve(mats, shifted_yields, bond, self._method)
        pnl = price - base_price
        return {
            "scenario": _label,
            "base_price": base_price,
            "price": price,
            "pnl": pnl,
            "pct": pnl / base_price if base_price else np.nan,
            "maturities": mats,
            "base_yields": base_yields,
            "shifted_yields": shifted_yields,
        }

    # ---------------------------------------------------- portfolio sweep
    def run_all_scenarios(
        self,
        model: ShortRateModel,
        r0: float,
        portfolio: Sequence[Mapping[str, Any]],
        maturities: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Apply every pre-built scenario to a portfolio of bonds.

        Parameters
        ----------
        portfolio : sequence of mapping
            Bond specifications. Each may carry an optional ``weight`` or
            ``notional`` (default 1.0) scaling its P&L, and an optional
            ``name`` used as the column label.
        maturities : np.ndarray, optional
            Curve grid.

        Returns
        -------
        pandas.DataFrame
            Indexed by scenario name with one column per bond P&L plus a
            ``total`` column (weighted sum across the portfolio).
        """
        mats, base_yields = base_curve(model, r0, maturities)
        labels = [
            str(b.get("name", f"bond_{i}")) for i, b in enumerate(portfolio)
        ]
        weights = [
            float(b.get("weight", b.get("notional", 1.0))) for b in portfolio
        ]
        base_prices = [
            price_bond_off_curve(mats, base_yields, b, self._method)
            for b in portfolio
        ]

        rows: dict[str, dict[str, float]] = {}
        for name, tenor_shifts in self.scenarios.items():
            shift = shift_from_tenor_dict(mats, tenor_shifts, method=self._method)
            shifted_yields = base_yields + shift
            row: dict[str, float] = {}
            total = 0.0
            for label, w, base_p, b in zip(
                labels, weights, base_prices, portfolio
            ):
                price = price_bond_off_curve(mats, shifted_yields, b, self._method)
                pnl = w * (price - base_p)
                row[label] = pnl
                total += pnl
            row["total"] = total
            rows[name] = row

        return pd.DataFrame.from_dict(rows, orient="index")
