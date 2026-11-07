"""Yield-curve twist (steepener / flattener) scenarios.

A twist anchors a shift at a short tenor and a long tenor and interpolates the
shift linearly across maturities between the anchors (with flat extrapolation
beyond them). Steepeners increase the long-minus-short slope; flatteners
decrease it. The ``bull``/``bear`` prefixes describe the *direction of the
overall level move*: a bull move has rates falling (bond-positive), a bear move
has rates rising (bond-negative).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from models import ShortRateModel

from ._curve import ONE_BP, base_curve, price_bond_off_curve


class TwistScenario:
    """Construct steepener / flattener twists and reprice a bond off the curve."""

    def __init__(
        self,
        short_tenor: float = 2.0,
        long_tenor: float = 10.0,
        interp_method: str = "cubic",
    ) -> None:
        self.short_tenor = float(short_tenor)
        self.long_tenor = float(long_tenor)
        self._method = interp_method

    # ----------------------------------------------------------- core twist
    def _twist_shift(
        self,
        maturities: np.ndarray,
        short_shift_bps: float,
        long_shift_bps: float,
        short_tenor: float,
        long_tenor: float,
    ) -> np.ndarray:
        """Linear-in-maturity shift (decimal) anchored at two tenors.

        The shift equals ``short_shift_bps`` at ``short_tenor`` and
        ``long_shift_bps`` at ``long_tenor``; between them it is linearly
        interpolated and beyond them it is held flat (clamped) to avoid
        runaway extrapolation at the wings.
        """
        maturities = np.asarray(maturities, dtype=float)
        if long_tenor == short_tenor:
            shift_bps = np.full_like(maturities, long_shift_bps)
        else:
            slope = (long_shift_bps - short_shift_bps) / (long_tenor - short_tenor)
            shift_bps = short_shift_bps + slope * (maturities - short_tenor)
            lo, hi = sorted((short_shift_bps, long_shift_bps))
            shift_bps = np.clip(shift_bps, lo, hi)
        return shift_bps * ONE_BP

    def _apply(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float,
        long_shift_bps: float,
        short_tenor: float,
        long_tenor: float,
        maturities: Optional[np.ndarray],
        label: str,
    ) -> dict:
        mats, base_yields = base_curve(model, r0, maturities)
        base_price = price_bond_off_curve(mats, base_yields, bond, self._method)
        shift = self._twist_shift(
            mats, short_shift_bps, long_shift_bps, short_tenor, long_tenor
        )
        shifted_yields = base_yields + shift
        price = price_bond_off_curve(mats, shifted_yields, bond, self._method)
        return {
            "scenario": label,
            "short_shift_bps": float(short_shift_bps),
            "long_shift_bps": float(long_shift_bps),
            "base_price": base_price,
            "price": price,
            "pnl": price - base_price,
            "pct_change": (price - base_price) / base_price if base_price else np.nan,
            "maturities": mats,
            "base_yields": base_yields,
            "shifted_yields": shifted_yields,
        }

    # --------------------------------------------------------- public twists
    def steepener(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float,
        long_shift_bps: float,
        short_tenor: float = 2.0,
        long_tenor: float = 10.0,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Steepener twist: long end rises relative to the short end.

        Returns
        -------
        dict
            ``price``, ``pnl``, ``pct_change`` and the ``base``/``shifted``
            curves (``maturities``, ``base_yields``, ``shifted_yields``).
        """
        return self._apply(
            model, r0, bond, short_shift_bps, long_shift_bps,
            short_tenor, long_tenor, maturities, "steepener",
        )

    def flattener(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float,
        long_shift_bps: float,
        short_tenor: float = 2.0,
        long_tenor: float = 10.0,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Flattener twist: short end rises relative to the long end.

        Has the same signature as :meth:`steepener`; the naming simply reflects
        the intended slope move. Returns the same dictionary structure.
        """
        return self._apply(
            model, r0, bond, short_shift_bps, long_shift_bps,
            short_tenor, long_tenor, maturities, "flattener",
        )

    # ------------------------------------------------- convenience wrappers
    def bull_steepener(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float = -75.0,
        long_shift_bps: float = -25.0,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Bull steepener: rates fall, short end falls more (curve steepens)."""
        return self.steepener(
            model, r0, bond, short_shift_bps, long_shift_bps,
            self.short_tenor, self.long_tenor, maturities,
        )

    def bear_steepener(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float = 25.0,
        long_shift_bps: float = 75.0,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Bear steepener: rates rise, long end rises more (curve steepens)."""
        return self.steepener(
            model, r0, bond, short_shift_bps, long_shift_bps,
            self.short_tenor, self.long_tenor, maturities,
        )

    def bull_flattener(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float = -25.0,
        long_shift_bps: float = -75.0,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Bull flattener: rates fall, long end falls more (curve flattens)."""
        return self.flattener(
            model, r0, bond, short_shift_bps, long_shift_bps,
            self.short_tenor, self.long_tenor, maturities,
        )

    def bear_flattener(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping[str, Any],
        short_shift_bps: float = 75.0,
        long_shift_bps: float = 25.0,
        maturities: Optional[np.ndarray] = None,
    ) -> dict:
        """Bear flattener: rates rise, short end rises more (curve flattens)."""
        return self.flattener(
            model, r0, bond, short_shift_bps, long_shift_bps,
            self.short_tenor, self.long_tenor, maturities,
        )
