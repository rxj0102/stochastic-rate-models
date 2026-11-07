"""Interest rate caps, floors and the underlying swap (Hull-White).

:class:`CapsFloorsPricer` delegates the analytical caplet/cap/floor pricing to
the Hull-White model and supplies the matching forward-starting swap value used
to verify cap-floor parity ``Cap - Floor = Swap``.
"""

from __future__ import annotations

from models import HullWhiteModel

#: Default cap/floor payment frequency (quarterly).
DEFAULT_FREQ: int = 4
#: Parity tolerance in fraction of notional (1 bp).
PARITY_TOL: float = 1.0e-4


def _payment_grid(maturity: float, freq: int) -> list[tuple[float, float]]:
    """Return the ``(start, end)`` accrual periods used by the HW cap pricer.

    The Hull-White ``cap_price`` builds reset times ``(i+1)/freq`` for
    ``i = 0 .. n-1`` (``n = round(maturity*freq)``) and prices caplets over
    consecutive pairs, i.e. periods ``[i/freq, (i+1)/freq]`` for
    ``i = 1 .. n-1``. The swap value must telescope over the same periods.
    """
    n = int(round(maturity * freq))
    times = [(i + 1) / freq for i in range(n)]
    return [(times[i - 1], times[i]) for i in range(1, len(times))]


class CapsFloorsPricer:
    """Price caps, floors and the underlying swap under Hull-White."""

    def caplet_price_hw(
        self,
        hw_model: HullWhiteModel,
        K_cap: float,
        T_start: float,
        T_end: float,
        notional: float = 1.0,
    ) -> float:
        """Price a single caplet (delegates to the Hull-White model)."""
        return float(hw_model.caplet_price(K_cap, T_start, T_end, notional))

    def cap_price(
        self,
        hw_model: HullWhiteModel,
        K_cap: float,
        maturity: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Price an interest rate cap (delegates to the Hull-White model)."""
        return float(hw_model.cap_price(K_cap, maturity, notional, freq))

    def floor_price(
        self,
        hw_model: HullWhiteModel,
        K_floor: float,
        maturity: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Price an interest rate floor (delegates to the Hull-White model)."""
        return float(hw_model.floor_price(K_floor, maturity, notional, freq))

    def swap_value(
        self,
        hw_model: HullWhiteModel,
        swap_rate: float,
        maturity: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Value of a receive-float pay-fixed swap on the cap/floor grid.

        The floating leg telescopes to ``P(0, first_start) - P(0, last_end)``;
        the fixed leg is ``swap_rate * sum(delta * P(0, t_end))``. The accrual
        grid is identical to the one used by the Hull-White cap/floor pricers so
        that ``Cap - Floor = Swap`` holds exactly.

        Returns
        -------
        float
            Net swap value (positive when floating exceeds fixed).
        """
        periods = _payment_grid(maturity, freq)
        if not periods:
            return 0.0
        delta = 1.0 / freq
        first_start = periods[0][0]
        last_end = periods[-1][1]

        float_leg = hw_model.market_zcb(first_start) - hw_model.market_zcb(last_end)
        fixed_leg = swap_rate * sum(
            delta * hw_model.market_zcb(end) for _, end in periods
        )
        return float(notional * (float_leg - fixed_leg))

    def cap_floor_parity_check(
        self,
        hw_model: HullWhiteModel,
        rate: float,
        maturity: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
    ) -> dict:
        """Verify cap-floor parity ``Cap - Floor = Swap``.

        Returns
        -------
        dict
            Keys ``cap``, ``floor``, ``swap``, ``cap_minus_floor``,
            ``difference`` (``Cap - Floor - Swap``) and ``passes`` (whether the
            absolute difference is within 1 bp of notional).
        """
        cap = self.cap_price(hw_model, rate, maturity, notional, freq)
        floor = self.floor_price(hw_model, rate, maturity, notional, freq)
        swap = self.swap_value(hw_model, rate, maturity, notional, freq)
        cap_minus_floor = cap - floor
        difference = cap_minus_floor - swap
        return {
            "cap": cap,
            "floor": floor,
            "swap": swap,
            "cap_minus_floor": cap_minus_floor,
            "difference": difference,
            "passes": abs(difference) < PARITY_TOL * notional,
        }
