"""DV01 and key rate duration analytics for fixed income instruments.

This module exposes :class:`DV01Engine`, which computes the dollar value of a
one basis point move (DV01) for single bonds and portfolios, together with key
rate (partial) durations obtained by a triangular "tent" bump of the model
implied zero curve.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from models import ShortRateModel

from ._bond_utils import bond_cashflows, price_bond, price_bond_from_curve

#: One basis point.
ONE_BP: float = 1.0e-4
#: Numerical guard for divisions.
EPS: float = 1.0e-14
#: Default key tenors (years) for key rate durations.
DEFAULT_KEY_TENORS: tuple[float, ...] = (2.0, 5.0, 10.0, 30.0)


class DV01Engine:
    """Compute DV01, key rate durations and portfolio DV01."""

    def dv01(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
    ) -> float:
        """Dollar value of a 1bp parallel shift in the short rate.

        The bond is repriced at ``r0 - 0.5bp`` and ``r0 + 0.5bp`` and the
        central difference gives the price change for a full ``+1bp`` move.
        The sign convention returns a positive number when an upward rate move
        produces a loss (the usual case for a long bond position).

        Parameters
        ----------
        model : ShortRateModel
            Discounting model.
        r0 : float
            Base short rate.
        bond : Mapping
            Bond specification.

        Returns
        -------
        float
            DV01: the loss (positive) for a ``+1bp`` parallel rate increase.
        """
        half = 0.5 * ONE_BP
        p_up = price_bond(model, r0 + half, bond)
        p_dn = price_bond(model, r0 - half, bond)
        # price change for +1bp = P(r0 + 0.5bp) - P(r0 - 0.5bp); loss => negate.
        return float(-(p_up - p_dn))

    def key_rate_durations(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
        key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    ) -> dict[float, float]:
        """Key rate (partial) durations via triangular tent bumps.

        The model-implied continuously-compounded zero curve is built on a fine
        grid. Around each key tenor a triangular "tent" of unit peak height is
        applied: the bump at a key tenor is ``1bp`` and decays linearly to zero
        at the neighbouring key tenors (and to the curve endpoints for the first
        and last buckets). The bond is repriced off the bumped curve and the key
        rate duration is

        .. math::

            KRD_k = -\\frac{P_{bumped} - P_0}{P_0 \\cdot \\Delta y},

        with ``\\Delta y = 1bp``. Because the tents form a partition of unity
        over the curve, summing the key rate durations approximately reproduces
        the effective duration of the bond.

        Parameters
        ----------
        model : ShortRateModel
            Discounting model.
        r0 : float
            Base short rate.
        bond : Mapping
            Bond specification.
        key_tenors : sequence of float, optional
            Key tenor buckets in years. Default ``(2, 5, 10, 30)``.

        Returns
        -------
        dict
            Mapping ``{tenor: key_rate_duration}``.
        """
        key_tenors = sorted(float(t) for t in key_tenors)
        times, _ = bond_cashflows(bond)
        max_time = float(max(times.max(), key_tenors[-1]))

        # Fine zero-curve grid covering all cashflows and key tenors.
        grid = np.unique(
            np.concatenate(
                [
                    np.linspace(EPS, max_time, 200),
                    np.asarray(key_tenors, dtype=float),
                    times,
                ]
            )
        )
        base_rates = np.array(
            [-np.log(max(model.zcb_price(r0, float(T)), EPS)) / max(float(T), EPS) for T in grid]
        )

        p_base = price_bond_from_curve(grid, base_rates, bond)
        if abs(p_base) < EPS:
            return {t: 0.0 for t in key_tenors}

        krd: dict[float, float] = {}
        for i, tenor in enumerate(key_tenors):
            left = key_tenors[i - 1] if i > 0 else grid[0]
            right = key_tenors[i + 1] if i < len(key_tenors) - 1 else grid[-1]
            tent = self._tent(grid, left, tenor, right)
            bumped = base_rates + ONE_BP * tent
            p_bumped = price_bond_from_curve(grid, bumped, bond)
            krd[tenor] = float(-(p_bumped - p_base) / (p_base * ONE_BP))
        return krd

    @staticmethod
    def _tent(
        grid: np.ndarray, left: float, peak: float, right: float
    ) -> np.ndarray:
        """Triangular tent basis of unit peak height at ``peak``.

        Rises linearly from 0 at ``left`` to 1 at ``peak`` and falls to 0 at
        ``right``; zero outside ``[left, right]``.
        """
        tent = np.zeros_like(grid)
        up = (grid > left) & (grid <= peak)
        if peak > left:
            tent[up] = (grid[up] - left) / (peak - left)
        else:
            tent[grid <= peak] = 1.0
        dn = (grid > peak) & (grid < right)
        if right > peak:
            tent[dn] = (right - grid[dn]) / (right - peak)
        return tent

    def dv01_portfolio(
        self,
        model: ShortRateModel,
        r0: float,
        bond_list: Sequence[Mapping],
        weights: Sequence[float],
    ) -> float:
        """Weighted sum of individual bond DV01s.

        Parameters
        ----------
        model : ShortRateModel
            Discounting model.
        r0 : float
            Base short rate.
        bond_list : sequence of Mapping
            Bonds in the portfolio.
        weights : sequence of float
            Position weights (e.g. number of bonds held) aligned with
            ``bond_list``.

        Returns
        -------
        float
            Portfolio DV01.
        """
        if len(bond_list) != len(weights):
            raise ValueError("bond_list and weights must have equal length")
        return float(
            sum(w * self.dv01(model, r0, b) for b, w in zip(bond_list, weights))
        )
