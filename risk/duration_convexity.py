"""Cashflow- and model-based interest rate sensitivity measures.

This module exposes :class:`RateSensitivity`, a collection of stateless
methods for the standard first and second order interest rate risk measures.
Two families of routines are provided:

* **Cashflow based** measures (Macaulay/modified duration, dollar duration,
  convexity, and a Taylor price-change approximation) that depend only on a
  set of cashflows, their payment times and a yield to maturity.
* **Model based** measures (:meth:`RateSensitivity.empirical_duration`) that
  reprice a bond off a short rate model under small parallel shifts of the
  initial short rate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from models import ShortRateModel

from ._bond_utils import (
    DEFAULT_FREQ as _DEFAULT_FREQ,
    bond_field,
    price_bond,
    price_bond_parallel_shift,
)

#: Default coupon frequency (semi-annual).
DEFAULT_FREQ: int = 2
#: Numerical guard for divisions.
EPS: float = 1.0e-14


class RateSensitivity:
    """Duration and convexity measures for fixed income instruments."""

    # ------------------------------------------------------------------
    # Cashflow-based measures (no model required)
    # ------------------------------------------------------------------
    @staticmethod
    def macaulay_duration(
        cashflows: Sequence[float],
        times: Sequence[float],
        ytm: float,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Macaulay duration of a stream of cashflows.

        Parameters
        ----------
        cashflows : sequence of float
            Cashflow amounts.
        times : sequence of float
            Payment times (in years) of each cashflow.
        ytm : float
            Periodic-compounded yield to maturity.
        freq : int, optional
            Compounding frequency per year. Default 2.

        Returns
        -------
        float
            The present-value weighted average time to cashflow.
        """
        cfs = np.asarray(cashflows, dtype=float)
        ts = np.asarray(times, dtype=float)
        disc = (1.0 + ytm / freq) ** (-freq * ts)
        pvs = cfs * disc
        price = float(pvs.sum())
        if abs(price) < EPS:
            return 0.0
        return float(np.sum(ts * pvs) / price)

    @staticmethod
    def modified_duration(
        cashflows: Sequence[float],
        times: Sequence[float],
        ytm: float,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Modified duration ``D_mac / (1 + ytm / freq)``."""
        d_mac = RateSensitivity.macaulay_duration(cashflows, times, ytm, freq)
        return float(d_mac / (1.0 + ytm / freq))

    @staticmethod
    def dollar_duration(price: float, mod_duration: float) -> float:
        """Dollar (value) duration ``price * mod_duration``."""
        return float(price * mod_duration)

    @staticmethod
    def convexity(
        cashflows: Sequence[float],
        times: Sequence[float],
        ytm: float,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Convexity of a stream of cashflows.

        Uses the closed-form periodic-compounding convexity

        .. math::

            C = \\frac{1}{P}\\sum_i \\frac{CF_i\\, t_i (t_i + 1/m)}
                {(1 + y/m)^{m t_i + 2}}

        where ``m`` is the compounding frequency.
        """
        cfs = np.asarray(cashflows, dtype=float)
        ts = np.asarray(times, dtype=float)
        base = 1.0 + ytm / freq
        disc = base ** (-freq * ts)
        pvs = cfs * disc
        price = float(pvs.sum())
        if abs(price) < EPS:
            return 0.0
        numer = np.sum(pvs * ts * (ts + 1.0 / freq)) / base**2
        return float(numer / price)

    @staticmethod
    def price_change_approx(
        price: float, mod_dur: float, convexity: float, dy: float
    ) -> float:
        """Second-order Taylor approximation of the price change.

        Returns ``price * (-mod_dur * dy + 0.5 * convexity * dy^2)``.
        """
        return float(price * (-mod_dur * dy + 0.5 * convexity * dy**2))

    # ------------------------------------------------------------------
    # Model-based measures
    # ------------------------------------------------------------------
    @staticmethod
    def empirical_duration(
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
        dr: float = 1.0e-4,
    ) -> float:
        """Effective (empirical) duration via repricing under curve shifts.

        A parallel ``+/- dr`` shift is applied to the model-implied periodic
        zero curve and the bond is repriced; the central difference is
        normalised by the base price

        .. math::

            D_{eff} = \\frac{P(-dr) - P(+dr)}{2\\, dr\\, P(r_0)}.

        Shifting the zero curve in parallel (rather than the short rate ``r0``)
        makes the effective duration consistent with the analytic modified
        duration of the bond.

        Parameters
        ----------
        model : ShortRateModel
            Short rate model used to discount the bond cashflows.
        r0 : float
            Base short rate.
        bond : Mapping
            Bond specification (see :mod:`risk._bond_utils`).
        dr : float, optional
            Parallel shift applied to the zero curve. Default 1e-4.

        Returns
        -------
        float
            The effective duration of the bond.
        """
        freq = int(bond_field(bond, "freq", _DEFAULT_FREQ))
        p_up = price_bond_parallel_shift(model, r0, bond, dr, freq)
        p_dn = price_bond_parallel_shift(model, r0, bond, -dr, freq)
        p_0 = price_bond(model, r0, bond)
        if abs(p_0) < EPS or dr <= 0.0:
            return 0.0
        return float((p_dn - p_up) / (2.0 * dr * p_0))
