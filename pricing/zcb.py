"""Zero coupon bond pricing utilities.

This module provides :class:`ZCBPricer`, a thin model-agnostic wrapper around
the analytical and Monte Carlo zero coupon bond (ZCB) pricing capabilities of
the :mod:`models` package. It additionally exposes convenience routines for
comparing Monte Carlo against analytical prices and for tabulating the term
structure implied by a model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from models import ShortRateModel
from utils.statistics import mc_discount_factor

#: Numerical guard for the minimum maturity used in yield/forward calculations.
MIN_TIME: float = 1e-12
#: Basis point scaling factor (1 bp = 1e-4).
BPS: float = 1.0e4
#: Finite-difference bump used when computing instantaneous forward rates.
FORWARD_DT: float = 1.0e-4


class ZCBPricer:
    """Price zero coupon bonds analytically and by Monte Carlo simulation."""

    def price_analytical(self, model: ShortRateModel, r0: float, T: float) -> float:
        """Analytical zero coupon bond price :math:`P(0, T)`.

        Parameters
        ----------
        model : ShortRateModel
            Any short rate model exposing ``zcb_price``.
        r0 : float
            Current short rate.
        T : float
            Bond maturity in years.

        Returns
        -------
        float
            Present value of one unit of face delivered at ``T``.
        """
        return float(model.zcb_price(r0, float(T)))

    def yield_analytical(self, model: ShortRateModel, r0: float, T: float) -> float:
        """Continuously compounded spot yield :math:`R(0, T) = -\\ln P / T`."""
        T = max(float(T), MIN_TIME)
        price = self.price_analytical(model, r0, T)
        return float(-np.log(price) / T)

    def price_mc(
        self,
        model: ShortRateModel,
        r0: float,
        T: float,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: Optional[int] = None,
    ) -> tuple[float, float, tuple[float, float]]:
        """Monte Carlo zero coupon bond price via path discounting.

        The price is estimated as ``E[exp(-integral_0^T r(u) du)]`` using the
        trapezoidal rule on each simulated short rate path.

        Returns
        -------
        tuple
            ``(price, standard_error, (ci_low, ci_high))``.
        """
        sim = model.simulate(r0, float(T), n_paths, n_steps, seed)
        return mc_discount_factor(sim.paths, sim.t_grid)

    def compare_mc_vs_analytical(
        self,
        model: ShortRateModel,
        r0: float,
        maturities: np.ndarray,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Compare Monte Carlo against analytical ZCB prices across maturities.

        Returns
        -------
        pandas.DataFrame
            Columns ``maturity``, ``analytical``, ``mc``, ``stderr`` and
            ``deviation_bps`` (the MC minus analytical price difference in basis
            points of face value).
        """
        rows = []
        for T in np.asarray(maturities, dtype=float):
            analytical = self.price_analytical(model, r0, float(T))
            mc, stderr, _ = self.price_mc(
                model, r0, float(T), n_paths=n_paths, n_steps=n_steps, seed=seed
            )
            rows.append(
                {
                    "maturity": float(T),
                    "analytical": analytical,
                    "mc": mc,
                    "stderr": stderr,
                    "deviation_bps": (mc - analytical) * BPS,
                }
            )
        return pd.DataFrame(rows)

    def term_structure(
        self, model: ShortRateModel, r0: float, maturities: np.ndarray
    ) -> pd.DataFrame:
        """Tabulate the analytical term structure of a model.

        Returns
        -------
        pandas.DataFrame
            Columns ``T``, ``price`` (``P(0,T)``), ``yield`` (``R(0,T)``) and
            ``forward`` (the instantaneous forward ``f(0,T)``).
        """
        maturities = np.asarray(maturities, dtype=float)
        rows = []
        for T in maturities:
            T = float(T)
            price = self.price_analytical(model, r0, T)
            yld = self.yield_analytical(model, r0, T)
            fwd = self._forward_rate(model, r0, T)
            rows.append({"T": T, "price": price, "yield": yld, "forward": fwd})
        return pd.DataFrame(rows)

    @staticmethod
    def _forward_rate(model: ShortRateModel, r0: float, T: float) -> float:
        """Instantaneous forward rate ``f(0,T) = -d/dT ln P(0,T)``.

        Computed with a centred finite difference of the log discount curve.
        """
        T = max(float(T), MIN_TIME)
        dt = FORWARD_DT
        lp_up = np.log(model.zcb_price(r0, T + dt))
        lp_dn = np.log(model.zcb_price(r0, max(T - dt, MIN_TIME)))
        return float(-(lp_up - lp_dn) / (2.0 * dt))
