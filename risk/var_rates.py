"""Monte Carlo and historical Value-at-Risk for interest rate instruments.

This module exposes :class:`InterestRateVaR`, which simulates the short rate to
a risk horizon, reprices bonds off the simulated terminal rate (including
realised coupon income), and reports Value-at-Risk (VaR) and Conditional VaR
(CVaR / expected shortfall) of the profit-and-loss distribution. A historical
VaR helper operating on a supplied P&L array is also provided.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from models import ShortRateModel

from ._bond_utils import (
    price_bond,
    price_bond_at_horizon,
    realised_income,
)

#: Numerical guard.
EPS: float = 1.0e-14


class InterestRateVaR:
    """Value-at-Risk and Conditional VaR for rate-sensitive instruments."""

    def _simulate_terminal(
        self,
        model: ShortRateModel,
        r0: float,
        horizon: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """Simulate the short rate to ``horizon`` and return terminal rates."""
        sim = model.simulate(r0, float(horizon), n_paths, n_steps, seed)
        return np.asarray(sim.terminal, dtype=float)

    def pnl_distribution_data(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
        horizon: float,
        n_paths: int = 100_000,
        n_steps: int = 50,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulated P&L samples of holding the bond to ``horizon``.

        The P&L of each path is

        .. math::

            \\Pi = P(\\text{horizon}; r_T) + \\text{realised income} - P_0,

        where ``P_0`` is today's price, ``P(horizon; r_T)`` is the value of the
        remaining cashflows discounted from the horizon given the simulated
        terminal short rate ``r_T``, and realised income is the sum of coupons
        (and any redemption) paid on or before the horizon.

        Returns
        -------
        np.ndarray
            P&L samples, shape ``(n_paths,)``.
        """
        p0 = price_bond(model, r0, bond)
        r_term = self._simulate_terminal(
            model, r0, horizon, n_paths, n_steps, seed
        )
        p_horizon = price_bond_at_horizon(model, r_term, bond, horizon)
        income = realised_income(bond, horizon)
        return np.asarray(p_horizon + income - p0, dtype=float)

    @staticmethod
    def _var_cvar(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
        """Return ``(VaR, CVaR)`` as positive loss numbers at ``confidence``.

        VaR is the ``(1 - confidence)`` lower quantile of P&L, reported as a
        positive loss. CVaR is the mean loss in the tail beyond the VaR.
        """
        pnl = np.asarray(pnl, dtype=float)
        alpha = 1.0 - confidence
        q = float(np.quantile(pnl, alpha))
        var = -q
        tail = pnl[pnl <= q]
        cvar = -float(tail.mean()) if tail.size > 0 else var
        return var, cvar

    def bond_var(
        self,
        model: ShortRateModel,
        r0: float,
        bond: Mapping,
        horizon: float,
        confidence: float = 0.99,
        n_paths: int = 100_000,
        n_steps: int = 50,
        seed: Optional[int] = None,
    ) -> dict:
        """Monte Carlo VaR and CVaR for a single bond over ``horizon``.

        Returns
        -------
        dict
            Keys ``var``, ``cvar`` (positive losses at the chosen
            ``confidence``), ``mean_pnl``, ``std_pnl``, ``confidence`` and
            ``horizon``.
        """
        pnl = self.pnl_distribution_data(
            model, r0, bond, horizon, n_paths, n_steps, seed
        )
        var, cvar = self._var_cvar(pnl, confidence)
        return {
            "var": var,
            "cvar": cvar,
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std()),
            "confidence": float(confidence),
            "horizon": float(horizon),
        }

    def portfolio_var(
        self,
        model: ShortRateModel,
        r0: float,
        bonds: Sequence[Mapping],
        weights: Sequence[float],
        horizon: float,
        confidence: float = 0.99,
        n_paths: int = 100_000,
        n_steps: int = 50,
        seed: Optional[int] = None,
    ) -> dict:
        """Monte Carlo VaR/CVaR for a weighted bond portfolio.

        All bonds are repriced off the *same* simulated short rate paths so
        that diversification and offsetting positions are captured. The
        portfolio P&L is the weighted sum of the per-bond path P&Ls.

        Returns
        -------
        dict
            Same keys as :meth:`bond_var` plus ``n_bonds``.
        """
        if len(bonds) != len(weights):
            raise ValueError("bonds and weights must have equal length")
        r_term = self._simulate_terminal(
            model, r0, horizon, n_paths, n_steps, seed
        )
        income_factor = realised_income  # local alias

        port_pnl = np.zeros(int(n_paths), dtype=float)
        for bond, w in zip(bonds, weights):
            p0 = price_bond(model, r0, bond)
            p_h = price_bond_at_horizon(model, r_term, bond, horizon)
            pnl = p_h + income_factor(bond, horizon) - p0
            port_pnl += float(w) * pnl

        var, cvar = self._var_cvar(port_pnl, confidence)
        return {
            "var": var,
            "cvar": cvar,
            "mean_pnl": float(port_pnl.mean()),
            "std_pnl": float(port_pnl.std()),
            "confidence": float(confidence),
            "horizon": float(horizon),
            "n_bonds": len(bonds),
        }

    @staticmethod
    def historical_var(
        pnl_changes: Sequence[float], confidence: float = 0.99
    ) -> dict:
        """VaR and CVaR from an array of historical/realised P&L changes.

        Parameters
        ----------
        pnl_changes : sequence of float
            Observed profit-and-loss observations (gains positive).
        confidence : float, optional
            Confidence level. Default 0.99.

        Returns
        -------
        dict
            Keys ``var``, ``cvar`` (positive losses), ``mean_pnl``,
            ``std_pnl`` and ``confidence``.
        """
        pnl = np.asarray(pnl_changes, dtype=float)
        if pnl.size == 0:
            raise ValueError("pnl_changes must be non-empty")
        var, cvar = InterestRateVaR._var_cvar(pnl, confidence)
        return {
            "var": var,
            "cvar": cvar,
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std()),
            "confidence": float(confidence),
        }
