"""Forward rate curve construction utilities.

Provides :class:`ForwardCurveBuilder` for extracting both instantaneous and
simple (LIBOR-style) forward rates from a calibrated short rate model.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Default finite-difference bump (one trading day) for instantaneous forwards.
DEFAULT_DT: float = 1.0 / 252.0

#: Lower bound on maturities so a centred difference never crosses ``T = 0``.
MIN_TIME: float = 1e-12


class ForwardCurveBuilder:
    """Build forward curves from a short rate model.

    The builder relies only on the model's ``zcb_price`` method, so it works
    uniformly for every one-factor model in the library.
    """

    # ----------------------------------------------------- instantaneous fwd
    @staticmethod
    def instantaneous_forward(
        model,
        r0: float,
        maturities: np.ndarray,
        dt: float = DEFAULT_DT,
    ) -> np.ndarray:
        """Instantaneous forward rates ``f(0, T)`` by central differencing.

        Parameters
        ----------
        model : ShortRateModel
            Model exposing ``zcb_price(r0, T)``.
        r0 : float
            Initial short rate.
        maturities : np.ndarray
            Maturities in years.
        dt : float, optional
            Finite-difference half-step. Defaults to one trading day.

        Returns
        -------
        np.ndarray
            ``f(0, T) = -(ln P(0, T+dt) - ln P(0, T-dt)) / (2 dt)``.

        Notes
        -----
        The lower evaluation point is floored at ``MIN_TIME`` so the centred
        difference remains well defined as ``T -> dt``.
        """
        maturities = np.asarray(maturities, dtype=float)
        fwd = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            T = max(float(T), MIN_TIME)
            t_up = T + dt
            t_dn = max(T - dt, MIN_TIME)
            lp_up = np.log(model.zcb_price(r0, t_up))
            lp_dn = np.log(model.zcb_price(r0, t_dn))
            fwd[i] = -(lp_up - lp_dn) / (t_up - t_dn)
        return fwd

    # ------------------------------------------------------------- libor fwd
    @staticmethod
    def libor_forward(
        model,
        r0: float,
        T_start_list: np.ndarray,
        tenor: float,
    ) -> np.ndarray:
        """Simple (LIBOR) forward rates over a fixed tenor.

        Parameters
        ----------
        model : ShortRateModel
            Model exposing ``zcb_price(r0, T)``.
        r0 : float
            Initial short rate.
        T_start_list : np.ndarray
            Forward-rate start dates ``T1`` in years.
        tenor : float
            Accrual length so that ``T2 = T1 + tenor``.

        Returns
        -------
        np.ndarray
            ``L(T1, T2) = (P(0, T1) / P(0, T2) - 1) / tenor``.
        """
        starts = np.asarray(T_start_list, dtype=float)
        out = np.empty_like(starts)
        for i, t1 in enumerate(starts):
            t2 = float(t1) + tenor
            p1 = model.zcb_price(r0, max(float(t1), MIN_TIME))
            p2 = model.zcb_price(r0, t2)
            out[i] = ForwardCurveBuilder.discrete_forward(p1, p2, tenor)
        return out

    # --------------------------------------------------------------- helpers
    @staticmethod
    def discrete_forward(P1: float, P2: float, tau: float) -> float:
        """Simple forward rate between two discount factors.

        Parameters
        ----------
        P1 : float
            Discount factor to the start of the accrual period, ``P(0, T1)``.
        P2 : float
            Discount factor to the end of the accrual period, ``P(0, T2)``.
        tau : float
            Year fraction of the accrual period.

        Returns
        -------
        float
            ``(P1 / P2 - 1) / tau``.
        """
        tau = max(float(tau), MIN_TIME)
        return float((P1 / P2 - 1.0) / tau)
