"""European options on zero coupon and coupon bonds.

:class:`BondOptionPricer` provides a model-agnostic dispatch layer over the
heterogeneous bond-option APIs of the individual short rate models, plus a
Jamshidian decomposition for coupon-bond options and a Monte Carlo fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from models import ShortRateModel
from utils.statistics import integrated_rate

#: Model name constants (must match ``model.name``).
VASICEK: str = "Vasicek"
CIR: str = "CIR"
HULL_WHITE: str = "Hull-White"

#: Search bracket for the Jamshidian critical short rate.
R_STAR_LOW: float = -0.5
R_STAR_HIGH: float = 1.0
#: brentq tolerance.
BRENTQ_XTOL: float = 1.0e-12


class BondOptionPricer:
    """Price European options on bonds across all supported models."""

    def european_call_zcb(
        self,
        model: ShortRateModel,
        r0: float,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """European call on a zero coupon bond (analytical, model-dispatched)."""
        name = model.name
        if name == VASICEK:
            return float(model.call_on_zcb(r0, T_option, T_bond, K, face))
        if name == CIR:
            return float(model.bond_option_cir(r0, T_option, T_bond, K, face))
        if name == HULL_WHITE:
            return float(model.call_on_zcb(T_option, T_bond, K, face))
        raise NotImplementedError(
            f"european_call_zcb not supported for model '{name}'"
        )

    def european_put_zcb(
        self,
        model: ShortRateModel,
        r0: float,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """European put on a zero coupon bond (analytical, model-dispatched).

        Vasicek and Hull-White expose ``put_on_zcb`` directly; CIR is handled
        via put-call parity ``put = call - face*P(0,T_bond) + K*P(0,T_option)``.
        """
        name = model.name
        if name == VASICEK:
            return float(model.put_on_zcb(r0, T_option, T_bond, K, face))
        if name == HULL_WHITE:
            return float(model.put_on_zcb(T_option, T_bond, K, face))
        if name == CIR:
            call = self.european_call_zcb(model, r0, T_option, T_bond, K, face)
            P_bond = model.zcb_price(r0, T_bond)
            P_opt = model.zcb_price(r0, T_option)
            return float(call - face * P_bond + K * P_opt)
        raise NotImplementedError(
            f"european_put_zcb not supported for model '{name}'"
        )

    def european_call_coupon_bond(
        self,
        model: ShortRateModel,
        r0: float,
        coupon_schedule: list[tuple[float, float]],
        face: float,
        T_option: float,
        K: float,
    ) -> float:
        """European call on a coupon bond via the Jamshidian decomposition.

        Parameters
        ----------
        coupon_schedule : list of (time, cashflow)
            Cashflow times (all ``> T_option``) and amounts.
        face : float
            Face value (informational; cashflows are explicit).
        T_option : float
            Option expiry.
        K : float
            Strike price of the coupon bond.

        Notes
        -----
        For Hull-White the model's native ``call_on_coupon_bond`` is used. For
        the other models the critical short rate ``r*`` at ``T_option`` such
        that the bond reprices to ``K`` is found by root finding, after which
        the coupon-bond call decomposes into a sum of ZCB calls struck at the
        re-priced ZCBs ``P(r*, T_option, t_i)``.
        """
        if model.name == HULL_WHITE:
            return float(model.call_on_coupon_bond(coupon_schedule, face, T_option, K))

        times = np.array([t for t, _ in coupon_schedule], dtype=float)
        cfs = np.array([c for _, c in coupon_schedule], dtype=float)

        def bond_value(r_star: float) -> float:
            return float(
                np.sum(
                    [c * model.zcb_price(r_star, float(t), T_option)
                     for c, t in zip(cfs, times)]
                )
            )

        r_star = brentq(
            lambda r: bond_value(r) - K, R_STAR_LOW, R_STAR_HIGH, xtol=BRENTQ_XTOL
        )

        total = 0.0
        for c, t in zip(cfs, times):
            K_i = model.zcb_price(r_star, float(t), T_option)
            total += c * self.european_call_zcb(
                model, r0, T_option, float(t), K_i, face=1.0
            )
        return float(total)

    def mc_bond_option(
        self,
        model: ShortRateModel,
        r0: float,
        T_option: float,
        T_bond: float,
        K: float,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: Optional[int] = None,
        option: str = "call",
    ) -> tuple[float, float]:
        """Monte Carlo price of a European option on a ZCB.

        Simulates the short rate to ``T_option``, computes the underlying ZCB
        price ``P(r(T_option), T_option, T_bond)`` per path, applies the option
        payoff and discounts each path by ``exp(-integral_0^{T_option} r du)``.

        Returns
        -------
        tuple
            ``(price, standard_error)``.
        """
        option = option.lower()
        if option not in ("call", "put"):
            raise ValueError("option must be 'call' or 'put'")

        sim = model.simulate(r0, float(T_option), n_paths, n_steps, seed)
        r_T = sim.terminal
        disc = np.exp(-integrated_rate(sim.paths, sim.t_grid))

        # Underlying ZCB price at expiry given the simulated short rate; the
        # zcb_price interface is scalar in r0, so vectorise over unique rates is
        # not generally safe, hence map per path.
        underlying = np.array(
            [model.zcb_price(float(r), float(T_bond), float(T_option)) for r in r_T]
        )

        if option == "call":
            payoff = np.maximum(underlying - K, 0.0)
        else:
            payoff = np.maximum(K - underlying, 0.0)

        discounted = disc * payoff
        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
        return price, stderr
