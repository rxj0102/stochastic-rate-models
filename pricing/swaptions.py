"""European swaption pricing.

:class:`SwaptionPricer` delegates analytical payer/receiver swaption pricing to
the Hull-White model, implements Black implied-volatility inversion and a Monte
Carlo swaption valuation usable with any short rate model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from models import HullWhiteModel, ShortRateModel
from utils.statistics import integrated_rate

#: Default swap leg frequency (semi-annual).
DEFAULT_FREQ: int = 2
#: Brackets for the Black implied-vol root search.
VOL_LOW: float = 1.0e-6
VOL_HIGH: float = 5.0
#: Numerical guards.
EPS: float = 1.0e-14
MIN_TIME: float = 1.0e-12


class SwaptionPricer:
    """Price European swaptions analytically, by Black's model and by MC."""

    def payer_swaption_hw(
        self,
        hw_model: HullWhiteModel,
        swap_rate: float,
        tenor: float,
        T_option: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Price a payer swaption (delegates to Hull-White)."""
        return float(
            hw_model.swaption_price(
                swap_rate, tenor, T_option, notional, freq, payer=True
            )
        )

    def receiver_swaption_hw(
        self,
        hw_model: HullWhiteModel,
        swap_rate: float,
        tenor: float,
        T_option: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
    ) -> float:
        """Price a receiver swaption (delegates to Hull-White)."""
        return float(
            hw_model.swaption_price(
                swap_rate, tenor, T_option, notional, freq, payer=False
            )
        )

    @staticmethod
    def _black_payer(
        forward: float,
        strike: float,
        annuity: float,
        vol: float,
        T_option: float,
        notional: float,
    ) -> float:
        """Black-76 payer swaption price."""
        T_option = max(T_option, MIN_TIME)
        if vol <= 0 or T_option <= 0:
            return float(notional * annuity * max(forward - strike, 0.0))
        sqrt_t = np.sqrt(T_option)
        d1 = (np.log(forward / strike) + 0.5 * vol**2 * T_option) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        return float(
            notional * annuity * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
        )

    def swaption_vol_implied(
        self,
        market_price: float,
        forward_swap_rate: float,
        annuity: float,
        T_option: float,
        notional: float = 1.0,
        strike: Optional[float] = None,
    ) -> float:
        """Black implied volatility of a payer swaption via root finding.

        Parameters
        ----------
        market_price : float
            Observed swaption price.
        forward_swap_rate : float
            Forward par swap rate ``F``.
        annuity : float
            Present value of a basis point (sum of ``delta * P(0, t_i)``).
        T_option : float
            Option expiry.
        notional : float, optional
            Swap notional.
        strike : float, optional
            Strike rate; defaults to the forward (ATM).

        Returns
        -------
        float
            The Black volatility reproducing ``market_price``.
        """
        K = forward_swap_rate if strike is None else strike

        def objective(vol: float) -> float:
            return (
                self._black_payer(
                    forward_swap_rate, K, annuity, vol, T_option, notional
                )
                - market_price
            )

        return float(brentq(objective, VOL_LOW, VOL_HIGH, xtol=1.0e-12))

    def mc_swaption(
        self,
        model: ShortRateModel,
        r0: float,
        swap_rate: float,
        tenor: float,
        T_option: float,
        notional: float = 1.0,
        freq: int = DEFAULT_FREQ,
        payer: bool = True,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: Optional[int] = None,
    ) -> tuple[float, float]:
        """Monte Carlo European swaption price.

        Simulates the short rate to ``T_option``, values the underlying swap at
        each path node using ``model.zcb_price(r(T_option), t_i, T_option)`` for
        the fixed-leg payment dates, applies the swaption payoff and discounts
        by the realised money-market account over ``[0, T_option]``.

        The underlying swap is receive-float pay-fixed for a payer swaption.

        Returns
        -------
        tuple
            ``(price, standard_error)``.
        """
        n = int(round(tenor * freq))
        delta = 1.0 / freq
        pay_times = np.array([T_option + (i + 1) * delta for i in range(n)])

        sim = model.simulate(r0, float(T_option), n_paths, n_steps, seed)
        r_T = sim.terminal
        disc = np.exp(-integrated_rate(sim.paths, sim.t_grid))

        # Per-path discount factors P(r_T, t_i, T_option) for each payment date.
        # Build a (n_paths, n) matrix of forward ZCB prices seen at T_option.
        n_paths_eff = len(r_T)
        pvbp = np.zeros(n_paths_eff)
        float_minus_fixed = np.zeros(n_paths_eff)
        # P(T_option, T_option) = 1 by definition.
        for j, r in enumerate(r_T):
            r = float(r)
            zcbs = np.array(
                [model.zcb_price(r, float(t), float(T_option)) for t in pay_times]
            )
            annuity = delta * np.sum(zcbs)
            # Forward swap value (receive-float pay-fixed), per unit notional,
            # as seen at T_option: float leg = 1 - P(T_option, T_N).
            swap_val = (1.0 - zcbs[-1]) - swap_rate * annuity
            float_minus_fixed[j] = swap_val
            pvbp[j] = annuity

        sign = 1.0 if payer else -1.0
        payoff = notional * np.maximum(sign * float_minus_fixed, 0.0)
        discounted = disc * payoff
        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
        return price, stderr
