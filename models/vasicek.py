"""Vasicek (1977) one-factor short rate model.

The Vasicek model assumes the instantaneous short rate follows an
Ornstein-Uhlenbeck process

.. math:: dr(t) = \\kappa(\\theta - r(t))\\,dt + \\sigma\\,dW(t)

which is mean reverting with constant volatility. The model admits a closed
form Gaussian transition density, an affine term structure, and analytical
bond option prices (Jamshidian, 1989). Its principal weakness is that rates can
become negative with positive probability.

References
----------
Vasicek, O. (1977). "An equilibrium characterization of the term structure."
*Journal of Financial Economics*, 5(2), 177-188.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from .base import (
    MIN_KAPPA,
    MIN_TIME,
    ModelParams,
    ShortRateModel,
    SimResult,
    _timed_simulation,
)


@dataclass
class VasicekParams(ModelParams):
    """Structural parameters of the Vasicek model.

    Parameters
    ----------
    kappa : float
        Mean reversion speed (``> 0``).
    theta : float
        Long-run mean level of the short rate.
    sigma : float
        Instantaneous volatility (``> 0``).
    """

    kappa: float
    theta: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


class VasicekModel(ShortRateModel):
    """Vasicek mean-reverting Gaussian short rate model.

    Parameters
    ----------
    params : VasicekParams
        Model parameters.

    Examples
    --------
    >>> model = VasicekModel(VasicekParams(kappa=0.5, theta=0.05, sigma=0.02))
    >>> round(model.zcb_price(0.03, 1.0), 4)
    0.9679
    """

    def __init__(self, params: VasicekParams) -> None:
        self.params = params

    # ------------------------------------------------------------------ SDE
    def drift(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        return self.params.kappa * (self.params.theta - r)

    def diffusion(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        # Constant volatility: broadcast to the shape of ``r``.
        return self.params.sigma * np.ones_like(np.asarray(r, dtype=float))

    # ------------------------------------------------ conditional moments
    def conditional_mean(self, r_s: float, dt: float) -> float:
        """Mean of ``r(t) | r(s)`` for ``t - s = dt`` (exact)."""
        k, th = self.params.kappa, self.params.theta
        e = np.exp(-k * dt)
        return r_s * e + th * (1.0 - e)

    def conditional_variance(self, dt: float) -> float:
        """Variance of ``r(t) | r(s)`` for ``t - s = dt`` (exact)."""
        k, s = self.params.kappa, self.params.sigma
        return (s**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * dt))

    # ----------------------------------------------------- affine A and B
    def _B(self, tau: float) -> float:
        k = self.params.kappa
        return (1.0 - np.exp(-k * tau)) / k

    def _lnA(self, tau: float) -> float:
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma
        B = self._B(tau)
        return (th - s**2 / (2.0 * k**2)) * (B - tau) - s**2 * B**2 / (4.0 * k)

    # ------------------------------------------------------- ZCB pricing
    def zcb_price(self, r0: float, T: float, t: float = 0.0) -> float:
        """Analytical zero coupon bond price :math:`P(t, T) = A e^{-B r}`."""
        tau = max(T - t, 0.0)
        if tau <= MIN_TIME:
            return 1.0
        return float(np.exp(self._lnA(tau) - self._B(tau) * r0))

    def yield_curve(self, r0: float, maturities: np.ndarray) -> np.ndarray:
        """Continuously compounded spot yields :math:`R(0, T) = -\\ln P / T`."""
        maturities = np.asarray(maturities, dtype=float)
        yields = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            T = max(float(T), MIN_TIME)
            yields[i] = -np.log(self.zcb_price(r0, T)) / T
        return yields

    def long_run_yield(self) -> float:
        """Limiting yield :math:`\\theta - \\sigma^2 / (2\\kappa^2)` as ``T -> inf``."""
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma
        return th - s**2 / (2.0 * k**2)

    # --------------------------------------------------------- forwards
    def forward_curve(
        self, r0: float, maturities: np.ndarray, dt: float = 1.0 / 252
    ) -> np.ndarray:
        """Instantaneous forward rate ``f(0, T)`` in closed form.

        For the Vasicek model
        ``f(0, T) = r0 e^{-kT} + theta(1 - e^{-kT}) - (sigma^2/2k^2)(1 - e^{-kT})^2``.
        """
        maturities = np.asarray(maturities, dtype=float)
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma
        e = np.exp(-k * maturities)
        return r0 * e + th * (1.0 - e) - (s**2 / (2.0 * k**2)) * (1.0 - e) ** 2

    # ----------------------------------------------------- bond options
    def call_on_zcb(
        self,
        r0: float,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """Price a European call on a zero coupon bond (Jamshidian, 1989).

        Parameters
        ----------
        r0 : float
            Current short rate.
        T_option : float
            Option expiry.
        T_bond : float
            Maturity of the underlying bond (``> T_option``).
        K : float
            Strike per unit face.
        face : float, optional
            Face value of the underlying bond.

        Returns
        -------
        float
            Present value of the call option.
        """
        k, s = self.params.kappa, self.params.sigma
        P_T = self.zcb_price(r0, T_option)
        P_S = self.zcb_price(r0, T_bond)
        B = self._B(T_bond - T_option)
        sigma_P = s * B * np.sqrt((1.0 - np.exp(-2.0 * k * T_option)) / (2.0 * k))
        if sigma_P < 1e-14:
            return max(face * P_S - K * P_T, 0.0)
        d1 = (np.log(face * P_S / (K * P_T)) / sigma_P) + 0.5 * sigma_P
        d2 = d1 - sigma_P
        return float(face * P_S * stats.norm.cdf(d1) - K * P_T * stats.norm.cdf(d2))

    def put_on_zcb(
        self,
        r0: float,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """Price a European put on a ZCB via put-call parity."""
        call = self.call_on_zcb(r0, T_option, T_bond, K, face)
        P_T = self.zcb_price(r0, T_option)
        P_S = self.zcb_price(r0, T_bond)
        return float(call - face * P_S + K * P_T)

    # ------------------------------------------------------- diagnostics
    def negative_rate_probability(self, r0: float, t: float) -> float:
        """Probability ``P(r(t) < 0)`` — a known flaw of the Vasicek model."""
        mu = self.conditional_mean(r0, t)
        var = self.conditional_variance(t)
        if var <= 0:
            return 0.0
        return float(stats.norm.cdf(-mu / np.sqrt(var)))

    def term_premium(self, r0: float, T: float) -> float:
        """Yield in excess of the average expected short rate over ``[0, T]``.

        The expected average short rate is
        ``(1/T) integral_0^T E[r(u)] du``. The term premium is the analytical
        yield minus this quantity.
        """
        k, th = self.params.kappa, self.params.theta
        T = max(T, MIN_TIME)
        expected_avg = th + (r0 - th) * (1.0 - np.exp(-k * T)) / (k * T)
        yld = -np.log(self.zcb_price(r0, T)) / T
        return float(yld - expected_avg)

    def stationary_distribution(self) -> tuple[float, float]:
        """Return ``(mean, variance)`` of the stationary distribution of ``r``."""
        th = self.params.theta
        var = self.params.sigma**2 / (2.0 * self.params.kappa)
        return float(th), float(var)

    # -------------------------------------------------------- simulation
    @_timed_simulation
    def simulate_exact(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate using the exact Gaussian transition (no discretisation error).

        Each step draws ``r_{i+1} = mu(r_i, dt) + std * Z`` where the conditional
        mean and variance are computed in closed form.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        k, th = self.params.kappa, self.params.theta
        e = np.exp(-k * dt)
        std = np.sqrt(self.conditional_variance(dt))

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * e + th * (1.0 - e) + std * z[:, i]
        return SimResult(paths=paths, t_grid=t_grid)

    @_timed_simulation
    def simulate_euler(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate using the Euler-Maruyama discretisation scheme."""
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            paths[:, i + 1] = r + k * (th - r) * dt + s * sqrt_dt * z[:, i]
        return SimResult(paths=paths, t_grid=t_grid)

    def simulate(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Default simulation uses the exact Gaussian transition."""
        return self.simulate_exact(r0, T, n_paths, n_steps, seed)

    # -------------------------------------------------------- metadata
    @property
    def name(self) -> str:
        return "Vasicek"

    @property
    def has_analytical_zcb(self) -> bool:
        return True
