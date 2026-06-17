"""Cox-Ingersoll-Ross (1985) square-root short rate model.

The CIR model specifies

.. math:: dr(t) = \\kappa(\\theta - r(t))\\,dt + \\sigma\\sqrt{r(t)}\\,dW(t)

The square-root diffusion guarantees non-negative rates and, when the Feller
condition :math:`2\\kappa\\theta > \\sigma^2` holds, the process stays strictly
positive. The exact transition law is a scaled non-central chi-squared
distribution, which allows unbiased ("exact") simulation.

References
----------
Cox, J., Ingersoll, J., Ross, S. (1985). "A theory of the term structure of
interest rates." *Econometrica*, 53(2), 385-407.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from .base import (
    FELLER_THRESHOLD,
    MIN_TIME,
    ModelParams,
    ShortRateModel,
    SimResult,
    _timed_simulation,
)


@dataclass
class CIRParams(ModelParams):
    """Structural parameters of the CIR model.

    Parameters
    ----------
    kappa : float
        Mean reversion speed (``> 0``).
    theta : float
        Long-run mean level (``> 0``).
    sigma : float
        Volatility of the square-root diffusion (``> 0``).
    """

    kappa: float
    theta: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def feller(self) -> float:
        """Left-hand side of the Feller condition ``2*kappa*theta``."""
        return 2.0 * self.kappa * self.theta


class CIRModel(ShortRateModel):
    """Cox-Ingersoll-Ross square-root diffusion short rate model."""

    def __init__(self, params: CIRParams) -> None:
        self.params = params
        self._h = np.sqrt(params.kappa**2 + 2.0 * params.sigma**2)

    # ------------------------------------------------------------------ SDE
    def drift(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        return self.params.kappa * (self.params.theta - r)

    def diffusion(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        return self.params.sigma * np.sqrt(np.maximum(np.asarray(r, dtype=float), 0.0))

    # ------------------------------------------------------- Feller check
    def feller_condition_check(self) -> tuple[bool, float, float]:
        """Return ``(holds, 2*kappa*theta, sigma^2)``.

        The Feller condition ``2*kappa*theta > sigma^2`` guarantees the process
        remains strictly positive.
        """
        lhs = self.params.feller
        rhs = self.params.sigma**2
        return bool(lhs > rhs), float(lhs), float(rhs)

    # ----------------------------------------------- conditional moments
    def conditional_mean(self, r_s: float, dt: float) -> float:
        """Mean of ``r(t) | r(s)`` for horizon ``dt``."""
        k, th = self.params.kappa, self.params.theta
        e = np.exp(-k * dt)
        return r_s * e + th * (1.0 - e)

    def conditional_variance(self, r_s: float, dt: float) -> float:
        """Variance of ``r(t) | r(s)`` for horizon ``dt``."""
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma
        e = np.exp(-k * dt)
        return r_s * (s**2 / k) * (e - np.exp(-2.0 * k * dt)) + th * (
            s**2 / (2.0 * k)
        ) * (1.0 - e) ** 2

    # ----------------------------------------------------- affine A and B
    def _B(self, tau: float) -> float:
        k, h = self.params.kappa, self._h
        num = 2.0 * (np.exp(h * tau) - 1.0)
        den = (h + k) * (np.exp(h * tau) - 1.0) + 2.0 * h
        return num / den

    def _A(self, tau: float) -> float:
        k, th, s, h = self.params.kappa, self.params.theta, self.params.sigma, self._h
        num = 2.0 * h * np.exp((k + h) * tau / 2.0)
        den = (h + k) * (np.exp(h * tau) - 1.0) + 2.0 * h
        return (num / den) ** (2.0 * k * th / s**2)

    # ------------------------------------------------------- ZCB pricing
    def zcb_price(self, r0: float, T: float, t: float = 0.0) -> float:
        """Analytical zero coupon bond price :math:`P(t, T) = A e^{-B r}`."""
        tau = max(T - t, 0.0)
        if tau <= MIN_TIME:
            return 1.0
        return float(self._A(tau) * np.exp(-self._B(tau) * r0))

    def yield_curve(self, r0: float, maturities: np.ndarray) -> np.ndarray:
        maturities = np.asarray(maturities, dtype=float)
        yields = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            T = max(float(T), MIN_TIME)
            yields[i] = -np.log(self.zcb_price(r0, T)) / T
        return yields

    def long_run_yield(self) -> float:
        """Limiting yield ``2*kappa*theta / (h + kappa)`` as ``T -> inf``."""
        k, th, h = self.params.kappa, self.params.theta, self._h
        return 2.0 * k * th / (h + k)

    # -------------------------------------------------------- densities
    def spot_rate_density(
        self, r0: float, t: float, r_grid: np.ndarray
    ) -> np.ndarray:
        """PDF of ``r(t)`` given ``r(0) = r0`` (scaled non-central chi-squared).

        Parameters
        ----------
        r0 : float
            Initial short rate.
        t : float
            Horizon in years.
        r_grid : np.ndarray
            Points at which to evaluate the density.

        Returns
        -------
        np.ndarray
            Density values aligned with ``r_grid``.
        """
        r_grid = np.asarray(r_grid, dtype=float)
        scale, df, ncp = self._ncx2_params(r0, max(t, MIN_TIME))
        # r(t) = scale * Y with Y ~ ncx2(df, ncp); hence the density of r is
        # (1/scale) * f_Y(r / scale).
        return stats.ncx2.pdf(r_grid / scale, df, ncp) / scale

    def _ncx2_params(self, r_s: float, dt: float) -> tuple[float, float, float]:
        """Return ``(scale, df, ncp)`` of the exact transition law.

        ``r(t) = scale * Y`` where ``Y`` is non-central chi-squared with ``df``
        degrees of freedom and non-centrality ``ncp`` (Glasserman, 2004).
        """
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma
        scale = s**2 * (1.0 - np.exp(-k * dt)) / (4.0 * k)
        df = 4.0 * k * th / s**2
        ncp = r_s * np.exp(-k * dt) / scale
        return scale, df, ncp

    # ----------------------------------------------------- bond options
    def bond_option_cir(
        self,
        r0: float,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """European call on a ZCB under CIR via non-central chi-squared (CIR 1985).

        Returns
        -------
        float
            Present value of the call option struck at ``K`` per unit face.
        """
        k, th, s, h = self.params.kappa, self.params.theta, self.params.sigma, self._h
        P_T = self.zcb_price(r0, T_option)
        P_S = self.zcb_price(r0, T_bond)
        B_TS = self._B(T_bond - T_option)

        rho = 2.0 * h / (s**2 * (np.exp(h * T_option) - 1.0))
        psi = (k + h) / s**2
        # Critical rate r* such that A(T,S) exp(-B(T,S) r*) = K.
        A_TS = self._A(T_bond - T_option)
        if K <= 0 or A_TS <= 0:
            r_star = np.inf
        else:
            r_star = np.log(A_TS / K) / B_TS

        df = 4.0 * k * th / s**2
        # Two non-central chi-squared evaluations.
        x1 = 2.0 * r_star * (rho + psi + B_TS)
        ncp1 = (2.0 * rho**2 * r0 * np.exp(h * T_option)) / (rho + psi + B_TS)
        x2 = 2.0 * r_star * (rho + psi)
        ncp2 = (2.0 * rho**2 * r0 * np.exp(h * T_option)) / (rho + psi)

        term1 = face * P_S * stats.ncx2.cdf(x1, df, ncp1)
        term2 = K * P_T * stats.ncx2.cdf(x2, df, ncp2)
        return float(term1 - term2)

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
        """Exact simulation via the non-central chi-squared transition law.

        Draws are made with :func:`numpy.random.Generator.noncentral_chisquare`
        and rescaled by ``1/c``. This scheme has zero discretisation bias.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma
        scale = s**2 * (1.0 - np.exp(-k * dt)) / (4.0 * k)
        df = 4.0 * k * th / s**2
        decay = np.exp(-k * dt)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        for i in range(n_steps):
            ncp = paths[:, i] * decay / scale
            draw = rng.noncentral_chisquare(df, np.maximum(ncp, 0.0))
            paths[:, i + 1] = draw * scale
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
        """Euler-Maruyama with full truncation (``r = max(r, 0)``)."""
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = np.maximum(paths[:, i], 0.0)
            r_next = r + k * (th - r) * dt + s * np.sqrt(r) * sqrt_dt * z[:, i]
            paths[:, i + 1] = np.maximum(r_next, 0.0)
        return SimResult(paths=paths, t_grid=t_grid)

    @_timed_simulation
    def simulate_milstein(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Milstein scheme with the square-root diffusion correction.

        For ``b(r) = sigma sqrt(r)`` we have ``b b' = sigma^2 / 2`` so the
        Milstein update adds ``(sigma^2 / 4)(dt Z^2 - dt)``.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        k, th, s = self.params.kappa, self.params.theta, self.params.sigma

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = np.maximum(paths[:, i], 0.0)
            zi = z[:, i]
            r_next = (
                r
                + k * (th - r) * dt
                + s * np.sqrt(r) * sqrt_dt * zi
                + 0.25 * s**2 * dt * (zi**2 - 1.0)
            )
            paths[:, i + 1] = np.maximum(r_next, 0.0)
        return SimResult(paths=paths, t_grid=t_grid)

    def simulate(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Default simulation uses the exact chi-squared transition."""
        return self.simulate_exact(r0, T, n_paths, n_steps, seed)

    # -------------------------------------------------------- metadata
    @property
    def name(self) -> str:
        return "CIR"

    @property
    def has_analytical_zcb(self) -> bool:
        return True
