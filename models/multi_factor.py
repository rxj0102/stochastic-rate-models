"""Two-factor Gaussian short rate model (G2++ / Hull-White two-factor).

The G2++ model writes the short rate as the sum of two correlated mean-reverting
Gaussian factors plus a deterministic shift that fits the initial curve:

.. math::

    r(t) = x(t) + y(t) + \\varphi(t)
    \\\\
    dx(t) = -a\\,x(t)\\,dt + \\sigma\\,dW_1(t)
    \\\\
    dy(t) = -b\\,y(t)\\,dt + \\eta\\,dW_2(t),
    \\quad dW_1\\,dW_2 = \\rho\\,dt

It is equivalent to the two-factor Hull-White model and produces richer term
structure shapes (including humps and de-correlated curve movements) than any
one-factor model.

References
----------
Brigo, D., Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*,
2nd ed., Springer. Chapter 4 (G2++).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from .base import MIN_TIME, ModelParams, ShortRateModel, SimResult, _timed_simulation


@dataclass
class G2Params(ModelParams):
    """Parameters of the G2++ model.

    Parameters
    ----------
    a, b : float
        Mean reversion speeds of the two factors (``> 0``).
    sigma, eta : float
        Volatilities of the two factors (``> 0``).
    rho : float
        Instantaneous correlation between the factor Brownian motions.
    """

    a: float
    b: float
    sigma: float
    eta: float
    rho: float

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("a and b must be positive")
        if self.sigma <= 0 or self.eta <= 0:
            raise ValueError("sigma and eta must be positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")


class G2PlusPlusModel(ShortRateModel):
    """Two-factor additive Gaussian (G2++) short rate model."""

    def __init__(self, params: G2Params) -> None:
        self.params = params
        self._yield_spline: Optional[CubicSpline] = None
        self._max_maturity = 30.0

    # ----------------------------------------------- initial curve setup
    def set_initial_curve(
        self, market_maturities: np.ndarray, market_yields: np.ndarray
    ) -> "G2PlusPlusModel":
        """Fit the deterministic shift ``phi(t)`` to the market curve."""
        mats = np.asarray(market_maturities, dtype=float)
        ylds = np.asarray(market_yields, dtype=float)
        order = np.argsort(mats)
        mats, ylds = mats[order], ylds[order]
        if mats[0] > MIN_TIME:
            mats = np.insert(mats, 0, 0.0)
            ylds = np.insert(ylds, 0, ylds[0])
        self._yield_spline = CubicSpline(mats, ylds, extrapolate=True)
        self._max_maturity = float(mats[-1])
        return self

    def market_zcb(self, T: float) -> float:
        """Market discount factor ``P_mkt(0, T)``."""
        if self._yield_spline is None:
            raise RuntimeError("Initial curve not set. Call set_initial_curve().")
        T = max(float(T), 0.0)
        if T <= MIN_TIME:
            return 1.0
        return float(np.exp(-self._yield_spline(T) * T))

    # ---------------------------------------------------- variance term
    def _V(self, t: float, T: float) -> float:
        """Variance term ``V(t, T)`` of the integrated short rate (Brigo-Mercurio)."""
        a, b = self.params.a, self.params.b
        s, eta, rho = self.params.sigma, self.params.eta, self.params.rho
        tau = T - t
        term_x = (s**2 / a**2) * (
            tau + (2.0 / a) * np.exp(-a * tau) - (1.0 / (2.0 * a)) * np.exp(-2.0 * a * tau) - 1.5 / a
        )
        term_y = (eta**2 / b**2) * (
            tau + (2.0 / b) * np.exp(-b * tau) - (1.0 / (2.0 * b)) * np.exp(-2.0 * b * tau) - 1.5 / b
        )
        term_xy = (2.0 * rho * s * eta / (a * b)) * (
            tau
            + (np.exp(-a * tau) - 1.0) / a
            + (np.exp(-b * tau) - 1.0) / b
            - (np.exp(-(a + b) * tau) - 1.0) / (a + b)
        )
        return term_x + term_y + term_xy

    def _Bx(self, t: float, T: float) -> float:
        a = self.params.a
        return (1.0 - np.exp(-a * (T - t))) / a

    def _By(self, t: float, T: float) -> float:
        b = self.params.b
        return (1.0 - np.exp(-b * (T - t))) / b

    # ------------------------------------------------------------------ SDE
    def drift(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        # Not a one-factor drift; provided for interface completeness using the
        # marginal mean reversion of the combined factors is ill-defined, so we
        # return zero (the factor-level drifts are used in ``simulate``).
        return np.zeros_like(np.asarray(r, dtype=float))

    def diffusion(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        total_var = self.params.sigma**2 + self.params.eta**2 + 2.0 * self.params.rho * self.params.sigma * self.params.eta
        return np.sqrt(total_var) * np.ones_like(np.asarray(r, dtype=float))

    # ------------------------------------------------------- ZCB pricing
    def zcb_price_xy(self, x: float, y: float, t: float, T: float) -> float:
        """ZCB price ``P(t, T)`` given factor values ``x(t), y(t)``."""
        if T - t <= MIN_TIME:
            return 1.0
        Pt = self.market_zcb(t)
        PT = self.market_zcb(T)
        A = (PT / Pt) * np.exp(
            0.5 * (self._V(t, T) - self._V(0.0, T) + self._V(0.0, t))
        )
        return float(A * np.exp(-self._Bx(t, T) * x - self._By(t, T) * y))

    def zcb_price(self, r0: float, T: float, t: float = 0.0) -> float:
        """ZCB price assuming the factors are at their initial value ``x=y=0``.

        At ``t = 0`` this returns ``P_mkt(0, T)`` exactly.
        """
        if t <= MIN_TIME:
            return self.market_zcb(T)
        return self.zcb_price_xy(0.0, 0.0, t, T)

    def yield_curve(
        self, x0: float, y0: float, maturities: np.ndarray
    ) -> np.ndarray:  # type: ignore[override]
        """Analytical spot yields given initial factor values ``x0, y0``."""
        maturities = np.asarray(maturities, dtype=float)
        yields = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            T = max(float(T), MIN_TIME)
            P = self.zcb_price_xy(x0, y0, 0.0, T)
            yields[i] = -np.log(P) / T
        return yields

    # -------------------------------------------------------- simulation
    @_timed_simulation
    def simulate_factors(
        self,
        x0: float,
        y0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the two factors and the reconstructed short rate.

        Returns
        -------
        tuple
            ``(r_paths, x_paths, y_paths, t_grid)``.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        a, b = self.params.a, self.params.b
        s, eta, rho = self.params.sigma, self.params.eta, self.params.rho
        sqrt_dt = np.sqrt(dt)

        x = np.empty((n_paths, n_steps + 1))
        y = np.empty((n_paths, n_steps + 1))
        x[:, 0] = x0
        y[:, 0] = y0

        z1 = rng.standard_normal((n_paths, n_steps))
        z2 = rng.standard_normal((n_paths, n_steps))
        w2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2  # correlated increments
        for i in range(n_steps):
            x[:, i + 1] = x[:, i] - a * x[:, i] * dt + s * sqrt_dt * z1[:, i]
            y[:, i + 1] = y[:, i] - b * y[:, i] * dt + eta * sqrt_dt * w2[:, i]

        # phi(t) = f_mkt(0,t) + variance correction; approximate via forward.
        phi = np.array([self._phi(float(t)) for t in t_grid])
        r = x + y + phi[None, :]
        return r, x, y, t_grid

    def _phi(self, t: float) -> float:
        """Deterministic shift reproducing the initial forward curve."""
        if self._yield_spline is None:
            return 0.0
        t = max(t, MIN_TIME)
        R = float(self._yield_spline(t))
        dR = float(self._yield_spline(t, 1))
        f = R + t * dR
        a, b = self.params.a, self.params.b
        s, eta, rho = self.params.sigma, self.params.eta, self.params.rho
        corr = (
            (s**2 / (2.0 * a**2)) * (1.0 - np.exp(-a * t)) ** 2
            + (eta**2 / (2.0 * b**2)) * (1.0 - np.exp(-b * t)) ** 2
            + rho * s * eta / (a * b) * (1.0 - np.exp(-a * t)) * (1.0 - np.exp(-b * t))
        )
        return f + corr

    def simulate(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate the short rate (factors started at zero)."""
        r, x, y, t_grid = self.simulate_factors(0.0, 0.0, T, n_paths, n_steps, seed)
        return SimResult(paths=r, t_grid=t_grid)

    # ---------------------------------------------------------- swaptions
    def swaption_price(
        self,
        swap_rate: float,
        tenor: float,
        T_option: float,
        notional: float = 1.0,
        freq: int = 2,
        payer: bool = True,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: Optional[int] = 12345,
    ) -> float:
        """Monte Carlo swaption price under G2++.

        Simulates the factors to ``T_option`` and values the underlying swap
        analytically at each node.
        """
        r, x, y, t_grid = self.simulate_factors(
            0.0, 0.0, T_option, n_paths, n_steps, seed
        )
        x_T, y_T = x[:, -1], y[:, -1]
        n = int(round(tenor * freq))
        delta = 1.0 / freq
        pay_times = [T_option + (i + 1) * delta for i in range(n)]

        # Discount factors from T_option to each payment date for every path.
        annuity = np.zeros(n_paths)
        float_leg = np.zeros(n_paths)
        P_last = np.ones(n_paths)
        for j, t in enumerate(pay_times):
            Pj = np.array(
                [self.zcb_price_xy(xi, yi, T_option, t) for xi, yi in zip(x_T, y_T)]
            )
            annuity += delta * Pj
            if j == len(pay_times) - 1:
                P_last = Pj
        swap_value = (1.0 - P_last) - swap_rate * annuity  # payer swap value
        if not payer:
            swap_value = -swap_value
        payoff = np.maximum(swap_value, 0.0)
        # Discount payoff back to today.
        disc = self.market_zcb(T_option)
        return float(notional * disc * payoff.mean())

    # -------------------------------------------------------- metadata
    @property
    def name(self) -> str:
        return "G2++"

    @property
    def has_analytical_zcb(self) -> bool:
        return True
