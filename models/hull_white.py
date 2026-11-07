"""Hull-White (1990) one-factor extended Vasicek model.

The Hull-White model augments Vasicek with a time-dependent drift that is
calibrated to reproduce the initial term structure exactly:

.. math:: dr(t) = (\\theta(t) - \\kappa r(t))\\,dt + \\sigma\\,dW(t)

The function :math:`\\theta(t)` is recovered from the market instantaneous
forward curve. Because of this construction the model prices today's bonds
without error, which makes it the workhorse for pricing interest rate
derivatives (caps, floors, swaptions).

References
----------
Hull, J., White, A. (1990). "Pricing interest-rate-derivative securities."
*Review of Financial Studies*, 3(4), 573-592.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline

from .base import (
    MIN_TIME,
    ModelParams,
    ShortRateModel,
    SimResult,
    _timed_simulation,
)


@dataclass
class HullWhiteParams(ModelParams):
    """Structural parameters of the Hull-White model.

    Parameters
    ----------
    kappa : float
        Mean reversion speed (``> 0``). Sometimes denoted ``a``.
    sigma : float
        Constant volatility (``> 0``).

    Notes
    -----
    ``theta(t)`` is *not* a free parameter; it is derived from the initial
    market curve via :meth:`HullWhiteModel.set_initial_curve`.
    """

    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


class HullWhiteModel(ShortRateModel):
    """Hull-White one-factor model fitted to an initial yield curve."""

    def __init__(self, params: HullWhiteParams) -> None:
        self.params = params
        self._fwd_spline: Optional[CubicSpline] = None
        self._yield_spline: Optional[CubicSpline] = None
        self._r0: Optional[float] = None

    # ----------------------------------------------- initial curve setup
    def set_initial_curve(
        self,
        market_maturities: np.ndarray,
        market_yields: np.ndarray,
    ) -> "HullWhiteModel":
        """Fit a spline to the market curve and derive ``theta(t)``.

        Parameters
        ----------
        market_maturities : np.ndarray
            Maturities (years) of the observed continuously compounded yields.
        market_yields : np.ndarray
            Observed continuously compounded zero rates.

        Returns
        -------
        HullWhiteModel
            ``self`` (for chaining).
        """
        mats = np.asarray(market_maturities, dtype=float)
        ylds = np.asarray(market_yields, dtype=float)
        order = np.argsort(mats)
        mats, ylds = mats[order], ylds[order]

        # Prepend a near-zero maturity anchored at the short end so that the
        # forward curve is well defined at T = 0.
        if mats[0] > MIN_TIME:
            mats = np.insert(mats, 0, 0.0)
            ylds = np.insert(ylds, 0, ylds[0])

        self._yield_spline = CubicSpline(mats, ylds, extrapolate=True)
        self._max_maturity = float(mats[-1])
        self._r0 = float(ylds[0]) if mats[0] <= MIN_TIME else float(self._yield_spline(0.0))

        # Instantaneous forward f(0,T) = R(T) + T R'(T).
        def fwd(T: np.ndarray) -> np.ndarray:
            T = np.atleast_1d(np.asarray(T, dtype=float))
            R = self._yield_spline(T)
            dR = self._yield_spline(T, 1)
            return R + T * dR

        # Tabulate the forward curve on a fine grid and spline it for reuse.
        grid = np.linspace(0.0, self._max_maturity, 400)
        self._fwd_spline = CubicSpline(grid, fwd(grid), extrapolate=True)
        return self

    def _require_curve(self) -> None:
        if self._fwd_spline is None:
            raise RuntimeError(
                "Initial curve not set. Call set_initial_curve() first."
            )

    def market_forward(self, T: float | np.ndarray) -> np.ndarray | float:
        """Market instantaneous forward rate ``f(0, T)``."""
        self._require_curve()
        return self._fwd_spline(np.asarray(T, dtype=float))  # type: ignore[misc]

    def market_zcb(self, T: float) -> float:
        """Market discount factor ``P_mkt(0, T) = exp(-R(T) T)``."""
        self._require_curve()
        T = max(float(T), 0.0)
        if T <= MIN_TIME:
            return 1.0
        return float(np.exp(-self._yield_spline(T) * T))  # type: ignore[operator]

    @property
    def r0(self) -> float:
        """Short rate implied by the initial curve, ``f(0, 0)``."""
        self._require_curve()
        return float(self._r0)  # type: ignore[arg-type]

    # ------------------------------------------------------- theta(t)
    def theta(self, t: float) -> float:
        """Time-dependent drift level reproducing the initial curve.

        ``theta(t) = df/dt + kappa f(0, t) + (sigma^2 / 2kappa)(1 - e^{-2kt})``.
        """
        self._require_curve()
        k, s = self.params.kappa, self.params.sigma
        f = float(self._fwd_spline(t))  # type: ignore[arg-type]
        dfdt = float(self._fwd_spline(t, 1))  # type: ignore[call-arg]
        return dfdt + k * f + (s**2 / (2.0 * k)) * (1.0 - np.exp(-2.0 * k * t))

    # ------------------------------------------------------------------ SDE
    def drift(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        return self.theta(t) - self.params.kappa * r

    def diffusion(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        return self.params.sigma * np.ones_like(np.asarray(r, dtype=float))

    # ----------------------------------------------------- affine B and ZCB
    def _B(self, t: float, T: float) -> float:
        k = self.params.kappa
        return (1.0 - np.exp(-k * (T - t))) / k

    def zcb_price(self, r0: float, T: float, t: float = 0.0) -> float:
        """Analytical ZCB price reproducing the market curve when ``t = 0``.

        For ``t = 0`` this returns ``P_mkt(0, T)`` exactly. For ``t > 0`` it uses
        the standard Hull-White reconstitution formula given the short rate
        ``r0 = r(t)``.
        """
        self._require_curve()
        if T - t <= MIN_TIME:
            return 1.0
        if t <= MIN_TIME:
            return self.market_zcb(T)

        k, s = self.params.kappa, self.params.sigma
        B = self._B(t, T)
        Pt = self.market_zcb(t)
        PT = self.market_zcb(T)
        f0t = float(self._fwd_spline(t))  # type: ignore[arg-type]
        lnA = np.log(PT / Pt) + B * f0t - (s**2 / (4.0 * k)) * B**2 * (
            1.0 - np.exp(-2.0 * k * t)
        )
        return float(np.exp(lnA - B * r0))

    def yield_curve(self, r0: float, maturities: np.ndarray) -> np.ndarray:
        maturities = np.asarray(maturities, dtype=float)
        yields = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            T = max(float(T), MIN_TIME)
            yields[i] = -np.log(self.zcb_price(r0, T)) / T
        return yields

    def yield_curve_fitted(self, maturities: np.ndarray) -> np.ndarray:
        """Model spot yields at ``t = 0`` — match the market input exactly."""
        maturities = np.asarray(maturities, dtype=float)
        return np.array(
            [-np.log(self.market_zcb(float(T))) / max(float(T), MIN_TIME) for T in maturities]
        )

    def forward_curve_fitted(self, maturities: np.ndarray) -> np.ndarray:
        """Model instantaneous forward rates at ``t = 0`` (match market)."""
        maturities = np.asarray(maturities, dtype=float)
        return np.asarray(self.market_forward(maturities))

    # ----------------------------------------------------- bond options
    def call_on_zcb(
        self,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """European call on a ZCB under Hull-White (analytical)."""
        k, s = self.params.kappa, self.params.sigma
        P_T = self.market_zcb(T_option)
        P_S = self.market_zcb(T_bond)
        B = self._B(T_option, T_bond)
        sigma_P = s * B * np.sqrt((1.0 - np.exp(-2.0 * k * T_option)) / (2.0 * k))
        if sigma_P < 1e-14:
            return max(face * P_S - K * P_T, 0.0)
        d1 = np.log(face * P_S / (K * P_T)) / sigma_P + 0.5 * sigma_P
        d2 = d1 - sigma_P
        return float(face * P_S * stats.norm.cdf(d1) - K * P_T * stats.norm.cdf(d2))

    def put_on_zcb(
        self,
        T_option: float,
        T_bond: float,
        K: float,
        face: float = 1.0,
    ) -> float:
        """European put on a ZCB under Hull-White (analytical)."""
        k, s = self.params.kappa, self.params.sigma
        P_T = self.market_zcb(T_option)
        P_S = self.market_zcb(T_bond)
        B = self._B(T_option, T_bond)
        sigma_P = s * B * np.sqrt((1.0 - np.exp(-2.0 * k * T_option)) / (2.0 * k))
        if sigma_P < 1e-14:
            return max(K * P_T - face * P_S, 0.0)
        d1 = np.log(face * P_S / (K * P_T)) / sigma_P + 0.5 * sigma_P
        d2 = d1 - sigma_P
        return float(K * P_T * stats.norm.cdf(-d2) - face * P_S * stats.norm.cdf(-d1))

    def call_on_coupon_bond(
        self,
        coupon_schedule: list[tuple[float, float]],
        face: float,
        T_option: float,
        K: float,
    ) -> float:
        """European call on a coupon bond via the Jamshidian decomposition.

        Parameters
        ----------
        coupon_schedule : list of (time, cashflow)
            Cashflow times (years, all ``> T_option``) and amounts. The final
            entry should include the redemption of ``face``.
        face : float
            Face value (used only for documentation; cashflows are explicit).
        T_option : float
            Option expiry.
        K : float
            Strike price of the coupon bond.

        Returns
        -------
        float
            Present value of the option.
        """
        from scipy.optimize import brentq

        times = np.array([t for t, _ in coupon_schedule])
        cfs = np.array([c for _, c in coupon_schedule])

        # Solve for the critical short rate r* at T_option where the underlying
        # coupon bond value equals the strike.
        def bond_value(r_star: float) -> float:
            return sum(
                c * self.zcb_price(r_star, float(t), T_option)
                for c, t in zip(cfs, times)
            )

        r_star = brentq(lambda r: bond_value(r) - K, -0.5, 1.0, xtol=1e-10)

        # Each cashflow becomes a call on a ZCB with strike K_i = P(r*, t_i).
        total = 0.0
        for c, t in zip(cfs, times):
            K_i = self.zcb_price(r_star, float(t), T_option)
            total += c * self.call_on_zcb(T_option, float(t), K_i, face=1.0)
        return float(total)

    # ----------------------------------------------------- caps and floors
    def caplet_price(
        self,
        K_cap: float,
        T_start: float,
        T_end: float,
        notional: float = 1.0,
    ) -> float:
        """Price a single caplet as a put on a ZCB (Hull-White analytical)."""
        delta = T_end - T_start
        strike = 1.0 / (1.0 + K_cap * delta)
        # A caplet equals (1 + K delta) puts on the T_end ZCB struck at `strike`.
        put = self.put_on_zcb(T_start, T_end, strike, face=1.0)
        return float(notional * (1.0 + K_cap * delta) * put)

    def floorlet_price(
        self,
        K_floor: float,
        T_start: float,
        T_end: float,
        notional: float = 1.0,
    ) -> float:
        """Price a single floorlet as a call on a ZCB (Hull-White analytical)."""
        delta = T_end - T_start
        strike = 1.0 / (1.0 + K_floor * delta)
        call = self.call_on_zcb(T_start, T_end, strike, face=1.0)
        return float(notional * (1.0 + K_floor * delta) * call)

    def cap_price(
        self,
        cap_rate: float,
        maturity: float,
        notional: float = 1.0,
        freq: int = 4,
    ) -> float:
        """Price an interest rate cap as a portfolio of caplets."""
        n = int(round(maturity * freq))
        times = np.array([(i + 1) / freq for i in range(n)])
        total = 0.0
        for i in range(1, len(times)):
            total += self.caplet_price(cap_rate, times[i - 1], times[i], notional)
        return float(total)

    def floor_price(
        self,
        floor_rate: float,
        maturity: float,
        notional: float = 1.0,
        freq: int = 4,
    ) -> float:
        """Price an interest rate floor as a portfolio of floorlets."""
        n = int(round(maturity * freq))
        times = np.array([(i + 1) / freq for i in range(n)])
        total = 0.0
        for i in range(1, len(times)):
            total += self.floorlet_price(floor_rate, times[i - 1], times[i], notional)
        return float(total)

    # ---------------------------------------------------------- swaptions
    def swaption_price(
        self,
        swap_rate: float,
        tenor: float,
        T_option: float,
        notional: float = 1.0,
        freq: int = 2,
        payer: bool = True,
    ) -> float:
        """Price a European swaption via Jamshidian decomposition.

        A payer swaption is a put on the fixed leg (a coupon bond) struck at par;
        a receiver swaption is the corresponding call.
        """
        from scipy.optimize import brentq

        n = int(round(tenor * freq))
        delta = 1.0 / freq
        pay_times = np.array([T_option + (i + 1) * delta for i in range(n)])
        # Fixed-leg cashflows of the underlying coupon bond (strike = par = 1).
        cfs = np.array([swap_rate * delta] * n)
        cfs[-1] += 1.0
        K = 1.0

        def bond_value(r_star: float) -> float:
            return sum(
                c * self.zcb_price(r_star, float(t), T_option)
                for c, t in zip(cfs, pay_times)
            )

        r_star = brentq(lambda r: bond_value(r) - K, -0.5, 1.0, xtol=1e-10)

        total = 0.0
        for c, t in zip(cfs, pay_times):
            K_i = self.zcb_price(r_star, float(t), T_option)
            if payer:
                total += c * self.put_on_zcb(T_option, float(t), K_i)
            else:
                total += c * self.call_on_zcb(T_option, float(t), K_i)
        return float(notional * total)

    # -------------------------------------------------------- simulation
    @_timed_simulation
    def simulate(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Euler scheme with the time-varying Hull-White drift.

        If ``r0`` is ``None`` (passed as ``np.nan``) the short rate implied by
        the initial curve is used.
        """
        self._require_curve()
        rng = np.random.default_rng(seed)
        t_grid, dt = self._time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        k, s = self.params.kappa, self.params.sigma
        if r0 is None or (isinstance(r0, float) and np.isnan(r0)):
            r0 = self.r0

        # Pre-compute theta on the grid (deterministic) for vectorised speed.
        theta_grid = np.array([self.theta(float(t)) for t in t_grid[:-1]])

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            paths[:, i + 1] = r + (theta_grid[i] - k * r) * dt + s * sqrt_dt * z[:, i]
        return SimResult(paths=paths, t_grid=t_grid)

    # ----------------------------------------------------- trinomial tree
    def trinomial_tree(
        self, r0: float, T: float, n_steps: int
    ) -> dict:
        """Build the Hull-White trinomial tree (Hull-White two-stage procedure).

        Returns
        -------
        dict
            Dictionary with the rate lattice ``rates`` (list of arrays per step),
            the displacement ``alpha`` per step and the spacing ``dr`` and ``dt``.

        Notes
        -----
        Branching probabilities for the standard ("normal") geometry are

        ``pu = 1/6 + (eta^2 + eta)/2``, ``pm = 2/3 - eta^2``,
        ``pd = 1/6 + (eta^2 - eta)/2`` with ``eta = j_target - j``.
        """
        self._require_curve()
        k, s = self.params.kappa, self.params.sigma
        dt = max(T, MIN_TIME) / n_steps
        dr = s * np.sqrt(3.0 * dt)
        # Maximum width of the tree (Hull-White j_max heuristic).
        j_max = int(np.ceil(0.184 / (k * dt)))

        # Stage 1: build the tree for the auxiliary process x with x(0)=0.
        rates: list[np.ndarray] = []
        probs: list[list[tuple[float, float, float, int]]] = []
        alpha: list[float] = []

        # Arrow-Debreu prices for state j at current step.
        Q = {0: 1.0}
        for m in range(n_steps + 1):
            js = sorted(Q.keys())
            j_arr = np.array(js)
            x_vals = j_arr * dr

            # Fit alpha_m so that the tree reprices P_mkt(0, (m+1) dt).
            P_target = self.market_zcb((m + 1) * dt)
            sum_q = sum(Q[j] * np.exp(-j * dr * dt) for j in js)
            alpha_m = (np.log(sum_q / P_target)) / dt if sum_q > 0 else 0.0
            alpha.append(alpha_m)
            rates.append(alpha_m + x_vals)

            if m == n_steps:
                probs.append([])
                break

            # Stage 2: roll Arrow-Debreu prices forward.
            Q_next: dict[int, float] = {}
            step_probs: list[tuple[float, float, float, int]] = []
            for j in js:
                eta_mean = -k * j * dt  # M = -a j dt
                if j == j_max:
                    k_target = j - 1  # downward branching
                elif j == -j_max:
                    k_target = j + 1  # upward branching
                else:
                    k_target = j
                eta = eta_mean - (k_target - j)

                if j == j_max:
                    pu = 7.0 / 6.0 + (eta**2 + 3.0 * eta) / 2.0
                    pm = -1.0 / 3.0 - eta**2 - 2.0 * eta
                    pd = 1.0 / 6.0 + (eta**2 + eta) / 2.0
                elif j == -j_max:
                    pu = 1.0 / 6.0 + (eta**2 - eta) / 2.0
                    pm = -1.0 / 3.0 - eta**2 + 2.0 * eta
                    pd = 7.0 / 6.0 + (eta**2 - 3.0 * eta) / 2.0
                else:
                    pu = 1.0 / 6.0 + (eta**2 + eta) / 2.0
                    pm = 2.0 / 3.0 - eta**2
                    pd = 1.0 / 6.0 + (eta**2 - eta) / 2.0

                step_probs.append((pu, pm, pd, k_target))
                disc = np.exp(-(alpha_m + j * dr) * dt)
                for prob, kk in ((pu, k_target + 1), (pm, k_target), (pd, k_target - 1)):
                    Q_next[kk] = Q_next.get(kk, 0.0) + Q[j] * prob * disc
            probs.append(step_probs)
            Q = Q_next

        return {
            "rates": rates,
            "alpha": np.array(alpha),
            "probs": probs,
            "dr": dr,
            "dt": dt,
            "j_max": j_max,
        }

    # -------------------------------------------------------- metadata
    @property
    def name(self) -> str:
        return "Hull-White"

    @property
    def has_analytical_zcb(self) -> bool:
        return True
