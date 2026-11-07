"""Abstract base class and shared data structures for short rate models.

This module defines the common interface implemented by every short rate model
in the library (Vasicek, CIR, Hull-White, G2++). Working against the
:class:`ShortRateModel` interface allows the pricing, simulation, risk, and
scenario engines to remain model agnostic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
DAYS_PER_YEAR: int = 252
TRADING_DAYS: int = 252
FELLER_THRESHOLD: float = 2.0

#: Numerical guards used throughout the library to avoid division by zero.
MIN_KAPPA: float = 1e-6
MIN_SIGMA: float = 1e-6
MIN_TIME: float = 1e-6


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass
class ModelParams:
    """Base class for model parameter storage.

    Concrete models subclass this dataclass to hold their structural
    parameters (mean reversion speed, volatility, ...). Keeping parameters in a
    dedicated dataclass makes calibration round-trips and serialisation
    straightforward.
    """

    def as_dict(self) -> dict:
        """Return the parameters as a plain dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    """Container for the output of a Monte Carlo simulation.

    Attributes
    ----------
    paths : np.ndarray
        Simulated short rate paths with shape ``(n_paths, n_steps + 1)``.
    t_grid : np.ndarray
        Time grid with shape ``(n_steps + 1,)`` in years.
    mean_path : np.ndarray
        Cross-sectional mean of ``paths`` at each time, shape ``(n_steps + 1,)``.
    std_path : np.ndarray
        Cross-sectional standard deviation at each time.
    elapsed_seconds : float
        Wall-clock time taken to generate the simulation.
    """

    paths: np.ndarray
    t_grid: np.ndarray
    mean_path: np.ndarray = field(default=None)  # type: ignore[assignment]
    std_path: np.ndarray = field(default=None)  # type: ignore[assignment]
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.mean_path is None:
            self.mean_path = self.paths.mean(axis=0)
        if self.std_path is None:
            self.std_path = self.paths.std(axis=0)

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.paths.shape[1] - 1)

    @property
    def terminal(self) -> np.ndarray:
        """Short rate values at the simulation horizon, shape ``(n_paths,)``."""
        return self.paths[:, -1]

    def percentile_paths(self, percentiles: tuple[float, ...]) -> dict[float, np.ndarray]:
        """Return cross-sectional percentile paths.

        Parameters
        ----------
        percentiles : tuple of float
            Percentiles in ``[0, 100]``.

        Returns
        -------
        dict
            Mapping from percentile to a path array of shape ``(n_steps + 1,)``.
        """
        return {p: np.percentile(self.paths, p, axis=0) for p in percentiles}


@dataclass
class BondPrice:
    """Container for a bond valuation and its first order risk metrics."""

    price: float
    yield_: float
    duration: float
    convexity: float
    model_name: str
    modified_duration: Optional[float] = None
    dv01: Optional[float] = None


# ---------------------------------------------------------------------------
# Abstract base model
# ---------------------------------------------------------------------------
class ShortRateModel(ABC):
    """Abstract base class for one-factor short rate models.

    A short rate model specifies the risk-neutral dynamics of the instantaneous
    short rate ``r(t)`` through a stochastic differential equation

    .. math:: dr(t) = \\mu(r, t)\\,dt + \\sigma(r, t)\\,dW(t)

    Concrete subclasses must implement the drift, diffusion, an (optional)
    analytical zero coupon bond price, the analytical yield curve, and a Monte
    Carlo simulation routine.
    """

    # -- core SDE coefficients ------------------------------------------------
    @abstractmethod
    def drift(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        """Instantaneous drift :math:`\\mu(r, t)` of the short rate SDE."""

    @abstractmethod
    def diffusion(self, r: np.ndarray | float, t: float) -> np.ndarray | float:
        """Instantaneous diffusion :math:`\\sigma(r, t)` of the short rate SDE."""

    # -- analytical pricing ---------------------------------------------------
    @abstractmethod
    def zcb_price(self, r0: float, T: float, t: float = 0.0) -> float:
        """Analytical zero coupon bond price :math:`P(t, T)` for one unit face."""

    @abstractmethod
    def yield_curve(self, r0: float, maturities: np.ndarray) -> np.ndarray:
        """Analytical continuously compounded spot yields :math:`R(0, T)`."""

    # -- simulation -----------------------------------------------------------
    @abstractmethod
    def simulate(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate short rate paths and return a :class:`SimResult`."""

    # -- metadata -------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable model name."""

    @property
    @abstractmethod
    def has_analytical_zcb(self) -> bool:
        """Whether the model exposes a closed-form ZCB price."""

    # -- shared helpers -------------------------------------------------------
    def discount_factors(self, r0: float, maturities: np.ndarray) -> np.ndarray:
        """Vectorised analytical discount factors :math:`P(0, T)`."""
        maturities = np.asarray(maturities, dtype=float)
        return np.array([self.zcb_price(r0, float(T)) for T in maturities])

    def forward_curve(
        self, r0: float, maturities: np.ndarray, dt: float = 1.0 / DAYS_PER_YEAR
    ) -> np.ndarray:
        """Instantaneous forward rates :math:`f(0, T) = -\\partial_T \\ln P(0, T)`.

        Computed via a centred finite difference of the log discount curve.
        """
        maturities = np.asarray(maturities, dtype=float)
        fwd = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            T = max(float(T), MIN_TIME)
            lp_up = np.log(self.zcb_price(r0, T + dt))
            lp_dn = np.log(self.zcb_price(r0, max(T - dt, MIN_TIME)))
            fwd[i] = -(lp_up - lp_dn) / (2.0 * dt)
        return fwd

    def half_life(self) -> float:
        """Half-life of mean reversion ``ln(2) / kappa`` (if ``kappa`` exists)."""
        kappa = getattr(self.params, "kappa", None)  # type: ignore[attr-defined]
        if kappa is None or kappa <= MIN_KAPPA:
            return float("inf")
        return float(np.log(2.0) / kappa)

    @staticmethod
    def _time_grid(T: float, n_steps: int) -> tuple[np.ndarray, float]:
        """Return ``(t_grid, dt)`` for a uniform partition of ``[0, T]``."""
        T = max(float(T), MIN_TIME)
        dt = T / n_steps
        t_grid = np.linspace(0.0, T, n_steps + 1)
        return t_grid, dt

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}({getattr(self, 'params', '')})"


def _timed_simulation(func):
    """Decorator that records wall-clock time into a returned :class:`SimResult`."""

    def wrapper(*args, **kwargs) -> SimResult:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        result.elapsed_seconds = time.perf_counter() - start
        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
