"""Unified simulation interface dispatching to the appropriate scheme.

:class:`PathManager` wraps a single :class:`~models.base.ShortRateModel` and
routes simulation requests to the best available scheme:

* exact Gaussian transition for Vasicek,
* exact non-central chi-squared transition for CIR,
* Euler-Maruyama (or Milstein) for every other model (Hull-White, G2++),

while exposing a common ``simulate`` signature and a ``compare_schemes`` helper
for benchmarking accuracy and runtime across schemes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from models import CIRModel, ShortRateModel, SimResult, VasicekModel

from .euler_maruyama import EulerMaruyamaSimulator
from .exact_cir import ExactCIRSimulator
from .exact_vasicek import ExactVasicekSimulator
from .milstein import MilsteinSimulator

#: Supported dispatch methods.
_METHODS: frozenset[str] = frozenset({"auto", "exact", "euler", "milstein"})


class PathManager:
    """Unified, model-aware Monte Carlo path generator.

    Parameters
    ----------
    model : ShortRateModel
        The short rate model to simulate. For Hull-White / G2++ the initial
        market curve must have been set (``set_initial_curve``) beforehand.

    Notes
    -----
    The ``'auto'`` method selects the exact scheme for models that have one
    (Vasicek, CIR) and Euler-Maruyama otherwise.
    """

    def __init__(self, model: ShortRateModel) -> None:
        self.model = model
        self._euler = EulerMaruyamaSimulator()
        self._milstein = MilsteinSimulator()
        self._exact_vasicek = ExactVasicekSimulator()
        self._exact_cir = ExactCIRSimulator()

    # ------------------------------------------------------------- helpers
    def _has_exact(self) -> bool:
        """Whether the wrapped model exposes an exact transition simulator."""
        return isinstance(self.model, (VasicekModel, CIRModel))

    def _resolve(self, method: str) -> str:
        """Map ``'auto'`` to a concrete scheme and validate the request."""
        if method not in _METHODS:
            raise ValueError(
                f"Unknown method {method!r}; expected one of {sorted(_METHODS)}."
            )
        if method == "auto":
            return "exact" if self._has_exact() else "euler"
        if method == "exact" and not self._has_exact():
            raise ValueError(
                f"Model {self.model.name!r} has no exact scheme; "
                "use 'euler' or 'milstein'."
            )
        return method

    def _dispatch(
        self,
        scheme: str,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int],
    ) -> SimResult:
        """Run a concrete (already resolved) scheme."""
        if scheme == "exact":
            if isinstance(self.model, VasicekModel):
                return self._exact_vasicek.simulate(self.model, r0, T, n_paths, n_steps, seed)
            return self._exact_cir.simulate(self.model, r0, T, n_paths, n_steps, seed)
        if scheme == "milstein":
            return self._milstein.simulate(self.model, r0, T, n_paths, n_steps, seed)
        return self._euler.simulate(self.model, r0, T, n_paths, n_steps, seed)

    # ------------------------------------------------------------ public API
    def simulate(
        self,
        method: str = "auto",
        *,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate short rate paths using the requested (or auto) scheme.

        Parameters
        ----------
        method : {'auto', 'exact', 'euler', 'milstein'}, optional
            Discretisation scheme. ``'auto'`` (default) picks the exact scheme
            for Vasicek/CIR and Euler-Maruyama for Hull-White/G2++.
        r0 : float
            Initial short rate at ``t = 0``.
        T : float
            Simulation horizon in years.
        n_paths : int
            Number of Monte Carlo paths.
        n_steps : int
            Number of time steps; the grid has ``n_steps + 1`` points.
        seed : int, optional
            Seed for :func:`numpy.random.default_rng`.

        Returns
        -------
        SimResult
            Simulated paths with ``elapsed_seconds`` populated.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``'exact'`` is requested for a model
            without an exact transition law.
        """
        scheme = self._resolve(method)
        return self._dispatch(scheme, r0, T, n_paths, n_steps, seed)

    def available_schemes(self) -> list[str]:
        """Return the concrete schemes applicable to the wrapped model."""
        schemes = ["euler", "milstein"]
        if self._has_exact():
            schemes.insert(0, "exact")
        return schemes

    def compare_schemes(
        self,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Benchmark terminal statistics and runtime across available schemes.

        Parameters
        ----------
        r0 : float
            Initial short rate.
        T : float
            Simulation horizon in years.
        n_paths : int
            Number of Monte Carlo paths.
        n_steps : int
            Number of time steps.
        seed : int, optional
            Seed reused across schemes for a controlled comparison.

        Returns
        -------
        pandas.DataFrame
            One row per available scheme with columns ``scheme``,
            ``terminal_mean``, ``terminal_std``, ``min_rate`` and
            ``elapsed_seconds``.
        """
        rows = []
        for scheme in self.available_schemes():
            res = self._dispatch(scheme, r0, T, n_paths, n_steps, seed)
            term = res.terminal
            rows.append(
                {
                    "scheme": scheme,
                    "terminal_mean": float(term.mean()),
                    "terminal_std": float(term.std(ddof=1)),
                    "min_rate": float(res.paths.min()),
                    "elapsed_seconds": float(res.elapsed_seconds),
                }
            )
        return pd.DataFrame(rows)
