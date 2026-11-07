"""Euler-Maruyama Monte Carlo simulator for short rate models.

The Euler-Maruyama scheme is the first-order strong/weak discretisation of an
Ito stochastic differential equation

.. math:: dr(t) = \\mu(r, t)\\,dt + \\sigma(r, t)\\,dW(t)

approximated on a uniform time grid by

.. math:: r_{i+1} = r_i + \\mu(r_i, t_i)\\,\\Delta t
          + \\sigma(r_i, t_i)\\sqrt{\\Delta t}\\,Z_i,\\quad Z_i \\sim N(0, 1).

It works for any :class:`~models.base.ShortRateModel` because it only relies on
the model's ``drift`` and ``diffusion`` coefficients. For square-root models
such as CIR a full-truncation floor at zero is applied to keep the argument of
the square root non-negative.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models import CIRModel, ShortRateModel, SimResult
from models.base import _timed_simulation

#: Models whose state must be floored at zero each step (square-root diffusion).
_FLOORED_MODEL_NAMES: frozenset[str] = frozenset({"CIR"})


class EulerMaruyamaSimulator:
    """First-order Euler-Maruyama path simulator.

    The simulator is model agnostic: it dispatches to ``model.drift`` and
    ``model.diffusion`` and therefore supports every short rate model in the
    library, including the (non-affine in time) Hull-White and G2++ models.

    Notes
    -----
    The Euler scheme has weak order one and strong order one half. It introduces
    a discretisation bias that vanishes as ``n_steps -> infinity``; for models
    with an exact transition law (Vasicek, CIR) the dedicated exact simulators
    should be preferred when bias matters.
    """

    @staticmethod
    def _needs_floor(model: ShortRateModel) -> bool:
        """Return whether the model state must be floored at zero each step."""
        return isinstance(model, CIRModel) or model.name in _FLOORED_MODEL_NAMES

    @_timed_simulation
    def simulate(
        self,
        model: ShortRateModel,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate short rate paths with the Euler-Maruyama scheme.

        Parameters
        ----------
        model : ShortRateModel
            Model providing ``drift(r, t)`` and ``diffusion(r, t)``.
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
            Simulated paths of shape ``(n_paths, n_steps + 1)`` with the
            ``elapsed_seconds`` field populated by the timing decorator.

        Notes
        -----
        For CIR (square-root diffusion) the state is floored at zero before each
        update (full truncation) so that ``sqrt(r)`` is well defined.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = model._time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        floor = self._needs_floor(model)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            if floor:
                r = np.maximum(r, 0.0)
            t = t_grid[i]
            r_next = r + model.drift(r, t) * dt + model.diffusion(r, t) * sqrt_dt * z[:, i]
            if floor:
                r_next = np.maximum(r_next, 0.0)
            paths[:, i + 1] = r_next
        return SimResult(paths=paths, t_grid=t_grid)
