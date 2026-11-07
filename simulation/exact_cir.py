"""Exact (bias-free) simulator for the CIR model.

The Cox-Ingersoll-Ross short rate has an exact transition law that is a scaled
non-central chi-squared distribution:

.. math:: r(t + \\Delta t) = c \\cdot Y,\\quad Y \\sim \\chi'^2(d, \\lambda)

with scale ``c = sigma^2 (1 - e^{-k dt}) / (4k)``, degrees of freedom
``d = 4 k theta / sigma^2`` and non-centrality
``lambda = r(t) e^{-k dt} / c``. Sampling directly from this law (via
:meth:`numpy.random.Generator.noncentral_chisquare`) produces unbiased,
strictly non-negative paths with zero discretisation error.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models import CIRModel, SimResult
from models.base import _timed_simulation

#: Numerical guard to keep the scale factor strictly positive.
_MIN_SCALE: float = 1e-300


class ExactCIRSimulator:
    """Bias-free CIR path simulator using the non-central chi-squared transition.

    Notes
    -----
    The transition parameters ``(scale, df)`` are constant across steps for a
    uniform grid; only the non-centrality ``ncp`` depends on the current state
    and is therefore vectorised over paths. All draws are non-negative by
    construction, so no truncation is needed.
    """

    @_timed_simulation
    def simulate(
        self,
        model: CIRModel,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate CIR paths with the exact non-central chi-squared transition.

        Parameters
        ----------
        model : CIRModel
            CIR model providing ``kappa``, ``theta`` and ``sigma``.
        r0 : float
            Initial short rate at ``t = 0`` (``>= 0``).
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
            ``elapsed_seconds`` field populated by the timing decorator. All
            path values are non-negative.

        Notes
        -----
        Draws use ``rng.noncentral_chisquare(df, ncp_array)`` where ``ncp_array``
        is computed per path from the previous step, then rescaled by ``scale``.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = model._time_grid(T, n_steps)
        k, th, s = model.params.kappa, model.params.theta, model.params.sigma

        scale = max(s**2 * (1.0 - np.exp(-k * dt)) / (4.0 * k), _MIN_SCALE)
        df = 4.0 * k * th / s**2
        decay = np.exp(-k * dt)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        for i in range(n_steps):
            ncp = np.maximum(paths[:, i], 0.0) * decay / scale
            draw = rng.noncentral_chisquare(df, np.maximum(ncp, 0.0))
            paths[:, i + 1] = draw * scale
        return SimResult(paths=paths, t_grid=t_grid)
