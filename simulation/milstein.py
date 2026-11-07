"""Milstein Monte Carlo simulator for short rate models.

The Milstein scheme augments the Euler-Maruyama update with a second-order term
that accounts for the variation of the diffusion coefficient with the state:

.. math:: r_{i+1} = r_i + a\\,\\Delta t + b\\,\\sqrt{\\Delta t}\\,Z_i
          + \\tfrac{1}{2} b\\,b'\\,(\\Delta t\\,Z_i^2 - \\Delta t)

where ``a = mu(r, t)``, ``b = sigma(r, t)`` and ``b' = db/dr``. The extra term
raises the strong order of convergence from one half (Euler) to one.

Closed forms used here
----------------------
* Vasicek: ``b = sigma`` is constant so ``b' = 0`` and Milstein reduces to
  Euler.
* CIR: ``b = sigma sqrt(r)`` so ``b' = sigma / (2 sqrt(r))`` and the correction
  simplifies to ``(sigma^2 / 4)(Z^2 - 1) dt`` (the ``sqrt(r)`` cancels), which
  is well defined even at ``r = 0``.

Generic models fall back to a finite-difference estimate of ``db/dr``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models import CIRModel, ShortRateModel, SimResult, VasicekModel
from models.base import _timed_simulation

#: Models whose state must be floored at zero each step (square-root diffusion).
_FLOORED_MODEL_NAMES: frozenset[str] = frozenset({"CIR"})

#: Relative step used for the finite-difference estimate of ``db/dr``.
_FD_REL_STEP: float = 1e-6
#: Absolute floor for the finite-difference step to avoid a zero increment.
_FD_ABS_STEP: float = 1e-8


class MilsteinSimulator:
    """Second-order Milstein path simulator.

    Notes
    -----
    For state-independent diffusions (e.g. Vasicek) the Milstein correction is
    zero and the scheme is identical to Euler-Maruyama. For the CIR model the
    correction term is computed in closed form, avoiding the singularity of
    ``db/dr`` at ``r = 0``.
    """

    @staticmethod
    def _needs_floor(model: ShortRateModel) -> bool:
        """Return whether the model state must be floored at zero each step."""
        return isinstance(model, CIRModel) or model.name in _FLOORED_MODEL_NAMES

    @staticmethod
    def _db_dr(model: ShortRateModel, r: np.ndarray, t: float) -> np.ndarray:
        """Finite-difference estimate of ``d sigma(r, t) / dr`` (generic fallback)."""
        h = np.maximum(np.abs(r) * _FD_REL_STEP, _FD_ABS_STEP)
        b_up = np.asarray(model.diffusion(r + h, t), dtype=float)
        b_dn = np.asarray(model.diffusion(r - h, t), dtype=float)
        return (b_up - b_dn) / (2.0 * h)

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
        """Simulate short rate paths with the Milstein scheme.

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
        The CIR correction ``0.25 * sigma^2 * (Z^2 - 1) * dt`` is used in closed
        form. Vasicek uses no correction (``db/dr = 0``). Other models use a
        central finite-difference estimate of ``db/dr``.
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = model._time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        floor = self._needs_floor(model)

        is_vasicek = isinstance(model, VasicekModel)
        is_cir = isinstance(model, CIRModel)
        cir_corr = None
        if is_cir:
            # 0.5 * b * b' * (dt Z^2 - dt) with b b' = sigma^2 / 2.
            cir_corr = 0.25 * model.params.sigma**2 * dt

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            if floor:
                r = np.maximum(r, 0.0)
            t = t_grid[i]
            zi = z[:, i]
            a = model.drift(r, t)
            b = model.diffusion(r, t)
            r_next = r + a * dt + b * sqrt_dt * zi
            if is_cir:
                r_next = r_next + cir_corr * (zi**2 - 1.0)
            elif not is_vasicek:
                db = self._db_dr(model, r, t)
                r_next = r_next + 0.5 * b * db * (dt * zi**2 - dt)
            # Vasicek: db/dr = 0, no correction.
            if floor:
                r_next = np.maximum(r_next, 0.0)
            paths[:, i + 1] = r_next
        return SimResult(paths=paths, t_grid=t_grid)
