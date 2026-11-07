"""Exact (bias-free) simulator for the Vasicek model.

The Vasicek short rate is an Ornstein-Uhlenbeck process whose transition law is
Gaussian and known in closed form:

.. math:: r(t+\\Delta t) \\mid r(t) \\sim
          N\\!\\big(m(r(t), \\Delta t),\\, v(\\Delta t)\\big)

with conditional mean ``m`` and variance ``v`` supplied by the model. Sampling
directly from this law removes all discretisation bias regardless of the time
step size, which is the key advantage demonstrated by
:meth:`ExactVasicekSimulator.convergence_vs_euler`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models import SimResult, VasicekModel
from models.base import _timed_simulation


class ExactVasicekSimulator:
    """Bias-free Vasicek path simulator using the exact Gaussian transition.

    Notes
    -----
    Each step draws ``r_{i+1} = m(r_i, dt) + sqrt(v(dt)) * Z`` where ``m`` and
    ``v`` are the closed-form conditional mean and variance. Because the
    transition law is exact, the simulated terminal distribution is unbiased for
    any ``n_steps`` (including ``n_steps = 1``).
    """

    @_timed_simulation
    def simulate(
        self,
        model: VasicekModel,
        r0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SimResult:
        """Simulate Vasicek paths with the exact Gaussian transition law.

        Parameters
        ----------
        model : VasicekModel
            Vasicek model providing ``conditional_mean`` / ``conditional_variance``.
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
        """
        rng = np.random.default_rng(seed)
        t_grid, dt = model._time_grid(T, n_steps)
        k, th = model.params.kappa, model.params.theta
        decay = np.exp(-k * dt)
        std = np.sqrt(model.conditional_variance(dt))

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = r0
        z = rng.standard_normal((n_paths, n_steps))
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * decay + th * (1.0 - decay) + std * z[:, i]
        return SimResult(paths=paths, t_grid=t_grid)

    def convergence_vs_euler(
        self,
        model: VasicekModel,
        r0: float,
        T: float,
        n_steps_list: Sequence[int],
        n_paths: int,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compare exact vs Euler terminal-moment error against the analytic law.

        For each ``n_steps`` the terminal mean and variance produced by the exact
        and Euler-Maruyama schemes are compared to the analytic conditional mean
        ``m(r0, T)`` and variance ``v(T)`` of ``r(T) | r(0)``.

        Parameters
        ----------
        model : VasicekModel
            Vasicek model under study.
        r0 : float
            Initial short rate.
        T : float
            Simulation horizon in years.
        n_steps_list : sequence of int
            Time-step counts to evaluate.
        n_paths : int
            Number of Monte Carlo paths used at each ``n_steps``.
        seed : int, optional
            Base seed; reused across both schemes at each ``n_steps`` so the
            comparison is controlled.

        Returns
        -------
        pandas.DataFrame
            One row per ``n_steps`` with columns ``n_steps``,
            ``analytic_mean``, ``analytic_var``,
            ``exact_mean_err``, ``euler_mean_err``,
            ``exact_var_err`` and ``euler_var_err`` (absolute errors).

        Notes
        -----
        The exact-scheme errors reflect only Monte Carlo sampling noise (and are
        independent of ``n_steps``), whereas the Euler errors additionally carry
        a discretisation bias that grows as ``n_steps`` decreases, demonstrating
        the superiority of exact sampling.
        """
        # Import here to avoid a hard import-time dependency cycle.
        from .euler_maruyama import EulerMaruyamaSimulator

        euler = EulerMaruyamaSimulator()
        analytic_mean = float(model.conditional_mean(r0, T))
        analytic_var = float(model.conditional_variance(T))

        rows = []
        for n_steps in n_steps_list:
            exact_res = self.simulate(model, r0, T, n_paths, n_steps, seed=seed)
            euler_res = euler.simulate(model, r0, T, n_paths, n_steps, seed=seed)

            ex_term, eu_term = exact_res.terminal, euler_res.terminal
            rows.append(
                {
                    "n_steps": int(n_steps),
                    "analytic_mean": analytic_mean,
                    "analytic_var": analytic_var,
                    "exact_mean_err": abs(float(ex_term.mean()) - analytic_mean),
                    "euler_mean_err": abs(float(eu_term.mean()) - analytic_mean),
                    "exact_var_err": abs(float(ex_term.var(ddof=1)) - analytic_var),
                    "euler_var_err": abs(float(eu_term.var(ddof=1)) - analytic_var),
                }
            )
        return pd.DataFrame(rows)
