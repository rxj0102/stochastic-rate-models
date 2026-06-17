"""Monte Carlo convergence diagnostics for short rate models.

Two distinct sources of Monte Carlo error are isolated here:

* **Statistical error** — the ``O(1/sqrt(N))`` standard error from using a
  finite number of paths ``N``. Studied with :meth:`MCConvergence.zcb_convergence`.
* **Discretisation (bias) error** — the systematic bias of the Euler-Maruyama
  scheme that vanishes as the number of time steps grows. Studied with
  :meth:`MCConvergence.step_convergence`.

Both routines benchmark against the analytical zero coupon bond price exposed by
the model.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models import ShortRateModel
from simulation import EulerMaruyamaSimulator

#: Numerical guard.
_EPS: float = 1e-15


def _mc_zcb_from_paths(paths: np.ndarray, t_grid: np.ndarray) -> tuple[float, float]:
    """Monte Carlo ZCB price and its standard error from simulated paths.

    The discount factor on each path is ``exp(-integral_0^T r(u) du)`` with the
    integral evaluated by the trapezoidal rule.

    Parameters
    ----------
    paths : np.ndarray
        Short rate paths of shape ``(n_paths, n_steps + 1)``.
    t_grid : np.ndarray
        Time grid of shape ``(n_steps + 1,)``.

    Returns
    -------
    tuple of float
        ``(price, stderr)`` where ``price`` is the mean discount factor and
        ``stderr`` is its standard error.
    """
    integral = np.trapezoid(paths, x=t_grid, axis=1)
    disc = np.exp(-integral)
    n = disc.shape[0]
    price = float(disc.mean())
    stderr = float(disc.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return price, stderr


class MCConvergence:
    """Convergence studies for Monte Carlo bond pricing."""

    def __init__(self) -> None:
        self._euler = EulerMaruyamaSimulator()

    # -------------------------------------------------- path-count study
    def zcb_convergence(
        self,
        model: ShortRateModel,
        r0: float,
        T: float,
        n_paths_list: Sequence[int] = (1_000, 5_000, 10_000, 50_000, 100_000),
        n_steps: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Statistical convergence of the MC ZCB price as ``n_paths`` grows.

        Parameters
        ----------
        model : ShortRateModel
            Model with analytical ``zcb_price`` and a ``simulate`` method.
        r0 : float
            Initial short rate.
        T : float
            Bond maturity in years.
        n_paths_list : sequence of int, optional
            Path counts to evaluate.
        n_steps : int, optional
            Number of time steps (held fixed across the study).
        seed : int, optional
            Random seed (the same seed is used for every path count so that the
            error decay is monotone up to sampling noise).

        Returns
        -------
        pandas.DataFrame
            Columns ``n_paths``, ``mc_price``, ``stderr``, ``analytical``,
            ``abs_error`` and ``error_in_stderr`` (the absolute error expressed in
            units of the standard error). The absolute error should decay roughly
            like ``1/sqrt(n_paths)``.
        """
        analytical = float(model.zcb_price(r0, T))
        rows = []
        for n_paths in n_paths_list:
            sim = model.simulate(r0, T, int(n_paths), n_steps, seed)
            price, stderr = _mc_zcb_from_paths(sim.paths, sim.t_grid)
            abs_error = abs(price - analytical)
            rows.append(
                {
                    "n_paths": int(n_paths),
                    "mc_price": price,
                    "stderr": stderr,
                    "analytical": analytical,
                    "abs_error": abs_error,
                    "error_in_stderr": abs_error / stderr
                    if stderr and stderr > _EPS
                    else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    # -------------------------------------------------- time-step study
    def step_convergence(
        self,
        model: ShortRateModel,
        r0: float,
        T: float,
        n_steps_list: Sequence[int],
        n_paths: int = 50_000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Discretisation-bias convergence of the Euler scheme as steps grow.

        Parameters
        ----------
        model : ShortRateModel
            Model providing ``drift``/``diffusion`` (for the Euler simulator) and
            an analytical ``zcb_price``.
        r0 : float
            Initial short rate.
        T : float
            Bond maturity in years.
        n_steps_list : sequence of int
            Step counts to evaluate.
        n_paths : int, optional
            Number of paths (held fixed so that only the time-step bias varies).
        seed : int, optional
            Random seed.

        Returns
        -------
        pandas.DataFrame
            Columns ``n_steps``, ``dt``, ``mc_price``, ``stderr``,
            ``analytical``, ``bias`` (signed error ``mc_price - analytical``) and
            ``abs_bias``. The magnitude of the bias should fall as ``n_steps``
            increases.
        """
        analytical = float(model.zcb_price(r0, T))
        rows = []
        for n_steps in n_steps_list:
            sim = self._euler.simulate(model, r0, T, int(n_paths), int(n_steps), seed)
            price, stderr = _mc_zcb_from_paths(sim.paths, sim.t_grid)
            bias = price - analytical
            rows.append(
                {
                    "n_steps": int(n_steps),
                    "dt": T / int(n_steps),
                    "mc_price": price,
                    "stderr": stderr,
                    "analytical": analytical,
                    "bias": bias,
                    "abs_bias": abs(bias),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------- plot helper
    def plot_data(
        self, df: pd.DataFrame, x: str, y: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``(x, y)`` arrays from a convergence table for charting.

        Parameters
        ----------
        df : pandas.DataFrame
            Output of :meth:`zcb_convergence` or :meth:`step_convergence`.
        x, y : str
            Column names to extract.

        Returns
        -------
        tuple of np.ndarray
            ``(x_values, y_values)`` suitable for a log-log convergence plot.
        """
        return df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float)
