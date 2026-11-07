"""Monte Carlo simulation engines for short rate models.

This package provides discretisation schemes and exact (bias-free) samplers for
the models in :mod:`models`, plus a unified :class:`PathManager` dispatcher.

Classes
-------
EulerMaruyamaSimulator
    First-order Euler-Maruyama scheme (model agnostic).
MilsteinSimulator
    Second-order Milstein scheme with closed-form CIR correction.
ExactVasicekSimulator
    Bias-free Vasicek sampler via the Gaussian transition law.
ExactCIRSimulator
    Bias-free CIR sampler via the non-central chi-squared transition law.
PathManager
    Unified, model-aware interface dispatching to the appropriate scheme.
"""

from __future__ import annotations

from .euler_maruyama import EulerMaruyamaSimulator
from .exact_cir import ExactCIRSimulator
from .exact_vasicek import ExactVasicekSimulator
from .milstein import MilsteinSimulator
from .path_manager import PathManager

__all__ = [
    "EulerMaruyamaSimulator",
    "MilsteinSimulator",
    "ExactVasicekSimulator",
    "ExactCIRSimulator",
    "PathManager",
]
