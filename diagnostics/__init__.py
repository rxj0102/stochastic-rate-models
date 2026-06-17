"""Diagnostics for short rate models.

This package collects empirical and Monte Carlo diagnostics used to validate
fitted short rate models:

* :class:`RateStationarityTests` — unit-root / stationarity and mean-reversion
  tests plus theoretical-vs-empirical variance checks.
* :class:`MCConvergence` — Monte Carlo statistical and discretisation-bias
  convergence studies for bond pricing.
* :class:`ModelComparison` — cross-model yield-curve fit, bond pricing and
  parameter sensitivity.
* :class:`ResidualDiagnostics` — one-step residual normality and serial
  correlation tests.
"""

from .convergence import MCConvergence
from .model_comparison import ModelComparison
from .residuals import ResidualDiagnostics
from .stationarity import RateStationarityTests

__all__ = [
    "RateStationarityTests",
    "MCConvergence",
    "ModelComparison",
    "ResidualDiagnostics",
]
