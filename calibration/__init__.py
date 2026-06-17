"""Calibration of short rate models to historical and market data.

This package provides estimators that fit the models in :mod:`models` to data:

- :class:`~calibration.mle.MLECalibrator` — maximum likelihood (Vasicek, CIR).
- :class:`~calibration.ols.OLSCalibrator` — least squares (Vasicek, CIR).
- :class:`~calibration.market_calibration.MarketCalibrator` — Hull-White fit to
  market cap / bond quotes.
- :class:`~calibration.kalman_filter.KalmanFilterEstimator` — Kalman filter
  estimation of the Vasicek latent short rate from a yield panel.
"""

from .kalman_filter import KalmanFilterEstimator
from .market_calibration import MarketCalibrator
from .mle import MLECalibrator
from .ols import OLSCalibrator

__all__ = [
    "MLECalibrator",
    "OLSCalibrator",
    "MarketCalibrator",
    "KalmanFilterEstimator",
]
