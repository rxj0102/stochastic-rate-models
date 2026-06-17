"""Tests for calibration: OLS/MLE round-trips and Hull-White market fit."""

import numpy as np
import pytest

from calibration import MarketCalibrator, MLECalibrator, OLSCalibrator
from models import VasicekModel, VasicekParams


def _vasicek_series(seed=11):
    true = VasicekParams(kappa=0.5, theta=0.05, sigma=0.02)
    # Long span so the long-run mean theta is statistically identifiable.
    return VasicekModel(true).simulate_exact(0.04, 150.0, 1, 37_800, seed=seed).paths[0], true


def test_ols_recovers_params():
    series, true = _vasicek_series()
    rec = OLSCalibrator().calibrate_vasicek(series, dt=150.0 / 37_800)["params"]
    assert rec.theta == pytest.approx(true.theta, rel=0.10)
    assert rec.sigma == pytest.approx(true.sigma, rel=0.10)


def test_mle_finite_and_accurate():
    series, true = _vasicek_series()
    res = MLECalibrator().calibrate_vasicek(series, dt=150.0 / 37_800)
    assert np.isfinite(res["log_likelihood"])
    assert np.isfinite(res["aic"]) and np.isfinite(res["bic"])
    # Volatility is the best-identified parameter; MLE should nail it.
    assert res["params"].sigma == pytest.approx(true.sigma, rel=0.10)


def test_hw_market_calibration_reprices_curve():
    """HW calibrated to cap prices still reprices the input curve exactly."""
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    ylds = np.array([0.020, 0.022, 0.025, 0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.038])
    from models import HullWhiteModel, HullWhiteParams

    truth = HullWhiteModel(HullWhiteParams(0.1, 0.012)).set_initial_curve(mats, ylds)
    target_mats = [1, 2, 3, 5, 7, 10]
    cap_prices = [truth.cap_price(0.03, T, 1.0, 4) for T in target_mats]

    res = MarketCalibrator().calibrate_hw_to_caps(
        target_mats, cap_prices, mats, ylds, cap_rate=0.03, notional=1.0, freq=4
    )
    params = res["params"]
    fitted = HullWhiteModel(params).set_initial_curve(mats, ylds).yield_curve_fitted(mats)
    rmse_bps = np.sqrt(np.mean((fitted - ylds) ** 2)) * 1e4
    assert rmse_bps < 1.0  # curve fit is exact regardless of (kappa, sigma)
