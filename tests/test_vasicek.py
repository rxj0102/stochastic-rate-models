"""Tests for the Vasicek model: moments, term structure, calibration round-trip."""

import numpy as np
import pytest

from calibration import MLECalibrator, OLSCalibrator
from models import VasicekModel, VasicekParams


def test_conditional_mean_from_simulation(vasicek_model):
    """E[r(T)] from simulation matches the analytic conditional mean."""
    r0, T = 0.03, 3.0
    sim = vasicek_model.simulate_exact(r0, T, 60_000, 60, seed=1)
    analytic = vasicek_model.conditional_mean(r0, T)
    stderr = sim.terminal.std(ddof=1) / np.sqrt(sim.n_paths)
    assert abs(sim.terminal.mean() - analytic) < 4 * stderr


def test_conditional_variance_from_simulation(vasicek_model):
    """Var[r(T)] from simulation matches sigma^2/(2k)(1-e^{-2kT})."""
    r0, T = 0.03, 3.0
    sim = vasicek_model.simulate_exact(r0, T, 80_000, 60, seed=2)
    analytic = vasicek_model.conditional_variance(T)
    assert abs(sim.terminal.var() - analytic) / analytic < 0.05


def test_zcb_at_zero_is_one(vasicek_model):
    assert vasicek_model.zcb_price(0.03, 0.0) == pytest.approx(1.0)


def test_zcb_long_maturity_goes_to_zero(vasicek_model):
    assert vasicek_model.zcb_price(0.03, 100.0) < 1e-2


def test_long_run_yield(vasicek_model):
    """R(0,T) approaches theta - sigma^2/(2k^2) for very long maturities."""
    long_yield = vasicek_model.yield_curve(0.03, np.array([250.0]))[0]
    assert long_yield == pytest.approx(vasicek_model.long_run_yield(), abs=1e-3)


def test_negative_rate_probability_positive(vasicek_model):
    """A known Vasicek flaw: positive probability of negative rates."""
    assert vasicek_model.negative_rate_probability(0.01, 5.0) > 0.0


def test_ols_calibration_roundtrip():
    """Simulate from known params and recover them by OLS.

    A long sample is used because the long-run mean ``theta`` is only weakly
    identified over short horizons (its estimator is the sample average, whose
    sampling variance shrinks slowly with the span of the series).
    """
    true = VasicekParams(kappa=0.5, theta=0.05, sigma=0.02)
    series = VasicekModel(true).simulate_exact(0.04, 150.0, 1, 37_800, seed=11).paths[0]
    rec = OLSCalibrator().calibrate_vasicek(series, dt=150.0 / 37_800)["params"]
    assert rec.kappa == pytest.approx(true.kappa, rel=0.20)
    assert rec.theta == pytest.approx(true.theta, rel=0.10)
    assert rec.sigma == pytest.approx(true.sigma, rel=0.05)


def test_mle_calibration_roundtrip():
    """MLE recovers the parameters and returns finite information criteria."""
    true = VasicekParams(kappa=0.5, theta=0.05, sigma=0.02)
    series = VasicekModel(true).simulate_exact(0.04, 150.0, 1, 37_800, seed=11).paths[0]
    res = MLECalibrator().calibrate_vasicek(series, dt=150.0 / 37_800)
    rec = res["params"]
    assert rec.theta == pytest.approx(true.theta, rel=0.10)
    assert rec.sigma == pytest.approx(true.sigma, rel=0.05)
    assert np.isfinite(res["aic"]) and np.isfinite(res["bic"])
