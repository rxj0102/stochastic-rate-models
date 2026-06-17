"""Tests for the Hull-White model: exact curve fit, MC ZCB, cap-floor parity."""

import numpy as np

from pricing import CapsFloorsPricer


def test_fitted_curve_matches_market(hw_model, market_curve):
    """Hull-White reproduces the input yield curve to sub-basis-point accuracy."""
    mats, ylds = market_curve
    fitted = hw_model.yield_curve_fitted(mats)
    rmse_bps = np.sqrt(np.mean((fitted - ylds) ** 2)) * 1e4
    assert rmse_bps < 0.1


def test_mc_zcb_matches_analytical(hw_model):
    """Monte Carlo discounting reproduces the market discount factor."""
    r0 = hw_model.r0
    for T in (1.0, 3.0, 7.0):
        # Use a fine grid: the Euler scheme has an O(dt) bias in the discount
        # factor that grows with maturity, so resolve it with ~250 steps/year.
        sim = hw_model.simulate(r0, T, 40_000, int(T * 250), seed=21)
        disc = np.exp(-np.trapezoid(sim.paths, sim.t_grid, axis=1))
        mc, stderr = disc.mean(), disc.std(ddof=1) / np.sqrt(sim.n_paths)
        assert abs(mc - hw_model.market_zcb(T)) < 4 * stderr


def test_cap_floor_parity(hw_model):
    """Cap - Floor = Swap to within a basis point of notional."""
    res = CapsFloorsPricer().cap_floor_parity_check(hw_model, 0.03, 5.0, 1_000_000, 4)
    assert res["passes"]
    assert abs(res["difference"]) < 1e-4 * 1_000_000


def test_negative_rates_possible(hw_model):
    """Hull-White is Gaussian, so negative short rates can occur."""
    # Use a high-volatility, low-mean-reversion variant to make it likely.
    from models import HullWhiteModel, HullWhiteParams

    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    low = np.full_like(mats, 0.005, dtype=float)
    vol_hw = HullWhiteModel(HullWhiteParams(0.05, 0.02)).set_initial_curve(mats, low)
    sim = vol_hw.simulate(vol_hw.r0, 10.0, 20_000, 120, seed=22)
    assert sim.paths.min() < 0.0
