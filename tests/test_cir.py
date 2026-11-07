"""Tests for the CIR model: positivity, moments, ZCB, exact-vs-Euler variance."""

import numpy as np
import pytest

from models import CIRModel, CIRParams


def test_positivity_under_feller(cir_model):
    """When the Feller condition holds, exact simulation stays non-negative."""
    holds, lhs, rhs = cir_model.feller_condition_check()
    assert holds and lhs > rhs
    sim = cir_model.simulate_exact(0.03, 5.0, 40_000, 60, seed=3)
    assert (sim.paths >= 0).all()


def test_rates_approach_zero_when_feller_violated():
    """With a violated Feller condition, Euler paths can reach (near) zero."""
    model = CIRModel(CIRParams(kappa=0.2, theta=0.02, sigma=0.3))
    holds, _, _ = model.feller_condition_check()
    assert not holds
    sim = model.simulate_euler(0.02, 5.0, 20_000, 200, seed=4)
    assert sim.paths.min() < 1e-6


def test_conditional_mean(cir_model):
    r0, T = 0.03, 4.0
    sim = cir_model.simulate_exact(r0, T, 60_000, 60, seed=5)
    analytic = cir_model.conditional_mean(r0, T)
    stderr = sim.terminal.std(ddof=1) / np.sqrt(sim.n_paths)
    assert abs(sim.terminal.mean() - analytic) < 4 * stderr


def test_conditional_variance(cir_model):
    r0, T = 0.03, 4.0
    sim = cir_model.simulate_exact(r0, T, 80_000, 60, seed=6)
    analytic = cir_model.conditional_variance(r0, T)
    assert abs(sim.terminal.var() - analytic) / analytic < 0.06


def test_zcb_mc_matches_analytical(cir_model):
    """Discounting along exact paths reproduces the analytical ZCB price."""
    r0, T = 0.03, 3.0
    sim = cir_model.simulate_exact(r0, T, 60_000, 150, seed=7)
    disc = np.exp(-np.trapezoid(sim.paths, sim.t_grid, axis=1))
    mc, stderr = disc.mean(), disc.std(ddof=1) / np.sqrt(sim.n_paths)
    assert abs(mc - cir_model.zcb_price(r0, T)) < 4 * stderr


def test_exact_lower_variance_than_euler(cir_model):
    """At a coarse step count the exact scheme is far less biased than Euler."""
    r0, T, n_steps = 0.03, 2.0, 4
    analytic_var = cir_model.conditional_variance(r0, T)
    exact = cir_model.simulate_exact(r0, T, 60_000, n_steps, seed=8)
    euler = cir_model.simulate_euler(r0, T, 60_000, n_steps, seed=8)
    err_exact = abs(exact.terminal.var() - analytic_var)
    err_euler = abs(euler.terminal.var() - analytic_var)
    assert err_exact < err_euler
