"""Tests for risk metrics: key-rate durations, DV01, VaR monotonicity, scenarios."""

import numpy as np
import pytest

from models import VasicekModel, VasicekParams
from pricing import CouponBondPricer
from risk import DV01Engine, InterestRateVaR
from scenarios import ParallelShiftScenario


def test_key_rate_durations_sum_to_modified_duration(vasicek_model, sample_bond):
    """Key-rate durations partition the bond's modified duration."""
    bp = CouponBondPricer().price(vasicek_model, 0.03, 0.04, 10, 2, 100)
    krd = DV01Engine().key_rate_durations(vasicek_model, 0.03, sample_bond)
    assert sum(krd.values()) == pytest.approx(bp.modified_duration, rel=0.10)


def test_dv01_dollar_duration_relationship(vasicek_model, sample_bond):
    bp = CouponBondPricer().price(vasicek_model, 0.03, 0.04, 10, 2, 100)
    dv01 = DV01Engine().dv01(vasicek_model, 0.03, sample_bond)
    assert dv01 * 1e4 == pytest.approx(bp.modified_duration * bp.price, rel=0.02)


def test_portfolio_dv01_is_weighted_sum(vasicek_model):
    engine = DV01Engine()
    b1 = {"type": "coupon", "coupon_rate": 0.04, "maturity": 5, "freq": 2, "face": 100}
    b2 = {"type": "coupon", "coupon_rate": 0.05, "maturity": 10, "freq": 2, "face": 100}
    weights = [0.6, 0.4]
    total = engine.dv01_portfolio(vasicek_model, 0.03, [b1, b2], weights)
    manual = weights[0] * engine.dv01(vasicek_model, 0.03, b1) + weights[1] * engine.dv01(
        vasicek_model, 0.03, b2
    )
    assert total == pytest.approx(manual, abs=1e-9)


def test_var_monotone_in_confidence():
    """VaR at 99% is at least VaR at 95% (a coherence property)."""
    # r0 above theta so rates drift up over the horizon and the bond can lose.
    model = VasicekModel(VasicekParams(kappa=0.3, theta=0.06, sigma=0.02))
    bond = {"type": "coupon", "coupon_rate": 0.03, "maturity": 15, "freq": 2, "face": 100}
    engine = InterestRateVaR()
    v99 = engine.bond_var(model, 0.04, bond, 1.0, 0.99, n_paths=40_000, seed=41)["var"]
    v95 = engine.bond_var(model, 0.04, bond, 1.0, 0.95, n_paths=40_000, seed=41)["var"]
    assert v99 >= v95


def test_parallel_shift_matches_dv01(vasicek_model, sample_bond):
    """A 100bp shift P&L approximates -DV01 * 100 (linear regime)."""
    df = ParallelShiftScenario().run(vasicek_model, 0.03, sample_bond)
    dv01 = DV01Engine().dv01(vasicek_model, 0.03, sample_bond)
    row = df[df["shift_bps"] == 100].iloc[0]
    assert row["pnl"] == pytest.approx(-dv01 * 100, rel=0.05)
