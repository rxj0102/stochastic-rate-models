"""Tests for bond pricing: ZCB limits, YTM round-trip, duration/convexity/DV01."""

import numpy as np
import pytest

from pricing import BondOptionPricer, CouponBondPricer, ZCBPricer


def test_zcb_at_zero_all_models(vasicek_model, cir_model, hw_model):
    assert vasicek_model.zcb_price(0.03, 0.0) == pytest.approx(1.0)
    assert cir_model.zcb_price(0.03, 0.0) == pytest.approx(1.0)
    assert hw_model.zcb_price(hw_model.r0, 0.0) == pytest.approx(1.0)


def test_coupon_bond_exceeds_zcb(vasicek_model):
    """A coupon bond is worth more than the equivalent zero when coupon > 0."""
    pricer = CouponBondPricer()
    coupon = pricer.price(vasicek_model, 0.03, 0.05, 10, 2, 100).price
    zcb = ZCBPricer().price_analytical(vasicek_model, 0.03, 10.0) * 100
    assert coupon > zcb


def test_ytm_roundtrip():
    pricer = CouponBondPricer()
    y = 0.045
    price = pricer.price_from_yield(y, 0.04, 10, 2, 100)
    recovered = pricer.ytm_from_price(price, 0.04, 10, 2, 100)
    assert recovered == pytest.approx(y, abs=1e-8)


def test_dv01_matches_dollar_duration(vasicek_model):
    """DV01 should equal modified_duration * price * 1bp."""
    bp = CouponBondPricer().price(vasicek_model, 0.03, 0.04, 10, 2, 100)
    assert bp.dv01 == pytest.approx(bp.modified_duration * bp.price * 1e-4, rel=0.01)


def test_convexity_positive(vasicek_model):
    bp = CouponBondPricer().price(vasicek_model, 0.03, 0.04, 10, 2, 100)
    assert bp.convexity > 0


def test_price_yield_monotonic():
    pricer = CouponBondPricer()
    prices = [pricer.price_from_yield(y, 0.04, 10, 2, 100) for y in (0.02, 0.04, 0.06, 0.08)]
    assert all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def test_jamshidian_call_matches_mc(vasicek_model):
    """Analytical ZCB call agrees with the Monte Carlo estimate."""
    op = BondOptionPricer()
    analytic = op.european_call_zcb(vasicek_model, 0.03, 1.0, 3.0, 0.9)
    mc, stderr = op.mc_bond_option(vasicek_model, 0.03, 1.0, 3.0, 0.9, n_paths=40_000, seed=31)
    assert abs(analytic - mc) < 4 * stderr
