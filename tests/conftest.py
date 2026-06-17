"""Shared pytest fixtures for the stochastic-rate-models test suite."""

import numpy as np
import pytest

from models import (
    CIRModel,
    CIRParams,
    HullWhiteModel,
    HullWhiteParams,
    VasicekModel,
    VasicekParams,
)

MARKET_MATS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
MARKET_YLDS = np.array([0.020, 0.022, 0.025, 0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.038])


@pytest.fixture
def vasicek_model():
    return VasicekModel(VasicekParams(kappa=0.5, theta=0.05, sigma=0.02))


@pytest.fixture
def cir_model():
    # Feller condition holds: 2*0.5*0.05 = 0.05 > 0.1**2 = 0.01.
    return CIRModel(CIRParams(kappa=0.5, theta=0.05, sigma=0.1))


@pytest.fixture
def hw_model():
    return HullWhiteModel(HullWhiteParams(kappa=0.1, sigma=0.01)).set_initial_curve(
        MARKET_MATS, MARKET_YLDS
    )


@pytest.fixture
def market_curve():
    return MARKET_MATS, MARKET_YLDS


@pytest.fixture
def sample_bond():
    return {"type": "coupon", "coupon_rate": 0.04, "maturity": 10, "freq": 2, "face": 100}
