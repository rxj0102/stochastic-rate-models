"""Interest rate risk analytics.

The :mod:`risk` package provides model-agnostic risk engines for fixed income
instruments priced off the library's short rate models:

* :class:`RateSensitivity` -- cashflow and model based duration/convexity.
* :class:`DV01Engine` -- DV01, key rate durations and portfolio DV01.
* :class:`InterestRateVaR` -- Monte Carlo and historical Value-at-Risk / CVaR.
* :class:`PnLAttribution` -- carry/duration/convexity P&L decomposition.

A "bond" is the dict convention
``{"type", "coupon_rate", "maturity", "freq", "face"}`` understood by all
engines (see :mod:`risk._bond_utils`).
"""

from .duration_convexity import RateSensitivity
from .dv01 import DV01Engine
from .pnl_attribution import PnLAttribution
from .var_rates import InterestRateVaR

__all__ = [
    "RateSensitivity",
    "DV01Engine",
    "InterestRateVaR",
    "PnLAttribution",
]
