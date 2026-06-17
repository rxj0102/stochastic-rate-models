"""Internal helpers for pricing the dict-based bond convention.

A "bond" throughout the :mod:`risk` package is a plain dictionary::

    {
        "type": "coupon" | "zcb",
        "coupon_rate": float,   # ignored for zcb
        "maturity": float,
        "freq": int,            # ignored for zcb
        "face": float,
    }

These helpers centralise cashflow construction and model-based repricing so
that the public risk engines remain concise and consistent.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from models import ShortRateModel
from pricing import CouponBondPricer, ZCBPricer

#: Default coupon frequency (semi-annual).
DEFAULT_FREQ: int = 2
#: Default face value.
DEFAULT_FACE: float = 100.0
#: Numerical guard for time comparisons.
MIN_TIME: float = 1.0e-12

_COUPON = CouponBondPricer()
_ZCB = ZCBPricer()


def bond_field(bond: Mapping, key: str, default):
    """Return ``bond[key]`` falling back to ``default`` when absent/None."""
    val = bond.get(key, default)
    return default if val is None else val


def bond_cashflows(bond: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, cashflows)`` for a dict-specified bond.

    For a coupon bond the schedule is ``coupon_rate * face / freq`` paid at
    each period end with ``face`` redeemed at maturity. For a zero coupon bond
    a single cashflow of ``face`` is delivered at ``maturity``.
    """
    btype = str(bond.get("type", "coupon")).lower()
    maturity = float(bond_field(bond, "maturity", 1.0))
    face = float(bond_field(bond, "face", DEFAULT_FACE))

    if btype == "zcb":
        return np.array([maturity], dtype=float), np.array([face], dtype=float)

    freq = int(bond_field(bond, "freq", DEFAULT_FREQ))
    coupon_rate = float(bond_field(bond, "coupon_rate", 0.0))
    n = max(int(round(maturity * freq)), 1)
    coupon = coupon_rate * face / freq
    times = np.array([(i + 1) / freq for i in range(n)], dtype=float)
    cfs = np.full(n, coupon, dtype=float)
    cfs[-1] += face
    return times, cfs


def price_bond(model: ShortRateModel, r0: float, bond: Mapping) -> float:
    """Price any dict-specified bond off the model discount curve at ``t=0``.

    Coupon bonds are priced with :class:`CouponBondPricer`; zero coupon bonds
    with :class:`ZCBPricer` scaled by face value.
    """
    btype = str(bond.get("type", "coupon")).lower()
    maturity = float(bond_field(bond, "maturity", 1.0))
    face = float(bond_field(bond, "face", DEFAULT_FACE))

    if btype == "zcb":
        return float(_ZCB.price_analytical(model, r0, maturity) * face)

    freq = int(bond_field(bond, "freq", DEFAULT_FREQ))
    coupon_rate = float(bond_field(bond, "coupon_rate", 0.0))
    return float(_COUPON.price(model, r0, coupon_rate, maturity, freq, face).price)


def price_bond_parallel_shift(
    model: ShortRateModel,
    r0: float,
    bond: Mapping,
    db: float,
    freq: int = DEFAULT_FREQ,
) -> float:
    """Reprice a bond under a parallel ``db`` shift of the periodic zero curve.

    The model discount factor for each cashflow is converted to its
    periodic-compounded zero rate, shifted in parallel by ``db``, and converted
    back to a discount factor. This makes the resulting sensitivity consistent
    with the bond's modified duration (which is defined against a flat parallel
    yield move in the same compounding convention).

    Parameters
    ----------
    model : ShortRateModel
        Discounting model.
    r0 : float
        Base short rate.
    bond : Mapping
        Bond specification.
    db : float
        Parallel shift in (periodic-compounded) zero rates.
    freq : int, optional
        Compounding frequency used for the zero-rate conversion. Default 2.

    Returns
    -------
    float
        Present value of the bond under the shifted curve.
    """
    times, cfs = bond_cashflows(bond)
    dfs = np.array([model.zcb_price(r0, float(T)) for T in times])
    dfs = np.clip(dfs, MIN_TIME, None)
    # df = (1 + z/freq)^(-freq * T)  =>  z = freq * (df^(-1/(freq T)) - 1).
    z = freq * (dfs ** (-1.0 / (freq * times)) - 1.0)
    z_bumped = z + db
    dfs_bumped = (1.0 + z_bumped / freq) ** (-freq * times)
    return float(np.sum(cfs * dfs_bumped))


def price_bond_from_curve(
    times_curve: np.ndarray,
    zero_rates: np.ndarray,
    bond: Mapping,
) -> float:
    """Price a bond off a continuously-compounded zero curve.

    Discount factors ``exp(-R(T) * T)`` are obtained by linearly interpolating
    the supplied zero rate curve onto each cashflow time.

    Parameters
    ----------
    times_curve : np.ndarray
        Strictly increasing maturities of the zero curve.
    zero_rates : np.ndarray
        Continuously-compounded zero rates at ``times_curve``.
    bond : Mapping
        Bond specification.

    Returns
    -------
    float
        Present value of the bond cashflows off the supplied curve.
    """
    times, cfs = bond_cashflows(bond)
    rates = np.interp(times, times_curve, zero_rates)
    dfs = np.exp(-rates * times)
    return float(np.sum(cfs * dfs))


def price_bond_at_horizon(
    model: ShortRateModel,
    r_terminal: np.ndarray,
    bond: Mapping,
    horizon: float,
) -> np.ndarray:
    """Vectorised bond value at ``horizon`` for each terminal short rate.

    Only cashflows strictly after ``horizon`` are valued; cashflows at or
    before the horizon are treated as realised income (accumulated separately
    by the caller). Discounting uses ``model.zcb_price(r, T, t=horizon)`` which
    yields ``P(horizon, T)`` given the short rate ``r`` at the horizon.

    Parameters
    ----------
    model : ShortRateModel
        Model exposing ``zcb_price(r, T, t)``.
    r_terminal : np.ndarray
        Short rates at the horizon, shape ``(n_paths,)``.
    bond : Mapping
        Bond specification.
    horizon : float
        Valuation date (years from today).

    Returns
    -------
    np.ndarray
        Bond present value at the horizon for each path, shape ``(n_paths,)``.
    """
    times, cfs = bond_cashflows(bond)
    r_terminal = np.asarray(r_terminal, dtype=float)

    future = times > horizon + MIN_TIME
    fut_times = times[future]
    fut_cfs = cfs[future]
    if fut_times.size == 0:
        return np.zeros_like(r_terminal)

    # Build a (n_cashflows, n_paths) discount factor matrix. zcb_price is scalar
    # in r but affine models accept array r0; loop over the (few) cashflows and
    # pass the full path vector for vectorisation across paths.
    values = np.zeros_like(r_terminal)
    for cf, T in zip(fut_cfs, fut_times):
        dfs = _vec_zcb(model, r_terminal, float(T), horizon)
        values += cf * dfs
    return values


def realised_income(bond: Mapping, horizon: float) -> float:
    """Sum of cashflows paid at or before ``horizon`` (coupons + any redemption)."""
    times, cfs = bond_cashflows(bond)
    paid = times <= horizon + MIN_TIME
    return float(np.sum(cfs[paid]))


def _vec_zcb(
    model: ShortRateModel, r: np.ndarray, T: float, t: float
) -> np.ndarray:
    """Evaluate ``P(t, T)`` for an array of short rates ``r``.

    Affine models accept a vector ``r0`` directly; if that fails, fall back to
    a per-element loop so the helper works for any model.
    """
    try:
        out = model.zcb_price(r, T, t)
        return np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        return np.array([float(model.zcb_price(float(ri), T, t)) for ri in r])
