"""Shared curve utilities for the scenario engine.

Scenarios operate on the *zero (spot) yield curve* rather than directly on the
short rate. This keeps them model agnostic: a scenario simply maps a base zero
curve to a shifted zero curve, and bonds are repriced by discounting their
cashflows off the (interpolated) shifted curve. The helpers in this module

* build the base zero curve from any :class:`~models.ShortRateModel`,
* interpolate yields onto arbitrary maturities, and
* reprice coupon bonds / zero coupon bonds off a discrete zero curve.

All yields are continuously compounded and expressed in decimal (so 5% is
``0.05``); shifts are likewise in decimal unless explicitly suffixed ``_bps``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from scipy.interpolate import CubicSpline

from models import ShortRateModel

#: Default maturity grid (in years) used when callers do not supply one.
DEFAULT_MATURITIES: np.ndarray = np.array(
    [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float
)

#: One basis point expressed in decimal.
ONE_BP: float = 1.0e-4

#: Numerical guard for the smallest representable maturity.
MIN_TIME: float = 1.0e-6


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------
def base_curve(
    model: ShortRateModel,
    r0: float,
    maturities: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the base zero (spot) yield curve from a short rate model.

    Parameters
    ----------
    model : ShortRateModel
        Any short rate model exposing ``yield_curve``. For the two-factor
        G2++ model ``yield_curve(x0, y0, maturities)`` is called with the
        factor seed taken from ``r0`` (passed for both factors if a scalar).
    r0 : float
        Initial short rate (or, for G2++, the seed for both factors).
    maturities : np.ndarray, optional
        Maturity grid in years. Defaults to :data:`DEFAULT_MATURITIES`.

    Returns
    -------
    maturities : np.ndarray
        The maturity grid actually used (a float copy).
    base_yields : np.ndarray
        Continuously compounded spot yields in decimal.
    """
    maturities = (
        DEFAULT_MATURITIES.copy()
        if maturities is None
        else np.asarray(maturities, dtype=float)
    )
    yields = _model_yield_curve(model, r0, maturities)
    return maturities, np.asarray(yields, dtype=float)


def _model_yield_curve(
    model: ShortRateModel, r0: float, maturities: np.ndarray
) -> np.ndarray:
    """Call ``model.yield_curve`` handling the two-factor G2++ signature."""
    try:
        return model.yield_curve(r0, maturities)
    except TypeError:
        # G2++ expects two factor seeds: yield_curve(x0, y0, maturities).
        return model.yield_curve(0.0, 0.0, maturities)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def interp_yields(
    maturities: np.ndarray,
    yields: np.ndarray,
    query: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """Interpolate a zero curve onto ``query`` maturities.

    Parameters
    ----------
    maturities : np.ndarray
        Curve node maturities (years), strictly increasing.
    yields : np.ndarray
        Yields at ``maturities`` (decimal).
    query : np.ndarray
        Maturities at which to evaluate the curve.
    method : {"cubic", "linear"}, optional
        Interpolation scheme. ``cubic`` uses a natural-boundary cubic spline
        with flat extrapolation; ``linear`` uses piecewise-linear interpolation
        with flat extrapolation. Falls back to linear when there are fewer than
        four nodes (insufficient for a stable cubic spline).

    Returns
    -------
    np.ndarray
        Interpolated yields at ``query``.
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    query = np.atleast_1d(np.asarray(query, dtype=float))

    if method == "cubic" and maturities.size >= 4:
        spline = CubicSpline(maturities, yields, bc_type="natural", extrapolate=True)
        out = spline(query)
        # Flat extrapolation beyond the node range to avoid wild spline tails.
        out = np.where(query < maturities[0], yields[0], out)
        out = np.where(query > maturities[-1], yields[-1], out)
        return out

    return np.interp(query, maturities, yields, left=yields[0], right=yields[-1])


def shift_from_tenor_dict(
    maturities: np.ndarray,
    tenor_shifts_bps: Mapping[float, float],
    method: str = "cubic",
) -> np.ndarray:
    """Spline/linear-interpolate a per-key-tenor shift onto the full grid.

    Parameters
    ----------
    maturities : np.ndarray
        Full maturity grid (years).
    tenor_shifts_bps : mapping of float to float
        Shift in basis points keyed by tenor (years), e.g. ``{2: -300, 10: -200}``.
    method : {"cubic", "linear"}, optional
        Interpolation scheme passed to :func:`interp_yields`.

    Returns
    -------
    np.ndarray
        Per-maturity shift in *decimal* (bps / 1e4) aligned to ``maturities``.
    """
    if not tenor_shifts_bps:
        return np.zeros_like(np.asarray(maturities, dtype=float))
    key_tenors = np.array(sorted(tenor_shifts_bps), dtype=float)
    key_shifts = np.array([tenor_shifts_bps[float(t)] for t in key_tenors], dtype=float)
    shifts_bps = interp_yields(key_tenors, key_shifts, maturities, method=method)
    return shifts_bps * ONE_BP


# ---------------------------------------------------------------------------
# Bond repricing off a discrete curve
# ---------------------------------------------------------------------------
def _cashflow_schedule(
    coupon_rate: float, maturity: float, freq: int, face: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, cashflows)`` for a fixed-rate coupon bond.

    Mirrors the convention used by :class:`pricing.CouponBondPricer`: coupon
    ``coupon_rate * face / freq`` paid at each period end, redeeming ``face``
    with the final coupon.
    """
    n = max(int(round(maturity * freq)), 1)
    coupon = coupon_rate * face / freq
    times = np.array([(i + 1) / freq for i in range(n)], dtype=float)
    cfs = np.full(n, coupon, dtype=float)
    cfs[-1] += face
    return times, cfs


def price_bond_off_curve(
    maturities: np.ndarray,
    yields: np.ndarray,
    bond: Mapping[str, Any],
    method: str = "cubic",
) -> float:
    """Price a bond by discounting cashflows off a discrete zero curve.

    Each cashflow at time ``t`` is discounted with ``DF(t) = exp(-y(t) * t)``
    where ``y(t)`` is interpolated from ``(maturities, yields)``.

    Parameters
    ----------
    maturities : np.ndarray
        Zero curve node maturities (years).
    yields : np.ndarray
        Continuously compounded spot yields at ``maturities`` (decimal).
    bond : mapping
        Bond specification. Recognised keys::

            type        : "coupon" or "zcb"   (default "coupon")
            coupon_rate : annual coupon rate  (default 0.0; ignored for zcb)
            maturity    : maturity in years   (required)
            freq        : coupon frequency    (default 2)
            face        : face value          (default 100.0)

    method : {"cubic", "linear"}, optional
        Yield interpolation scheme.

    Returns
    -------
    float
        Present value of the bond off the supplied curve.
    """
    btype = str(bond.get("type", "coupon")).lower()
    maturity = float(bond["maturity"])
    face = float(bond.get("face", 100.0))

    if btype == "zcb":
        times = np.array([max(maturity, MIN_TIME)], dtype=float)
        cfs = np.array([face], dtype=float)
    else:
        coupon_rate = float(bond.get("coupon_rate", 0.0))
        freq = int(bond.get("freq", 2))
        times, cfs = _cashflow_schedule(coupon_rate, maturity, freq, face)

    times = np.maximum(times, MIN_TIME)
    y = interp_yields(maturities, yields, times, method=method)
    dfs = np.exp(-y * times)
    return float(np.sum(cfs * dfs))
