"""Rate statistics helper functions."""

from __future__ import annotations

import numpy as np

DAYS_PER_YEAR: int = 252


def trapezoid_integral(values: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Trapezoidal integral, compatible across NumPy versions.

    NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid``. This wrapper picks
    whichever is available.
    """
    fn = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return fn(values, x=x, axis=axis)


def integrated_rate(paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Return ``integral_0^T r(u) du`` per path via the trapezoidal rule.

    Parameters
    ----------
    paths : np.ndarray
        Short rate paths, shape ``(n_paths, n_steps + 1)``.
    t_grid : np.ndarray
        Time grid, shape ``(n_steps + 1,)``.

    Returns
    -------
    np.ndarray
        Integrated rate per path, shape ``(n_paths,)``.
    """
    return trapezoid_integral(paths, t_grid, axis=1)


def mc_discount_factor(
    paths: np.ndarray, t_grid: np.ndarray
) -> tuple[float, float, tuple[float, float]]:
    """Monte Carlo discount factor ``E[exp(-integral r du)]``.

    Returns
    -------
    tuple
        ``(price, standard_error, (ci_low, ci_high))`` for a 95% interval.
    """
    disc = np.exp(-integrated_rate(paths, t_grid))
    price = float(disc.mean())
    stderr = float(disc.std(ddof=1) / np.sqrt(len(disc)))
    ci = (price - 1.96 * stderr, price + 1.96 * stderr)
    return price, stderr, ci


def summary_statistics(x: np.ndarray) -> dict[str, float]:
    """Return common summary statistics for a 1-D sample."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "q05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "q95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "skew": float(_skew(x)),
        "kurtosis": float(_kurtosis(x)),
    }


def _skew(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def autocorrelation(x: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """Sample autocorrelation function up to ``max_lag``."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag + 1)
    acf = np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(max_lag + 1)])
    return acf


def annualize_rate(rate: float, periods_per_year: int = DAYS_PER_YEAR) -> float:
    """Annualise a per-period rate via continuous compounding."""
    return float(rate * periods_per_year)
