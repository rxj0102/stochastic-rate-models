"""Sequential bootstrapping of a zero curve from coupon bonds.

Given a set of coupon-bearing instruments ordered by maturity, the
:class:`YieldCurveBootstrapper` strips a discount-factor curve one instrument at
a time. Discount factors at intermediate coupon dates that fall before the
current instrument's maturity are obtained by log-linear interpolation of the
already-bootstrapped nodes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .interpolation import CurveInterpolator

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Default face value of the quoted bonds.
DEFAULT_FACE: float = 100.0

#: Default number of coupon payments per year.
DEFAULT_FREQ: int = 2

#: Lower bound on maturities used when annualising zero rates.
MIN_TIME: float = 1e-12

#: Bracketing bounds for the terminal discount factor root search.
_DF_LOWER: float = 1e-8
_DF_UPPER: float = 1.0 - 1e-12


def _solve_terminal_df(price_fn, target_price: float) -> float:
    """Solve ``price_fn(df) = target_price`` for the terminal discount factor.

    Parameters
    ----------
    price_fn : callable
        Maps a candidate terminal discount factor to a model bond price.
    target_price : float
        Observed market price to match.

    Returns
    -------
    float
        The discount factor reproducing ``target_price``.
    """
    return float(
        brentq(lambda df: price_fn(df) - target_price, _DF_LOWER, _DF_UPPER)
    )


class YieldCurveBootstrapper:
    """Strip a zero curve from market coupon bonds.

    Attributes
    ----------
    maturities_ : np.ndarray or None
        Bootstrapped node maturities after :meth:`bootstrap`.
    discount_factors_ : np.ndarray or None
        Bootstrapped discount factors aligned with ``maturities_``.
    zero_rates_ : np.ndarray or None
        Continuously compounded zero rates aligned with ``maturities_``.
    """

    def __init__(self) -> None:
        self.maturities_: Optional[np.ndarray] = None
        self.discount_factors_: Optional[np.ndarray] = None
        self.zero_rates_: Optional[np.ndarray] = None

    # ------------------------------------------------------------ bootstrap
    def bootstrap(
        self,
        bonds_df: pd.DataFrame,
        face: float = DEFAULT_FACE,
        freq: int = DEFAULT_FREQ,
    ) -> pd.DataFrame:
        """Bootstrap discount factors and zero rates from coupon bonds.

        Parameters
        ----------
        bonds_df : pandas.DataFrame
            One row per instrument with columns ``maturity`` (years),
            ``coupon_rate`` (annualised, decimal) and ``price`` (per ``face``).
            An optional per-row ``freq`` column overrides the default coupon
            frequency.
        face : float, optional
            Face value the prices are quoted against. Defaults to ``100``.
        freq : int, optional
            Coupon payments per year. Defaults to ``2`` (semi-annual).

        Returns
        -------
        pandas.DataFrame
            Columns ``maturity``, ``discount_factor`` and ``zero_rate``.

        Notes
        -----
        For each bond the present value of the known coupon cash flows is
        subtracted from the dirty price; the residual is attributed to the final
        cash flow (last coupon plus redemption) to solve for the terminal
        discount factor. Discount factors at coupon dates preceding the bond's
        maturity are interpolated log-linearly from the nodes solved so far.
        """
        bonds = bonds_df.sort_values("maturity").reset_index(drop=True)

        node_T: list[float] = []
        node_DF: list[float] = []

        for _, bond in bonds.iterrows():
            maturity = float(bond["maturity"])
            coupon_rate = float(bond["coupon_rate"])
            price = float(bond["price"])
            m = int(bond["freq"]) if "freq" in bond and not pd.isna(bond["freq"]) else int(freq)

            # Coupon payment schedule: t = 1/m, 2/m, ..., maturity.
            n_periods = max(int(round(maturity * m)), 1)
            delta = 1.0 / m
            pay_times = np.array(
                [maturity - (n_periods - 1 - k) * delta for k in range(n_periods)]
            )
            coupon = coupon_rate * delta * face
            final_cf = coupon + face

            def price_given_df(df_final: float) -> float:
                """Reprice the bond when the terminal discount factor is ``df_final``.

                Intermediate coupon dates are interpolated log-linearly over the
                already-bootstrapped nodes *augmented* with the trial terminal
                node, so that coupon dates falling between the last known node
                and ``maturity`` are priced consistently rather than
                flat-extrapolated.
                """
                grid_T = np.array(node_T + [maturity])
                grid_DF = np.array(node_DF + [df_final])
                interp = CurveInterpolator.log_linear_df(grid_T, grid_DF)
                pv = 0.0
                for t in pay_times[:-1]:
                    pv += coupon * float(interp(np.array([t]))[0])
                return pv + final_cf * df_final

            df_final = _solve_terminal_df(price_given_df, price)

            node_T.append(maturity)
            node_DF.append(float(df_final))

        maturities = np.asarray(node_T, dtype=float)
        dfs = np.asarray(node_DF, dtype=float)
        zero_rates = -np.log(dfs) / np.maximum(maturities, MIN_TIME)

        self.maturities_ = maturities
        self.discount_factors_ = dfs
        self.zero_rates_ = zero_rates

        return pd.DataFrame(
            {
                "maturity": maturities,
                "discount_factor": dfs,
                "zero_rate": zero_rates,
            }
        )

    # ----------------------------------------------------------- par curve
    def par_curve(self, maturities: np.ndarray) -> np.ndarray:
        """Par swap rates from the bootstrapped curve.

        Parameters
        ----------
        maturities : np.ndarray
            Annual payment-grid maturities (years) at which to compute par
            rates.

        Returns
        -------
        np.ndarray
            Par rate at each maturity,
            ``c_n = (1 - P(T_n)) / sum_i delta_i P(T_i)``.

        Raises
        ------
        RuntimeError
            If :meth:`bootstrap` has not been run.
        """
        if self.discount_factors_ is None:
            raise RuntimeError("Call bootstrap() before par_curve().")

        maturities = np.asarray(maturities, dtype=float)
        df_interp = CurveInterpolator.log_linear_df(
            self.maturities_, self.discount_factors_
        )

        par = np.empty_like(maturities)
        for i, T in enumerate(maturities):
            # Annual payment grid up to and including T.
            n = max(int(round(T)), 1)
            grid = np.arange(1, n + 1, dtype=float)
            grid[-1] = T  # honour fractional final maturity if any
            dfs = df_interp(grid)
            deltas = np.diff(grid, prepend=0.0)
            annuity = float(np.sum(deltas * dfs))
            annuity = max(annuity, MIN_TIME)
            par[i] = (1.0 - dfs[-1]) / annuity
        return par
