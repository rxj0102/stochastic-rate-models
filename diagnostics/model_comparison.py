"""Cross-model comparison and fit diagnostics for short rate models.

Utilities to benchmark several calibrated short rate models against a common
market yield curve, to price instruments across models, and to study the
sensitivity of the model-implied curve to a single structural parameter.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from models import G2PlusPlusModel, ShortRateModel

#: Basis points per unit of continuously-compounded rate.
_BPS: float = 1.0e4


def _model_yields(
    model: ShortRateModel, r0: float, maturities: np.ndarray
) -> np.ndarray:
    """Return model spot yields, handling the two-factor G2++ signature.

    Parameters
    ----------
    model : ShortRateModel
        Any short rate model.
    r0 : float
        Initial short rate. For G2++ it is ignored and the factors are started
        at ``x0 = y0 = 0``.
    maturities : np.ndarray
        Maturities (years).

    Returns
    -------
    np.ndarray
        Continuously-compounded spot yields.
    """
    maturities = np.asarray(maturities, dtype=float)
    if isinstance(model, G2PlusPlusModel):
        return np.asarray(model.yield_curve(0.0, 0.0, maturities), dtype=float)
    return np.asarray(model.yield_curve(r0, maturities), dtype=float)


class ModelComparison:
    """Compare yield curves, bond prices and parameter sensitivities."""

    # ------------------------------------------------------------- RMSE
    def yield_curve_rmse(
        self,
        model: ShortRateModel,
        market_yields: Sequence[float],
        maturities: Sequence[float],
        r0: float,
    ) -> float:
        """Root-mean-square yield error of a model against the market, in bps.

        Parameters
        ----------
        model : ShortRateModel
            Calibrated model. For G2++ the factors are started at zero.
        market_yields : sequence of float
            Observed continuously-compounded yields.
        maturities : sequence of float
            Maturities (years) aligned with ``market_yields``.
        r0 : float
            Initial short rate (ignored for G2++).

        Returns
        -------
        float
            RMSE in basis points.
        """
        maturities = np.asarray(maturities, dtype=float)
        market = np.asarray(market_yields, dtype=float)
        model_y = _model_yields(model, r0, maturities)
        rmse = np.sqrt(np.mean((model_y - market) ** 2))
        return float(rmse * _BPS)

    # ----------------------------------------------------- curve table
    def compare_yield_curves(
        self,
        models_dict: dict[str, ShortRateModel],
        r0_dict: dict[str, float],
        maturities: Sequence[float],
        market_yields: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Tabulate model-implied yield curves (and the market) on one grid.

        Parameters
        ----------
        models_dict : dict
            Mapping ``name -> model``.
        r0_dict : dict
            Mapping ``name -> initial short rate`` for each model.
        maturities : sequence of float
            Common maturity grid (years), used as the DataFrame index.
        market_yields : sequence of float, optional
            Market yields aligned with ``maturities``; added as a ``market``
            column when provided.

        Returns
        -------
        pandas.DataFrame
            Indexed by maturity, one column of yields per model plus an optional
            ``market`` column.
        """
        maturities = np.asarray(maturities, dtype=float)
        data: dict[str, np.ndarray] = {}
        for name, model in models_dict.items():
            r0 = float(r0_dict.get(name, 0.0))
            data[name] = _model_yields(model, r0, maturities)
        if market_yields is not None:
            data["market"] = np.asarray(market_yields, dtype=float)
        return pd.DataFrame(data, index=pd.Index(maturities, name="maturity"))

    # ----------------------------------------------------- ZCB across models
    def price_zcb_across_models(
        self,
        models_dict: dict[str, ShortRateModel],
        r0_dict: dict[str, float],
        T: float,
    ) -> dict[str, float]:
        """Price a single ``T``-maturity ZCB under every model.

        Parameters
        ----------
        models_dict : dict
            Mapping ``name -> model``.
        r0_dict : dict
            Mapping ``name -> initial short rate``.
        T : float
            Bond maturity in years.

        Returns
        -------
        dict
            Mapping ``name -> analytical ZCB price``.
        """
        prices: dict[str, float] = {}
        for name, model in models_dict.items():
            r0 = float(r0_dict.get(name, 0.0))
            prices[name] = float(model.zcb_price(r0, float(T)))
        return prices

    # ----------------------------------------------- parameter sensitivity
    def parameter_sensitivity(
        self,
        model_factory: Callable[[float], ShortRateModel],
        param_name: str,
        param_range: Sequence[float],
        maturities: Sequence[float],
        r0: float,
    ) -> dict[float, np.ndarray]:
        """Sensitivity of the yield curve to a single structural parameter.

        Parameters
        ----------
        model_factory : callable
            ``model_factory(value) -> model`` building a model for a given value
            of the parameter of interest.
        param_name : str
            Name of the parameter being swept (used only for documentation; the
            sweep is driven entirely by ``model_factory``).
        param_range : sequence of float
            Parameter values to evaluate.
        maturities : sequence of float
            Maturity grid (years).
        r0 : float
            Initial short rate (ignored for G2++).

        Returns
        -------
        dict
            Mapping ``param_value -> yield_curve_array`` over ``maturities``.
        """
        maturities = np.asarray(maturities, dtype=float)
        out: dict[float, np.ndarray] = {}
        for value in param_range:
            model = model_factory(value)
            out[float(value)] = _model_yields(model, r0, maturities)
        return out
