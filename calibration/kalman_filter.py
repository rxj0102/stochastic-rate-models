"""Kalman filter estimation of the Vasicek latent short rate.

The Vasicek model admits an exact linear Gaussian state-space representation,
which makes the Kalman filter the natural tool for estimating the latent short
rate and the structural parameters from a panel of observed yields.

State equation (exact Gaussian transition)
    ``r_{t+1} = phi r_t + (1 - phi) theta + eta_t``,
    ``phi = exp(-kappa dt)``,
    ``Var(eta_t) = sigma^2 / (2 kappa) (1 - exp(-2 kappa dt))``.

Observation equation
    Each observed yield is affine in the short rate. For maturity ``T`` the
    Vasicek model yield is
    ``R(r, T) = -(lnA(T) - B(T) r) / T = a_T + b_T r`` with
    ``a_T = -lnA(T) / T`` and ``b_T = B(T) / T``. Observed yields equal the
    model yields plus i.i.d. Gaussian measurement noise.

:class:`KalmanFilterEstimator` provides ``filter`` (run the filter for fixed
parameters, returning filtered rates and the Gaussian log-likelihood) and
``estimate`` (maximise that log-likelihood over the parameters).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import optimize

from models import VasicekParams

_MIN_KAPPA = 1e-6
_MIN_SIGMA = 1e-8
_MIN_NOISE = 1e-10


def _vasicek_affine_loadings(
    kappa: float, theta: float, sigma: float, maturities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Affine yield loadings ``a_T`` and ``b_T`` for each maturity.

    ``R(r, T) = a_T + b_T r`` where ``a_T = -lnA(T)/T`` and ``b_T = B(T)/T``.

    Parameters
    ----------
    kappa, theta, sigma : float
        Vasicek parameters.
    maturities : np.ndarray
        Yield maturities (years), strictly positive.

    Returns
    -------
    a : np.ndarray
        Intercept loadings, shape ``(n_maturities,)``.
    b : np.ndarray
        Slope loadings on the short rate, shape ``(n_maturities,)``.
    """
    tau = np.asarray(maturities, dtype=float)
    k = max(kappa, _MIN_KAPPA)
    B = (1.0 - np.exp(-k * tau)) / k
    lnA = (theta - sigma**2 / (2.0 * k**2)) * (B - tau) - sigma**2 * B**2 / (4.0 * k)
    a = -lnA / tau
    b = B / tau
    return a, b


class KalmanFilterEstimator:
    """Linear Gaussian Kalman filter for the Vasicek model on a yield panel."""

    def filter(
        self,
        observed_yields: np.ndarray,
        maturities: Sequence[float],
        dt: float,
        params: VasicekParams,
        obs_noise: float = 1e-4,
    ) -> Dict:
        """Run the Kalman filter for fixed parameters.

        Parameters
        ----------
        observed_yields : np.ndarray
            Observed yields, shape ``(T_obs, n_maturities)``.
        maturities : sequence of float
            Maturities (years) of the observed yields.
        dt : float
            Time step between observation dates (years).
        params : VasicekParams
            Vasicek parameters.
        obs_noise : float, optional
            Standard deviation of the i.i.d. yield measurement noise.

        Returns
        -------
        dict
            Keys: ``filtered_rates`` (shape ``(T_obs,)``),
            ``filtered_variances``, ``log_likelihood``.
        """
        y = np.atleast_2d(np.asarray(observed_yields, dtype=float))
        mats = np.asarray(maturities, dtype=float)
        kappa = max(params.kappa, _MIN_KAPPA)
        theta = float(params.theta)
        sigma = max(params.sigma, _MIN_SIGMA)

        T_obs, n_mat = y.shape
        a, b = _vasicek_affine_loadings(kappa, theta, sigma, mats)
        H = b.reshape(n_mat, 1)  # observation matrix (n_mat x 1)

        phi = float(np.exp(-kappa * dt))
        c = (1.0 - phi) * theta  # state intercept
        Q = (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * dt))
        R = np.eye(n_mat) * max(obs_noise, _MIN_NOISE) ** 2

        # Initialise at the stationary distribution.
        x = theta
        P = sigma**2 / (2.0 * kappa)

        filtered = np.empty(T_obs)
        filtered_var = np.empty(T_obs)
        log_lik = 0.0
        const = n_mat * np.log(2.0 * np.pi)

        for t in range(T_obs):
            # Predict.
            x_pred = phi * x + c
            P_pred = phi**2 * P + Q

            # Observation prediction and innovation.
            y_pred = a + b * x_pred
            innov = y[t] - y_pred
            S = (H @ H.T) * P_pred + R  # (n_mat x n_mat)

            try:
                S_inv = np.linalg.inv(S)
                sign, logdet = np.linalg.slogdet(S)
            except np.linalg.LinAlgError:
                sign, logdet, S_inv = 1.0, 0.0, np.eye(n_mat)

            # Kalman gain: K = P_pred H^T S^{-1}  (1 x n_mat).
            K = (P_pred * H.T) @ S_inv

            x = float(x_pred + (K @ innov).item())
            P = float((1.0 - (K @ H).item()) * P_pred)

            filtered[t] = x
            filtered_var[t] = P

            if sign <= 0 or not np.isfinite(logdet):
                log_lik += -1e10
            else:
                quad = float(innov @ S_inv @ innov)
                log_lik += -0.5 * (const + logdet + quad)

        return {
            "filtered_rates": filtered,
            "filtered_variances": filtered_var,
            "log_likelihood": float(log_lik),
        }

    def estimate(
        self,
        observed_yields: np.ndarray,
        maturities: Sequence[float],
        dt: float,
        x0_params: VasicekParams,
        obs_noise0: float = 1e-3,
    ) -> Dict:
        """Estimate Vasicek parameters by maximising the KF log-likelihood.

        Free parameters are ``(kappa, theta, sigma, obs_noise)``.

        Parameters
        ----------
        observed_yields : np.ndarray
            Observed yields, shape ``(T_obs, n_maturities)``.
        maturities : sequence of float
            Maturities (years) of the observed yields.
        dt : float
            Time step between observation dates (years).
        x0_params : VasicekParams
            Starting values for ``(kappa, theta, sigma)``.
        obs_noise0 : float, optional
            Starting value for the measurement-noise standard deviation.

        Returns
        -------
        dict
            Keys: ``params`` (:class:`VasicekParams`), ``obs_noise``,
            ``filtered_rates``, ``log_likelihood``, ``success``.
        """
        y = np.atleast_2d(np.asarray(observed_yields, dtype=float))
        mats = np.asarray(maturities, dtype=float)

        x0 = np.array([
            max(x0_params.kappa, 10.0 * _MIN_KAPPA),
            float(x0_params.theta),
            max(x0_params.sigma, 10.0 * _MIN_SIGMA),
            max(obs_noise0, 10.0 * _MIN_NOISE),
        ])
        bounds = [
            (_MIN_KAPPA, 5.0),
            (None, None),
            (_MIN_SIGMA, 1.0),
            (_MIN_NOISE, 0.5),
        ]

        def neg_loglik(p: np.ndarray) -> float:
            kappa, theta, sigma, noise = p
            if kappa <= _MIN_KAPPA or sigma <= _MIN_SIGMA:
                return 1e12
            try:
                out = self.filter(
                    y, mats, dt,
                    VasicekParams(kappa=kappa, theta=theta, sigma=sigma),
                    obs_noise=noise,
                )
            except Exception:
                return 1e12
            ll = out["log_likelihood"]
            if not np.isfinite(ll):
                return 1e12
            return -ll

        res = optimize.minimize(
            neg_loglik, x0, method="L-BFGS-B", bounds=bounds
        )
        kappa, theta, sigma, noise = (
            float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3])
        )
        params = VasicekParams(
            kappa=max(kappa, _MIN_KAPPA),
            theta=theta,
            sigma=max(sigma, _MIN_SIGMA),
        )
        out = self.filter(y, mats, dt, params, obs_noise=noise)

        return {
            "params": params,
            "obs_noise": noise,
            "filtered_rates": out["filtered_rates"],
            "log_likelihood": out["log_likelihood"],
            "success": bool(res.success),
        }
