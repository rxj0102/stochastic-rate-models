# stochastic-rate-models

**An institutional-grade stochastic interest rate modeling engine.**

`stochastic-rate-models` implements the workhorse one- and two-factor short
rate models of fixed income — **Vasicek (1977)**, **Cox–Ingersoll–Ross (1985)**,
**Hull–White (1990)** and **G2++** — together with everything needed to put them
to work: exact and Monte Carlo simulation, analytical and simulated bond
pricing, full yield-curve construction, model calibration to market and
historical data, interest-rate derivatives, duration/convexity/VaR risk, and
scenario & stress analysis. It ships as an importable Python library **and** a
live eight-tab Streamlit dashboard.

```
dr(t) = κ(θ − r(t)) dt + σ dW(t)                     # Vasicek
dr(t) = κ(θ − r(t)) dt + σ √r(t) dW(t)               # CIR  (Feller: 2κθ > σ²)
dr(t) = (θ(t) − κ r(t)) dt + σ dW(t)                 # Hull–White (fits curve exactly)
r(t)  = x(t) + y(t) + φ(t)                            # G2++  (two correlated factors)
```

---

## Features

| Area | Highlights |
|------|-----------|
| **Models** | Vasicek, CIR, Hull–White (curve-fitted), G2++ — affine ZCB formulas, analytical yields & forwards |
| **Simulation** | Euler–Maruyama, Milstein, **exact** Gaussian (Vasicek) and **exact** non-central χ² (CIR) transitions — fully vectorized, zero path loops |
| **Term structure** | Yield-curve builder, forward curves, Nelson–Siegel & Svensson fitting, bootstrapping, cubic/monotone interpolation |
| **Pricing** | Analytical + MC ZCB, coupon bonds (YTM/duration/convexity), bond options (Jamshidian), caps/floors, swaptions, portfolios |
| **Calibration** | MLE, OLS (AR(1)/Pearson–Sun), Hull–White to cap prices, Kalman filter for the latent short rate |
| **Risk** | Macaulay/modified duration, convexity, DV01, key-rate durations, VaR/CVaR, P&L attribution |
| **Scenarios** | Parallel shifts, steepeners/flatteners, five historical stress scenarios, custom curve shocks |
| **Diagnostics** | ADF/KPSS stationarity, MC convergence, cross-model comparison, calibration residual diagnostics |
| **App** | Eight-tab Streamlit dashboard with Plotly charts |

---

## Installation

```bash
git clone https://github.com/rxj0102/stochastic-rate-models.git
cd stochastic-rate-models
pip install -r requirements.txt
pip install -e .            # optional: install the library
```

Requires Python ≥ 3.10.

---

## Quick start

```python
import numpy as np
from models import VasicekModel, VasicekParams

model = VasicekModel(VasicekParams(kappa=0.5, theta=0.05, sigma=0.02))

# Analytical zero-coupon bond and yield curve
print(model.zcb_price(r0=0.03, T=5.0))                 # P(0, 5)
maturities = np.array([1, 2, 5, 10, 30])
print(model.yield_curve(r0=0.03, maturities=maturities))

# Exact (bias-free) Monte Carlo simulation
sim = model.simulate(r0=0.03, T=5.0, n_paths=10_000, n_steps=250, seed=42)
print(sim.terminal.mean(), sim.elapsed_seconds)
```

### Hull–White fitted to a market curve

```python
from models import HullWhiteModel, HullWhiteParams

mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
ylds = np.array([0.020, 0.022, 0.025, 0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.038])

hw = HullWhiteModel(HullWhiteParams(kappa=0.1, sigma=0.01)).set_initial_curve(mats, ylds)
hw.yield_curve_fitted(mats)          # reproduces `ylds` to < 0.1 bp
hw.cap_price(cap_rate=0.03, maturity=5.0, notional=1_000_000, freq=4)
hw.swaption_price(swap_rate=0.035, tenor=5.0, T_option=1.0, notional=1_000_000)
```

### Pricing, risk and scenarios

```python
from pricing import CouponBondPricer
from risk import DV01Engine
from scenarios import ParallelShiftScenario

bond = {"type": "coupon", "coupon_rate": 0.04, "maturity": 10, "freq": 2, "face": 100}

bp = CouponBondPricer().price(model, r0=0.03, coupon_rate=0.04, maturity=10, freq=2, face=100)
print(bp.price, bp.modified_duration, bp.convexity, bp.dv01)

print(DV01Engine().dv01(model, 0.03, bond))
print(ParallelShiftScenario().run(model, 0.03, bond))   # P&L across ±200 bp shifts
```

### Calibration to a historical rate series

```python
from calibration import MLECalibrator, OLSCalibrator

ols = OLSCalibrator().calibrate_vasicek(rate_series, dt=1/252)
mle = MLECalibrator().calibrate_vasicek(rate_series, dt=1/252)
print(mle["params"], mle["aic"], mle["bic"])
```

---

## The Streamlit dashboard

```bash
streamlit run app.py
```

Eight tabs:

1. **Rate Path Simulator** — fan charts, terminal distributions, path statistics
2. **Yield Curve Builder** — model vs market curves, forwards, curve-shape detection
3. **Bond Pricing** — price/YTM/duration/convexity/DV01, MC-vs-analytical convergence
4. **Model Calibration** — MLE / OLS / Kalman fits with residual diagnostics
5. **Derivatives Pricing** — bond options, caps/floors, swaptions, payoff diagrams
6. **Risk Dashboard** — portfolio DV01, key-rate durations, VaR/CVaR
7. **Scenario Analysis** — parallel/twist/historical/custom curve shocks
8. **Model Documentation** — SDEs, formulas, Feller condition, half-life explorers

---

## Project layout

```
models/          Vasicek, CIR, Hull-White, G2++ short rate models
simulation/      Euler, Milstein, exact Vasicek/CIR, unified path manager
term_structure/  Yield/forward curves, Nelson-Siegel, bootstrapping, interpolation
pricing/         ZCB, coupon bonds, bond options, caps/floors, swaptions, portfolios
calibration/     MLE, OLS, market (cap) calibration, Kalman filter
risk/            Duration/convexity, DV01/key-rate, VaR/CVaR, P&L attribution
scenarios/       Parallel shifts, twists, historical stress, scenario engine
diagnostics/     Stationarity, MC convergence, model comparison, residuals
utils/           Plotly chart builders, rate statistics
tests/           pytest suite
app.py           Streamlit dashboard
paper/           ReportLab SSRN-ready PDF generator
notebooks/       Deep-dive walkthrough
```

---

## Testing

```bash
pytest -q
```

The suite verifies analytical/simulation consistency (conditional moments,
ZCB limits), CIR positivity under Feller, Hull–White exact curve fit and
cap–floor parity, YTM/duration/convexity identities, calibration round-trips,
and risk-metric relationships (key-rate ↔ duration, VaR monotonicity).

---

## Mathematical notes

- **Vasicek** is Gaussian with closed-form transition `r(t)|r(s) ~ N(μ, v²)`,
  affine term structure `P(t,T)=A(t,T)e^{−B(t,T)r}`, and analytical bond options
  (Jamshidian, 1989). Rates can go negative — a known limitation.
- **CIR** uses a square-root diffusion guaranteeing non-negativity; under the
  Feller condition `2κθ > σ²` it stays strictly positive. The exact transition
  is a scaled non-central χ², enabling unbiased simulation.
- **Hull–White** makes the drift level `θ(t)` time-dependent so that today's
  curve is reproduced *exactly*, which is why it is the standard for pricing
  caps, floors and swaptions. Includes a calibrated trinomial tree.
- **G2++** adds a second correlated Gaussian factor, generating richer (e.g.
  humped, de-correlated) curve dynamics than any one-factor model.

---

## References

- Vasicek, O. (1977). *An equilibrium characterization of the term structure.* JFE 5(2).
- Cox, J., Ingersoll, J., Ross, S. (1985). *A theory of the term structure of interest rates.* Econometrica 53(2).
- Hull, J., White, A. (1990). *Pricing interest-rate-derivative securities.* RFS 3(4).
- Jamshidian, F. (1989). *An exact bond option formula.* Journal of Finance 44(1).
- Brigo, D., Mercurio, F. (2006). *Interest Rate Models — Theory and Practice* (2nd ed.). Springer.

---

## License

MIT — see [LICENSE](LICENSE).

> **Disclaimer.** This library is for research and educational purposes. It is
> not investment advice and carries no warranty of fitness for trading.
