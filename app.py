"""Streamlit dashboard for the stochastic-rate-models library.

An eight-tab institutional dashboard covering short-rate simulation, yield-curve
construction, bond pricing, calibration, derivatives, risk, scenarios and model
documentation. Launch with::

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from models import (
    CIRModel,
    CIRParams,
    G2Params,
    G2PlusPlusModel,
    HullWhiteModel,
    HullWhiteParams,
    VasicekModel,
    VasicekParams,
)
from utils import plotting

# --------------------------------------------------------------------------- #
# Constants & page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Stochastic Rate Models",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAMES = ["Vasicek", "CIR", "Hull-White", "G2++"]
DEFAULT_MATS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
DEFAULT_YLDS = np.array([0.020, 0.022, 0.025, 0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.038])
KEY_TENORS = [0.25, 0.5, 1, 2, 5, 10, 30]


# --------------------------------------------------------------------------- #
# Model construction (kept pure so callers can cache on hashable params)
# --------------------------------------------------------------------------- #
def build_model(name: str, params: dict):
    """Construct a model instance from a name and a parameter dict."""
    if name == "Vasicek":
        return VasicekModel(VasicekParams(params["kappa"], params["theta"], params["sigma"]))
    if name == "CIR":
        return CIRModel(CIRParams(params["kappa"], params["theta"], params["sigma"]))
    if name == "Hull-White":
        m = HullWhiteModel(HullWhiteParams(params["kappa"], params["sigma"]))
        return m.set_initial_curve(DEFAULT_MATS, DEFAULT_YLDS)
    if name == "G2++":
        m = G2PlusPlusModel(
            G2Params(params["a"], params["b"], params["sigma"], params["eta"], params["rho"])
        )
        return m.set_initial_curve(DEFAULT_MATS, DEFAULT_YLDS)
    raise ValueError(f"Unknown model: {name}")


def model_yields(model, r0: float, maturities: np.ndarray) -> np.ndarray:
    """Yield curve helper that handles the G2++ two-factor signature."""
    if isinstance(model, G2PlusPlusModel):
        return model.yield_curve(0.0, 0.0, maturities)
    return model.yield_curve(r0, maturities)


def effective_r0(model, r0: float) -> float:
    """Return the appropriate r0 (curve-implied for HW)."""
    if isinstance(model, HullWhiteModel):
        return model.r0
    return r0


# --------------------------------------------------------------------------- #
# Cached compute helpers (operate on hashable args, rebuild model internally)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_simulate(name, params, r0, T, n_paths, n_steps, seed):
    model = build_model(name, dict(params))
    r0 = effective_r0(model, r0)
    sim = model.simulate(r0, T, n_paths, n_steps, seed)
    return sim.paths, sim.t_grid, float(sim.elapsed_seconds)


@st.cache_data(show_spinner=False)
def cached_yield_curve(name, params, r0, maturities):
    model = build_model(name, dict(params))
    r0 = effective_r0(model, r0)
    mats = np.asarray(maturities)
    ylds = model_yields(model, r0, mats)
    prices = np.array([model.zcb_price(r0, float(T)) for T in mats])
    return ylds, prices


# --------------------------------------------------------------------------- #
# Sidebar: model selector and parameters
# --------------------------------------------------------------------------- #
def sidebar_model_inputs():
    """Render the shared model selector + parameter sliders once in the sidebar.

    Returns ``(name, params, r0)``. Called a single time per run from ``main``
    so the sidebar shows one coherent set of controls used across tabs.
    """
    st.sidebar.markdown("### Model")
    name = st.sidebar.selectbox("Short rate model", MODEL_NAMES, key="model")
    params: dict = {}
    r0 = st.sidebar.slider("Initial rate r₀", 0.0, 0.10, 0.03, 0.001, key="r0")

    st.sidebar.markdown("### Parameters")
    if name in ("Vasicek", "CIR"):
        params["kappa"] = st.sidebar.slider("κ (mean reversion)", 0.01, 3.0, 0.5, 0.01, key="k")
        params["theta"] = st.sidebar.slider("θ (long-run mean)", 0.0, 0.10, 0.05, 0.001, key="th")
        params["sigma"] = st.sidebar.slider("σ (volatility)", 0.001, 0.20, 0.02, 0.001, key="s")
        if name == "CIR":
            feller = 2 * params["kappa"] * params["theta"]
            sig2 = params["sigma"] ** 2
            if feller > sig2:
                st.sidebar.success(f"Feller OK: 2κθ={feller:.4f} > σ²={sig2:.4f}")
            else:
                st.sidebar.error(f"Feller violated: 2κθ={feller:.4f} ≤ σ²={sig2:.4f}")
    elif name == "Hull-White":
        params["kappa"] = st.sidebar.slider("κ (mean reversion)", 0.01, 2.0, 0.10, 0.01, key="k")
        params["sigma"] = st.sidebar.slider("σ (volatility)", 0.001, 0.10, 0.01, 0.001, key="s")
        st.sidebar.caption("θ(t) is fitted to the default market curve.")
    else:  # G2++
        params["a"] = st.sidebar.slider("a (factor 1 reversion)", 0.01, 2.0, 0.5, 0.01, key="a")
        params["b"] = st.sidebar.slider("b (factor 2 reversion)", 0.01, 2.0, 0.1, 0.01, key="b")
        params["sigma"] = st.sidebar.slider("σ (factor 1 vol)", 0.001, 0.10, 0.01, 0.001, key="s")
        params["eta"] = st.sidebar.slider("η (factor 2 vol)", 0.001, 0.10, 0.005, 0.001, key="e")
        params["rho"] = st.sidebar.slider("ρ (correlation)", -0.99, 0.99, -0.7, 0.01, key="rho")
    return name, params, r0


# --------------------------------------------------------------------------- #
# Tab 1 — Rate Path Simulator
# --------------------------------------------------------------------------- #
def tab_simulator(name, params, r0):
    st.header("Rate Path Simulator")
    c1, c2, c3 = st.columns(3)
    T = c1.slider("Horizon T (years)", 0.5, 30.0, 5.0, 0.5)
    n_paths = c2.select_slider("Paths", [1_000, 5_000, 10_000, 25_000], value=10_000)
    n_steps = c3.slider("Steps", 50, 500, 250, 10)

    try:
        paths, t_grid, elapsed = cached_simulate(name, tuple(sorted(params.items())), r0, T, n_paths, n_steps, 42)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return

    model = build_model(name, params)
    theta = params.get("theta") if name == "Vasicek" else None
    fig = plotting.fan_chart(t_grid, paths, mean_path=paths.mean(0), theta=theta,
                             title=f"{name}: Simulated Short Rate Paths")
    st.plotly_chart(fig, use_container_width=True)

    cda, cdb = st.columns([2, 1])
    with cda:
        terminal = paths[:, -1]
        dens_x = dens_y = None
        if name == "Vasicek":
            from scipy import stats as _st
            mu, sd = terminal.mean(), terminal.std()
            dens_x = np.linspace(terminal.min(), terminal.max(), 200)
            dens_y = _st.norm.pdf(dens_x, mu, sd)
        elif name == "CIR":
            try:
                dens_x = np.linspace(max(terminal.min(), 1e-6), terminal.max(), 200)
                dens_y = model.spot_rate_density(effective_r0(model, r0), T, dens_x)
            except Exception:
                dens_x = dens_y = None
        hist = plotting.histogram_with_density(terminal, dens_x, dens_y,
                                               title=f"Distribution of r(T={T:g})", xaxis="Short rate")
        st.plotly_chart(hist, use_container_width=True)
    with cdb:
        st.markdown("#### Path statistics")
        stats_rows = {
            "E[r(T)]": f"{paths[:, -1].mean():.4%}",
            "Var[r(T)]": f"{paths[:, -1].var():.2e}",
            "P(r(T) < 0)": f"{(paths[:, -1] < 0).mean():.2%}",
        }
        try:
            hl = model.half_life()
            stats_rows["Half-life (yrs)"] = f"{hl:.2f}" if np.isfinite(hl) else "∞"
        except Exception:
            pass
        stats_rows["Elapsed (s)"] = f"{elapsed:.3f}"
        st.table(pd.DataFrame(stats_rows.items(), columns=["Metric", "Value"]))

    csv = pd.DataFrame(paths[:200].T, index=t_grid).to_csv().encode()
    st.download_button("Download sample paths (CSV)", csv, "rate_paths.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab 2 — Yield Curve Builder
# --------------------------------------------------------------------------- #
def _classify_curve(mats, ylds) -> str:
    short, long = ylds[0], ylds[-1]
    slope = long - short
    hump = ylds.max() > max(short, long) + 5e-4
    if abs(slope) < 5e-4:
        return "Flat"
    if slope < 0:
        return "Inverted"
    if hump:
        return "Humped"
    return "Normal"


def tab_yield_curve(name, params, r0):
    st.header("Yield Curve Builder")
    mats = np.linspace(0.1, 30.0, 50)
    try:
        ylds, prices = cached_yield_curve(name, tuple(sorted(params.items())), r0, tuple(mats))
    except Exception as exc:
        st.error(f"Yield curve failed: {exc}")
        return

    model = build_model(name, params)
    r0e = effective_r0(model, r0)
    fwd = model.forward_curve(r0e, mats) if not isinstance(model, G2PlusPlusModel) else np.gradient(-np.log(prices), mats)

    shape = _classify_curve(mats, ylds)
    st.markdown(f"**Detected curve shape:** `{shape}`")

    fig = plotting.yield_curve_plot(mats, {"Spot yield": ylds, "Inst. forward": fwd},
                                    title=f"{name}: Spot & Forward Curves")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        df_disc = pd.DataFrame({"maturity": mats, "discount_factor": prices})
        st.plotly_chart(plotting.line_chart(mats, {"Discount factor": prices},
                        title="Discount Factor Curve", xaxis="Maturity", yaxis="P(0,T)"),
                        use_container_width=True)
    with c2:
        st.markdown("#### Cross-model comparison (same r₀)")
        curves = {}
        for nm in ["Vasicek", "CIR", "Hull-White"]:
            try:
                dp = {"kappa": 0.5, "theta": 0.05, "sigma": 0.02} if nm != "Hull-White" else {"kappa": 0.1, "sigma": 0.01}
                mdl = build_model(nm, dp)
                curves[nm] = model_yields(mdl, effective_r0(mdl, r0), mats)
            except Exception:
                pass
        curves["Market"] = np.interp(mats, DEFAULT_MATS, DEFAULT_YLDS)
        st.plotly_chart(plotting.yield_curve_plot(mats, curves, "Model vs Market"),
                        use_container_width=True)

    key_mats = np.array(KEY_TENORS)
    key_ylds = model_yields(model, r0e, key_mats)
    st.markdown("#### Key spot rates")
    st.table(pd.DataFrame({"Tenor (Y)": key_mats, "Spot rate": [f"{y:.3%}" for y in key_ylds]}))


# --------------------------------------------------------------------------- #
# Tab 3 — Bond Pricing
# --------------------------------------------------------------------------- #
def tab_bond_pricing(name, params, r0):
    st.header("Bond Pricing")
    from pricing import CouponBondPricer, ZCBPricer

    c1, c2, c3, c4 = st.columns(4)
    btype = c1.selectbox("Type", ["Coupon", "ZCB"])
    coupon = c2.slider("Coupon rate", 0.0, 0.10, 0.04, 0.005) if btype == "Coupon" else 0.0
    maturity = c3.slider("Maturity (Y)", 1.0, 30.0, 10.0, 1.0)
    freq = c4.selectbox("Frequency", [1, 2, 4], index=1)
    face = 100.0

    try:
        model = build_model(name, params)
        r0e = effective_r0(model, r0)
        if btype == "ZCB":
            price = ZCBPricer().price_analytical(model, r0e, maturity) * face
            ytm = -np.log(price / face) / maturity
            st.metric("Price", f"{price:.4f}")
            st.metric("Yield", f"{ytm:.4%}")
        else:
            bp = CouponBondPricer().price(model, r0e, coupon, maturity, freq, face)
            m1, m2, m3 = st.columns(3)
            m1.metric("Price", f"{bp.price:.4f}")
            m1.metric("YTM", f"{bp.yield_:.4%}")
            m2.metric("Macaulay Dur", f"{bp.duration:.4f}")
            m2.metric("Modified Dur", f"{bp.modified_duration:.4f}")
            m3.metric("Convexity", f"{bp.convexity:.4f}")
            m3.metric("DV01", f"{bp.dv01:.5f}")

            # Price-yield relationship (convexity visualisation).
            ys = np.linspace(max(bp.yield_ - 0.03, 0.001), bp.yield_ + 0.03, 60)
            ps = [CouponBondPricer().price_from_yield(y, coupon, maturity, freq, face) for y in ys]
            st.plotly_chart(plotting.line_chart(ys * 100, {"Price": np.array(ps)},
                            title="Price–Yield Relationship", xaxis="Yield (%)", yaxis="Price"),
                            use_container_width=True)

            # MC vs analytical for a ZCB of the same maturity.
            mc, se, ci = ZCBPricer().price_mc(model, r0e, maturity, n_paths=20_000, n_steps=200, seed=7)
            ana = ZCBPricer().price_analytical(model, r0e, maturity)
            st.caption(f"ZCB MC check: MC={mc:.6f} ± {se:.1e}, analytical={ana:.6f}, "
                       f"deviation={(mc-ana)*1e4:.2f} bps")
    except Exception as exc:
        st.error(f"Pricing failed: {exc}")


# --------------------------------------------------------------------------- #
# Tab 4 — Model Calibration
# --------------------------------------------------------------------------- #
def tab_calibration():
    st.header("Model Calibration")
    from calibration import MLECalibrator, OLSCalibrator
    from diagnostics import ResidualDiagnostics

    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", ["MLE", "OLS"])
    model_choice = c2.selectbox("Model", ["Vasicek", "CIR"])
    src = c3.radio("Data", ["Synthetic", "Upload CSV"], horizontal=True)

    series = None
    if src == "Upload CSV":
        up = st.file_uploader("CSV with one column of short rates", type="csv")
        if up is not None:
            try:
                series = pd.read_csv(up).iloc[:, 0].to_numpy(dtype=float)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
    if series is None:
        true = VasicekParams(0.5, 0.05, 0.02)
        series = VasicekModel(true).simulate_exact(0.04, 20.0, 1, 5040, seed=1).paths[0]
        st.caption("Using synthetic Fed-Funds-like series (Vasicek, κ=0.5, θ=0.05, σ=0.02).")

    dt = 1 / 252
    try:
        cal = OLSCalibrator() if method == "OLS" else MLECalibrator()
        res = cal.calibrate_vasicek(series, dt) if model_choice == "Vasicek" else cal.calibrate_cir(series, dt)
        p = res["params"]
        st.markdown("#### Calibrated parameters")
        st.json({k: round(float(v), 6) for k, v in p.as_dict().items()})
        if "aic" in res:
            st.caption(f"Log-likelihood={res['log_likelihood']:.2f}  AIC={res['aic']:.2f}  BIC={res['bic']:.2f}")

        st.plotly_chart(plotting.line_chart(np.arange(len(series)) * dt, {"Historical": series},
                        title="Observed Short Rate Series", xaxis="Years", yaxis="Rate"),
                        use_container_width=True)

        if model_choice == "Vasicek":
            diag = ResidualDiagnostics()
            resid = diag.compute_residuals(series, VasicekModel(p), dt)
            st.plotly_chart(plotting.histogram_with_density(resid, title="Standardised Residuals",
                            xaxis="Residual"), use_container_width=True)
    except Exception as exc:
        st.warning(f"Calibration issue: {exc}")


# --------------------------------------------------------------------------- #
# Tab 5 — Derivatives Pricing
# --------------------------------------------------------------------------- #
def tab_derivatives():
    st.header("Derivatives Pricing")
    from pricing import BondOptionPricer, CapsFloorsPricer, SwaptionPricer

    product = st.selectbox("Product", ["Bond Option", "Cap", "Floor", "Swaption"])
    hw = HullWhiteModel(HullWhiteParams(0.1, 0.01)).set_initial_curve(DEFAULT_MATS, DEFAULT_YLDS)
    notional = 1_000_000.0

    try:
        if product == "Bond Option":
            v = VasicekModel(VasicekParams(0.5, 0.05, 0.02))
            c1, c2, c3 = st.columns(3)
            Topt = c1.slider("Option expiry", 0.25, 5.0, 1.0, 0.25)
            Tbond = c2.slider("Bond maturity", Topt + 0.5, 10.0, 3.0, 0.25)
            K = c3.slider("Strike", 0.5, 1.0, 0.9, 0.01)
            call = BondOptionPricer().european_call_zcb(v, 0.03, Topt, Tbond, K)
            put = BondOptionPricer().european_put_zcb(v, 0.03, Topt, Tbond, K)
            m1, m2 = st.columns(2)
            m1.metric("Call (Vasicek)", f"{call:.6f}")
            m2.metric("Put (Vasicek)", f"{put:.6f}")
            rT = np.linspace(0.0, 0.1, 100)
            payoff = np.maximum(np.array([v.zcb_price(r, Tbond, Topt) for r in rT]) - K, 0)
            st.plotly_chart(plotting.line_chart(rT * 100, {"Call payoff": payoff},
                            title="Payoff at expiry", xaxis="r(T_opt) (%)", yaxis="Payoff"),
                            use_container_width=True)
        elif product in ("Cap", "Floor"):
            c1, c2 = st.columns(2)
            rate = c1.slider("Strike rate", 0.0, 0.08, 0.03, 0.005)
            mat = c2.slider("Maturity", 1.0, 10.0, 5.0, 1.0)
            if product == "Cap":
                price = CapsFloorsPricer().cap_price(hw, rate, mat, notional, 4)
            else:
                price = CapsFloorsPricer().floor_price(hw, rate, mat, notional, 4)
            st.metric(f"{product} price (HW)", f"{price:,.2f}")
            parity = CapsFloorsPricer().cap_floor_parity_check(hw, rate, mat, notional, 4)
            st.caption(f"Cap−Floor−Swap residual: {parity['difference']:.2e} (parity {'holds' if parity['passes'] else 'fails'})")
        else:  # Swaption
            c1, c2, c3 = st.columns(3)
            swap_rate = c1.slider("Swap rate", 0.0, 0.08, 0.035, 0.005)
            tenor = c2.slider("Swap tenor", 1.0, 10.0, 5.0, 1.0)
            Topt = c3.slider("Expiry", 0.5, 5.0, 1.0, 0.5)
            payer = SwaptionPricer().payer_swaption_hw(hw, swap_rate, tenor, Topt, notional)
            receiver = SwaptionPricer().receiver_swaption_hw(hw, swap_rate, tenor, Topt, notional)
            m1, m2 = st.columns(2)
            m1.metric("Payer swaption (HW)", f"{payer:,.2f}")
            m2.metric("Receiver swaption (HW)", f"{receiver:,.2f}")
    except Exception as exc:
        st.error(f"Derivative pricing failed: {exc}")


# --------------------------------------------------------------------------- #
# Tab 6 — Risk Dashboard
# --------------------------------------------------------------------------- #
def tab_risk(name, params, r0):
    st.header("Risk Dashboard")
    from risk import DV01Engine, InterestRateVaR
    st.markdown("#### Portfolio")
    default = pd.DataFrame({
        "coupon_rate": [0.04, 0.05], "maturity": [5.0, 10.0],
        "freq": [2, 2], "face": [100.0, 100.0], "weight": [0.5, 0.5],
    })
    edited = st.data_editor(default, num_rows="dynamic", use_container_width=True)

    try:
        model = build_model(name, params)
        r0e = effective_r0(model, r0)
        bonds, weights = [], []
        for _, row in edited.iterrows():
            bonds.append({"type": "coupon", "coupon_rate": float(row["coupon_rate"]),
                          "maturity": float(row["maturity"]), "freq": int(row["freq"]),
                          "face": float(row["face"])})
            weights.append(float(row["weight"]))
        engine = DV01Engine()
        total_dv01 = engine.dv01_portfolio(model, r0e, bonds, weights)
        st.metric("Portfolio DV01", f"{total_dv01:.5f}")

        krd = engine.key_rate_durations(model, r0e, bonds[0])
        st.plotly_chart(plotting.bar_chart([f"{k:g}Y" for k in krd], list(krd.values()),
                        title="Key-Rate Durations (first bond)", yaxis="KRD"),
                        use_container_width=True)

        var = InterestRateVaR()
        pnl = var.pnl_distribution_data(model, r0e, bonds[0], 1.0, n_paths=20_000, seed=9)
        res = var.bond_var(model, r0e, bonds[0], 1.0, 0.99, n_paths=20_000, seed=9)
        fig = plotting.histogram_with_density(pnl, title="1Y P&L distribution (first bond)", xaxis="P&L")
        try:
            fig.add_vline(x=-res["var"], line=dict(color="#E8536B", dash="dash"),
                          annotation_text="VaR 99%")
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"VaR(99%)={res['var']:.4f}   CVaR(99%)={res['cvar']:.4f}")
    except Exception as exc:
        st.error(f"Risk computation failed: {exc}")


# --------------------------------------------------------------------------- #
# Tab 7 — Scenario Analysis
# --------------------------------------------------------------------------- #
def tab_scenarios(name, params, r0):
    st.header("Scenario Analysis")
    from scenarios import ParallelShiftScenario, RateStressTester, TwistScenario
    model = build_model(name, params)
    r0e = effective_r0(model, r0)
    bond = {"type": "coupon", "coupon_rate": 0.04, "maturity": 10, "freq": 2, "face": 100}

    stype = st.selectbox("Scenario", ["Parallel Shift", "Twist", "Historical"])
    try:
        if stype == "Parallel Shift":
            shift = st.slider("Parallel shift (bps)", -300, 300, 100, 25)
            df = ParallelShiftScenario().run(model, r0e, bond, shifts_bps=[shift])
            row = df.iloc[0]
            st.metric("Bond price", f"{row['bond_price']:.4f}", f"{row['pnl']:.4f}")
            st.dataframe(ParallelShiftScenario().run(model, r0e, bond), use_container_width=True)
        elif stype == "Twist":
            c1, c2 = st.columns(2)
            short = c1.slider("Short-end shift (bps)", -150, 150, -50, 10)
            long = c2.slider("Long-end shift (bps)", -150, 150, 50, 10)
            res = TwistScenario().steepener(model, r0e, bond, short, long)
            st.metric("Bond price", f"{res['price']:.4f}", f"{res['pnl']:.4f}")
        else:
            tester = RateStressTester()
            rows = []
            for sc in tester.SCENARIOS:
                r = tester.apply_scenario(model, r0e, bond, sc)
                rows.append({"Scenario": sc, "Price": r["price"], "P&L": r["pnl"], "%": r["pct"]})
            sdf = pd.DataFrame(rows)
            st.dataframe(sdf, use_container_width=True)
            st.plotly_chart(plotting.bar_chart(sdf["Scenario"], sdf["P&L"],
                            title="Historical Stress P&L", yaxis="P&L"), use_container_width=True)
    except Exception as exc:
        st.error(f"Scenario failed: {exc}")


# --------------------------------------------------------------------------- #
# Tab 8 — Model Documentation
# --------------------------------------------------------------------------- #
def tab_docs():
    st.header("Model Documentation")
    comp = pd.DataFrame({
        "SDE": ["dr=κ(θ−r)dt+σdW", "dr=κ(θ−r)dt+σ√r dW",
                "dr=(θ(t)−κr)dt+σdW", "r=x+y+φ(t)"],
        "Analytical ZCB": ["Yes (affine)", "Yes (affine)", "Yes (curve-fit)", "Yes (2-factor)"],
        "Negative rates": ["Yes", "No (Feller)", "Yes", "Yes"],
        "Fits market": ["Approx.", "Approx.", "Exact", "Exact"],
        "Complexity": ["Low", "Low", "Medium", "High"],
        "Use case": ["Teaching/Gaussian", "Positive rates", "Derivatives", "Multi-factor risk"],
    }, index=["Vasicek", "CIR", "Hull-White", "G2++"])
    st.table(comp)

    st.subheader("Stochastic differential equations")
    st.latex(r"dr(t) = \kappa(\theta - r(t))\,dt + \sigma\,dW(t) \quad \text{(Vasicek)}")
    st.latex(r"dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t) \quad \text{(CIR)}")
    st.latex(r"dr(t) = (\theta(t) - \kappa r(t))\,dt + \sigma\,dW(t) \quad \text{(Hull-White)}")
    st.latex(r"P(t,T) = A(t,T)\,e^{-B(t,T)\,r(t)}, \quad B(t,T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}")

    st.subheader("Feller condition explorer (CIR)")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("κ", 0.01, 2.0, 0.5, 0.01, key="feller_k")
    th = c2.slider("θ", 0.001, 0.10, 0.05, 0.001, key="feller_th")
    sig = c3.slider("σ", 0.001, 0.30, 0.10, 0.001, key="feller_s")
    if 2 * k * th > sig ** 2:
        st.success(f"Feller holds: 2κθ = {2*k*th:.4f} > σ² = {sig**2:.4f} → rates stay strictly positive")
    else:
        st.error(f"Feller violated: 2κθ = {2*k*th:.4f} ≤ σ² = {sig**2:.4f} → rates can hit zero")

    st.subheader("Half-life of mean reversion")
    kk = st.slider("κ", 0.01, 3.0, 0.5, 0.01, key="hl_k")
    st.latex(r"\text{Half-life} = \frac{\ln 2}{\kappa} = %.3f \text{ years}" % (np.log(2) / kk))

    st.subheader("References")
    st.markdown(
        "- Vasicek (1977), *JFE* 5(2)\n"
        "- Cox, Ingersoll & Ross (1985), *Econometrica* 53(2)\n"
        "- Hull & White (1990), *RFS* 3(4)\n"
        "- Brigo & Mercurio (2006), *Interest Rate Models — Theory and Practice*"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("📈 Stochastic Rate Models")
    st.caption("Vasicek · CIR · Hull-White · G2++ — simulation, pricing, calibration, risk & scenarios")
    st.sidebar.title("Controls")

    # One shared model control set, reused by the model-driven tabs.
    name, params, r0 = sidebar_model_inputs()

    tabs = st.tabs([
        "1 · Simulator", "2 · Yield Curve", "3 · Bond Pricing", "4 · Calibration",
        "5 · Derivatives", "6 · Risk", "7 · Scenarios", "8 · Docs",
    ])
    with tabs[0]:
        tab_simulator(name, params, r0)
    with tabs[1]:
        tab_yield_curve(name, params, r0)
    with tabs[2]:
        tab_bond_pricing(name, params, r0)
    with tabs[3]:
        tab_calibration()
    with tabs[4]:
        tab_derivatives()
    with tabs[5]:
        tab_risk(name, params, r0)
    with tabs[6]:
        tab_scenarios(name, params, r0)
    with tabs[7]:
        tab_docs()


if __name__ == "__main__":
    main()
