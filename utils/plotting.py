"""Reusable Plotly chart builders for the dashboard and notebooks.

All functions return :class:`plotly.graph_objects.Figure` instances using a
consistent dark, institutional colour palette so the Streamlit app has a
coherent look and feel.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:  # Plotly is optional for headless / test environments.
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotting is a soft dependency
    go = None  # type: ignore[assignment]

# Institutional colour palette.
PALETTE = {
    "primary": "#4C9BE8",
    "accent": "#F2A93B",
    "green": "#37C978",
    "red": "#E8536B",
    "grey": "#8A93A6",
    "bg": "#0E1117",
    "panel": "#1A1D26",
}

MODEL_COLORS = {
    "Vasicek": "#4C9BE8",
    "CIR": "#37C978",
    "Hull-White": "#F2A93B",
    "G2++": "#B06BE8",
    "Market": "#E8536B",
}


def _require_plotly():
    if go is None:  # pragma: no cover
        raise ImportError("plotly is required for plotting utilities")


def _base_layout(fig, title: str, xaxis: str, yaxis: str):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(family="Inter, Arial", size=13, color="#D6DAE3"),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def fan_chart(
    t_grid: np.ndarray,
    paths: np.ndarray,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
    n_sample_paths: int = 30,
    mean_path: Optional[np.ndarray] = None,
    title: str = "Simulated Short Rate Paths",
    theta: Optional[float] = None,
):
    """Build a fan chart with percentile bands and sample paths overlaid."""
    _require_plotly()
    fig = go.Figure()

    pct = {p: np.percentile(paths, p, axis=0) for p in percentiles}
    lo, hi = min(percentiles), max(percentiles)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_grid, t_grid[::-1]]),
            y=np.concatenate([pct[hi], pct[lo][::-1]]),
            fill="toself",
            fillcolor="rgba(76,155,232,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{lo}-{hi} pct",
            hoverinfo="skip",
        )
    )
    if 25 in pct and 75 in pct:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t_grid, t_grid[::-1]]),
                y=np.concatenate([pct[75], pct[25][::-1]]),
                fill="toself",
                fillcolor="rgba(76,155,232,0.30)",
                line=dict(color="rgba(0,0,0,0)"),
                name="25-75 pct",
                hoverinfo="skip",
            )
        )

    rng = np.random.default_rng(0)
    idx = rng.choice(paths.shape[0], size=min(n_sample_paths, paths.shape[0]), replace=False)
    for j in idx:
        fig.add_trace(
            go.Scatter(
                x=t_grid,
                y=paths[j],
                line=dict(color="rgba(138,147,166,0.25)", width=0.7),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    mp = mean_path if mean_path is not None else paths.mean(axis=0)
    fig.add_trace(
        go.Scatter(x=t_grid, y=mp, line=dict(color=PALETTE["accent"], width=2.5), name="Mean")
    )
    if theta is not None:
        fig.add_hline(
            y=theta, line=dict(color=PALETTE["red"], dash="dash"),
            annotation_text="theta", annotation_position="top left",
        )
    return _base_layout(fig, title, "Time (years)", "Short rate")


def yield_curve_plot(
    maturities: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str = "Yield Curve",
):
    """Overlay one or more yield curves keyed by label."""
    _require_plotly()
    fig = go.Figure()
    for label, ys in curves.items():
        fig.add_trace(
            go.Scatter(
                x=maturities,
                y=np.asarray(ys) * 100,
                mode="lines+markers",
                name=label,
                line=dict(color=MODEL_COLORS.get(label, PALETTE["primary"]), width=2.2),
            )
        )
    return _base_layout(fig, title, "Maturity (years)", "Yield (%)")


def histogram_with_density(
    samples: np.ndarray,
    density_x: Optional[np.ndarray] = None,
    density_y: Optional[np.ndarray] = None,
    title: str = "Distribution",
    xaxis: str = "Value",
):
    """Histogram of samples with an optional fitted density overlay."""
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=samples, histnorm="probability density", name="Empirical",
            marker_color=PALETTE["primary"], opacity=0.65, nbinsx=60,
        )
    )
    if density_x is not None and density_y is not None:
        fig.add_trace(
            go.Scatter(
                x=density_x, y=density_y, name="Fitted",
                line=dict(color=PALETTE["accent"], width=2.5),
            )
        )
    return _base_layout(fig, title, xaxis, "Density")


def bar_chart(
    categories: Sequence[str],
    values: Sequence[float],
    title: str = "",
    yaxis: str = "Value",
    color: Optional[str] = None,
):
    """Simple labelled bar chart."""
    _require_plotly()
    fig = go.Figure(
        go.Bar(x=list(categories), y=list(values), marker_color=color or PALETTE["primary"])
    )
    return _base_layout(fig, title, "", yaxis)


def line_chart(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    title: str = "",
    xaxis: str = "x",
    yaxis: str = "y",
):
    """Generic multi-series line chart."""
    _require_plotly()
    fig = go.Figure()
    for label, y in series.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
    return _base_layout(fig, title, xaxis, yaxis)


def waterfall(
    labels: Sequence[str],
    values: Sequence[float],
    title: str = "P&L Attribution",
):
    """P&L waterfall chart."""
    _require_plotly()
    fig = go.Figure(
        go.Waterfall(
            x=list(labels),
            y=list(values),
            connector=dict(line=dict(color=PALETTE["grey"])),
            increasing=dict(marker=dict(color=PALETTE["green"])),
            decreasing=dict(marker=dict(color=PALETTE["red"])),
        )
    )
    return _base_layout(fig, title, "", "P&L")
