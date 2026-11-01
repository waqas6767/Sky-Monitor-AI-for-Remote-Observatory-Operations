"""Shared look-and-feel helpers."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .core import CATEGORIES

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.verdict {border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;
          border: 1px solid rgba(255,255,255,.12);}
.verdict .cat {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.verdict .adv {font-size: 1rem; opacity: .9; margin-top: .25rem;}
.card {border: 1px solid rgba(255,255,255,.12); border-radius: 12px; padding: .9rem 1rem;
       background: rgba(255,255,255,.03); height: 100%;}
.card h4 {margin: 0 0 .3rem 0;}
.small {font-size: .85rem; opacity: .8;}
.pill {display:inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem;
       font-weight: 600; color: #0b1020;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def verdict_card(category: str, extra: str = "") -> None:
    c = CATEGORIES[category]
    st.markdown(
        f"""<div class="verdict" style="background: {c['color']}22; border-color: {c['color']}88;">
        <div class="cat">{c['icon']} {category}</div>
        <div class="adv">{c['advice']}</div>
        {f'<div class="small" style="margin-top:.4rem">{extra}</div>' if extra else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'


def evidence_figure(display: np.ndarray, fine_map: np.ndarray, mask: np.ndarray, opacity: float,
                    height: int = 460) -> go.Figure:
    """Sky image with the ensemble evidence map on top (red = looks clear, blue = looks cloudy)."""
    z = np.where(mask, fine_map, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Image(z=display, hoverinfo="skip"))
    if opacity > 0:
        fig.add_trace(go.Heatmap(
            z=z, zmin=0, zmax=1, colorscale="RdBu_r", opacity=opacity,
            colorbar=dict(title=dict(text="local<br>P(clear)", side="right"), thickness=12, len=.8),
            hovertemplate="local P(clear) = %{z:.2f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x", autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def dark_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    return fig
