import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from skyapp.core import load_config, load_static, load_tsne, thumb_path
from skyapp.ui import dark_layout

cfg = load_config()
stat = load_static()
FINE = cfg["fine_index"]
names = cfg["cluster_names"]
df = load_tsne().copy()
df["true label"] = [cfg["classes"][i].replace("Data_", "") for i in df.label]
df["sky type"] = [names[i] for i in df.sky_type]
df["GMM cluster"] = ["fine-like" if g == cfg["gmm_fine_cluster"] else "cloudy-like" for g in df.gmm2]

DESCRIPTIONS = stat["cluster_descriptions"]
COLORS = ["#2b3a67", "#c2703d", "#7fb3ff", "#e0524f", "#d9d9d9"]

st.title("🧭 Explore sky types")
st.markdown(
    "Each dot is one sky image, placed with **t-SNE** so that images the network finds similar sit close "
    "together. The positions come from ImageNet ResNet-18 features — **no labels were used**. "
    "Colour the map in different ways, then **select dots** (click, box or lasso) to see the images."
)

colour = st.radio("Colour by", ["sky type", "true label", "GMM cluster", "brightness", "night"], horizontal=True)
kw = dict(x="x", y="y", custom_data=["stem"], render_mode="svg", hover_data={"x": False, "y": False, "night": True,
                                                            "true label": True, "sky type": True})
if colour == "sky type":
    fig = px.scatter(df, color="sky type", category_orders={"sky type": names},
                     color_discrete_sequence=COLORS, **kw)
elif colour == "brightness":
    fig = px.scatter(df, color="brightness", color_continuous_scale="magma", **kw)
else:
    fig = px.scatter(df, color=colour, **kw)
fig.update_traces(marker=dict(size=4, opacity=0.75))
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
ev = st.plotly_chart(dark_layout(fig, height=560), width="stretch", on_select="rerun",
                     selection_mode=("points", "box", "lasso"), key="tsne")

pts = ev.selection.points if ev and ev.selection else []
if pts:
    stems = []
    for p in pts:
        cd = p.get("customdata")
        if cd:
            stems.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
        else:                                            # fall back to the nearest dot
            j = int(((df.x - p["x"]) ** 2 + (df.y - p["y"]) ** 2).idxmin()); stems.append(df.stem[j])
    sel = df.set_index("stem").loc[[s for s in dict.fromkeys(stems)]]
    st.markdown(f"**{len(sel)} selected** — " + ", ".join(f"{k}: {v}" for k, v in sel["sky type"].value_counts().items()))
    cols = st.columns(8)
    for i, (stem, r) in enumerate(sel.head(24).iterrows()):
        tp = thumb_path(stem, "tsne")
        if tp:
            cols[i % 8].image(str(tp), caption=f"{r['true label']} · {stem[:6]}", width="stretch")
else:
    st.caption("Tip: drag a box around a group of dots to see what the network put together.")

st.divider()
st.subheader("The five sky types")
tab = pd.crosstab(df["sky type"], df["true label"]).reindex(names)
rng = np.random.default_rng(0)
for c, name in enumerate(names):
    sub = df[df.sky_type == c]
    n_f = int((sub.label == FINE).sum()); n_c = len(sub) - n_f
    with st.container(border=True):
        a, b = st.columns([1, 2.2])
        with a:
            st.markdown(f"#### {name}")
            st.markdown(f"{n_c} cloudy · {n_f} fine · **{max(n_f, n_c) / max(len(sub), 1):.0%} pure**")
            st.caption(DESCRIPTIONS[c])
        with b:
            picks = rng.choice(sub.stem.to_numpy(), size=min(6, len(sub)), replace=False) if len(sub) else []
            cols = st.columns(6)
            for col, s in zip(cols, picks):
                tp = thumb_path(s, "tsne")
                if tp:
                    col.image(str(tp), width="stretch")

st.divider()
st.subheader("How well do clusters match the labels?")
st.markdown("Scores for 4,000 images. **ARI** 0 = random, 1 = identical to the labels; **purity** is the share "
            "of images in their cluster's majority class (0.56 = chance).")
st.dataframe(pd.DataFrame(stat["clustering"]), hide_index=True, width="stretch")
st.caption("Hand-crafted brightness statistics form clusters that split bright from dark skies (moonlight), "
           "unrelated to clouds (ARI ≈ 0). CNN features recover clear vs cloudy without labels "
           "(GMM: ARI 0.55, purity 87%).")
