import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from skyapp.core import MODEL_KEYS, MODEL_LABELS, auc, load_config, load_static, load_test_predictions, smooth
from skyapp.ui import dark_layout

cfg = load_config()
stat = load_static()
FINE = cfg["fine_index"]
df = load_test_predictions()
y = (df.label == FINE).to_numpy()
ALL = ["ensemble"] + MODEL_KEYS + ["handcrafted"]

st.title("📊 Results")
st.markdown(
    f"Evaluation on **{len(df):,} frames from {df.night.nunique()} test nights** that no model saw during "
    "training or model selection. The final model (the ensemble) was chosen on the validation nights."
)
thr = st.slider("Decision threshold on P(clear)", 0.1, 0.9, 0.5, 0.05,
                help="All numbers below update with the threshold. AUC does not depend on it.")

# ---------------------------------------------------------------- summary table
nights = sorted(df.night.unique())
rows = []
for k in ALL:
    p = df[f"p_{k}"].to_numpy(); pred = p >= thr
    per = np.array([(pred[df.night == n] == y[df.night == n]).mean() for n in nights])
    rows.append({"Model": MODEL_LABELS[k], "AUC": auc(y, p), "Accuracy": (pred == y).mean(),
                 "Recall clear": pred[y].mean(), "Recall cloudy": (~pred[~y]).mean(),
                 "Per-night mean": per.mean(), "Per-night sd": per.std(), "Worst night": per.min()})
summary = pd.DataFrame(rows)
st.subheader("Model comparison")
st.dataframe(summary.style.format({c: "{:.3f}" for c in summary.columns if c != "Model"})
             .highlight_max(subset=["AUC", "Accuracy", "Per-night mean", "Worst night"], color="#2fb37a55")
             .highlight_min(subset=["Per-night sd"], color="#2fb37a55"),
             hide_index=True, width="stretch")
st.caption("Recall clear = share of fine frames recognised as fine; recall cloudy likewise. A low per-night "
           "standard deviation means the model is equally reliable from one night to the next.")

# ---------------------------------------------------------------- per night
st.subheader("Accuracy per test night")
mat = pd.DataFrame({MODEL_LABELS[k]: [((df[f"p_{k}"] >= thr)[df.night == n] == y[df.night == n]).mean()
                                      for n in nights] for k in ALL}, index=nights).T
fig = px.imshow(mat, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=0.3, zmax=1, aspect="auto")
fig.update_xaxes(title=None, type="category"); fig.update_yaxes(title=None)
st.plotly_chart(dark_layout(fig, height=300), width="stretch")
moon = df.groupby("night").agg(illum=("moon_illum", "mean"), up=("moon_alt", lambda a: (a > 0).mean()),
                               cloudy=("label", lambda s: (s != FINE).sum()), fine=("label", lambda s: (s == FINE).sum()))
st.dataframe(moon.rename(columns={"illum": "Moon illumination", "up": "Frames with moon up",
                                  "cloudy": "Cloudy frames", "fine": "Fine frames"})
             .style.format({"Moon illumination": "{:.0%}", "Frames with moon up": "{:.0%}"}), width="stretch")
st.caption("Different models fail on different nights (e.g. ResNet-18 v2 on the full-moon night 12-19, v3 on 11-29), "
           "which is why averaging them helps.")

# ---------------------------------------------------------------- moon effect + smoothing
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("The moon effect")
    bins = pd.cut(df.moon_alt, [-90, 0, 20, 40, 90], labels=["below horizon", "0–20°", "20–40°", ">40°"])
    long = []
    for k in ["ensemble", "smallcnn", "resnet_v2", "handcrafted"]:
        ok = (df[f"p_{k}"] >= thr).to_numpy() == y
        for b, v in pd.Series(ok).groupby(bins.to_numpy(), observed=False).mean().items():
            long.append({"Moon altitude": b, "Accuracy": v, "Model": MODEL_LABELS[k]})
    fig = px.bar(pd.DataFrame(long), x="Moon altitude", y="Accuracy", color="Model", barmode="group",
                 category_orders={"Moon altitude": ["below horizon", "0–20°", "20–40°", ">40°"]})
    fig.update_yaxes(range=[0.5, 1])
    st.plotly_chart(dark_layout(fig, height=360), width="stretch")
    st.caption("All models do best with the moon below the horizon. The ensemble varies least across moon "
               "conditions, hand-crafted brightness features the most. Moon positions computed with astropy.")
with c2:
    st.subheader("Temporal smoothing")
    wins = [1, 3, 5, 7, 9, 11, 15, 21]
    sm = []
    for w in wins:
        for mode, centred in [("centred", True), ("trailing (real-time)", False)]:
            ps = smooth(df, "p_ensemble", w, centred)
            sm.append({"Window (frames ≈ min)": w, "Accuracy": ((ps >= thr).to_numpy() == y).mean(), "Mode": mode})
    fig = px.line(pd.DataFrame(sm), x="Window (frames ≈ min)", y="Accuracy", color="Mode", markers=True)
    st.plotly_chart(dark_layout(fig, height=360), width="stretch")
    st.caption("A rolling median over neighbouring frames gives only ~1 point: errors come in long runs (a "
               "systematic condition such as moonlight or a mislabelled block), not as isolated frames.")

# ---------------------------------------------------------------- confusion matrices
st.subheader("Confusion matrices")
cols = st.columns(len(ALL))
for col, k in zip(cols, ALL):
    pred = (df[f"p_{k}"] >= thr).to_numpy()
    cm = np.array([[(~y & ~pred).sum(), (~y & pred).sum()], [(y & ~pred).sum(), (y & pred).sum()]])
    fig = px.imshow(cm, text_auto=True, x=["→ cloudy", "→ fine"], y=["cloudy", "fine"],
                    color_continuous_scale="Blues", aspect="equal")
    fig.update_layout(coloraxis_showscale=False, title=dict(text=MODEL_LABELS[k], font=dict(size=13)))
    col.plotly_chart(dark_layout(fig, height=240), width="stretch", config={"displayModeBar": False})
st.caption("Rows: dataset label. Columns: prediction. For an observatory, 'cloudy → fine' (observing "
           "through clouds) and 'fine → cloudy' (lost observing time) have different costs.")

# ---------------------------------------------------------------- training
st.subheader("Training: how much of the network should learn?")
logs = dict(stat["training_logs"])
for k, h in cfg.get("history", {}).items():
    logs[{"smallcnn": "Small CNN (from scratch)", "resnet_v3": "ResNet-18 v3: layer4 + regularisation"}.get(k, k)] = {
        "val_loss": h["va_loss"], "train_loss": h["tr_loss"], "val_acc": h["va_acc"], "train_acc": h["tr_acc"]}
pick = st.multiselect("Runs", list(logs), default=list(logs))
fig = go.Figure()
for name in pick:
    h = logs[name]
    fig.add_trace(go.Scatter(y=h["val_loss"], mode="lines+markers", name=f"{name} — val"))
fig.update_layout(xaxis_title="epoch (all stages in order)", yaxis_title="validation loss")
st.plotly_chart(dark_layout(fig, height=380), width="stretch")
st.caption(stat["training_note"])

# ---------------------------------------------------------------- baselines + validation
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Hand-crafted baseline")
    st.dataframe(pd.DataFrame(stat["feature_auc"]), hide_index=True, width="stretch")
    st.caption("Single physical features measured on the sky pixels only. Star count is best; brightness is "
               "weakest because the moon brightens clear and cloudy skies alike.")
with c2:
    st.subheader("Validation (model selection)")
    st.dataframe(pd.DataFrame(stat["validation"]), hide_index=True, width="stretch")
    st.caption("Both ensembles beat every single model on validation, so the decision to ensemble was made "
               "before looking at the test nights.")

st.subheader("What limits performance")
st.markdown(stat["limitations_md"])
