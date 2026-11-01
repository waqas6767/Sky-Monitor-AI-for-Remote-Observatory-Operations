import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skyapp import sources as src
from skyapp.core import (CATEGORIES, DEFAULT_THR, MODEL_KEYS, MODEL_LABELS, analyse_items, load_config,
                         load_test_predictions, sky_verdict, smooth, thumb_path)
from skyapp.ui import dark_layout

cfg = load_config()
FINE = cfg["fine_index"]
MAX_N = 600
st.title("🌙 Night timeline & batch analysis")
st.caption("Classify a whole set of frames — a test night, your own uploads, a Google Drive folder or a folder on "
           "this computer — and follow the sky minute by minute.")

source = st.radio("Images from", ["Test nights", "Upload files / zip", "Google Drive link", "Folder on this computer"],
                  horizontal=True)
ss = st.session_state


def run_batch(items, key):
    bar = st.progress(0.0, "Starting…")
    df = analyse_items(items, progress=lambda f, t: bar.progress(min(f, 1.0), t))
    bar.empty()
    if df.empty:
        st.error("None of the files could be read as images.")
        st.stop()
    ss["batch_key"], ss["batch_df"], ss["batch_imgs"] = key, df, {n.rsplit(".", 1)[0]: b for n, b in items}


# ---------------------------------------------------------------- choose / compute the data
has_labels = False
if source == "Test nights":
    allp = load_test_predictions()
    nights = sorted(allp.night.unique())
    comp = allp.groupby("night").label.agg(lambda s: f"{(s != FINE).sum()} cloudy / {(s == FINE).sum()} fine")
    night = st.selectbox("Night", nights, index=nights.index("2021-11-24") if "2021-11-24" in nights else 0,
                         format_func=lambda n: f"{n}  ({comp[n]})")
    df = allp[allp.night == night].copy()
    has_labels = True

elif source == "Upload files / zip":
    files = st.file_uploader("Frames (JPG/PNG) or one .zip of frames", type=["jpg", "jpeg", "png", "zip"],
                             accept_multiple_files=True)
    if files:
        key = ("up",) + tuple((f.name, f.size) for f in files)
        if ss.get("batch_key") != key and st.button("Analyse", type="primary"):
            items = []
            for f in files:
                items += src.images_from_zip(f.getvalue(), MAX_N) if f.name.lower().endswith(".zip") \
                    else [(f.name, f.getvalue())]
            run_batch(items[:MAX_N], key)

elif source == "Google Drive link":
    st.markdown("Paste a link to a Drive **folder** or a **.zip file**, shared as *Anyone with the link can view*. "
                "Subfolders (e.g. `Data_fine`, `Data_cloudy`) are included.")
    link = st.text_input("Google Drive link", placeholder="https://drive.google.com/drive/folders/…")
    kind, fid = src.drive_id(link) if link else (None, None)
    if link and not kind:
        st.error("That doesn't look like a Google Drive folder or file link.")
    elif kind == "folder":
        if st.button("List images"):
            with st.spinner("Reading the folder listing…"):
                try:
                    ss["drive_list"] = (fid, src.list_drive_folder(fid))
                except Exception as e:
                    st.error(f"Could not read the folder ({e}). Is it shared as 'Anyone with the link'?")
        if ss.get("drive_list", (None,))[0] == fid:
            files = ss["drive_list"][1]
            if not files:
                st.warning("No images found in this folder.")
            else:
                folders = sorted({f["folder"] for f in files})
                st.success(f"Found {len(files):,} images" + (f" in {len(folders)} folders" if len(folders) > 1 else ""))
                c1, c2, c3 = st.columns(3)
                which = c1.selectbox("Folder", ["(all)"] + folders) if len(folders) > 1 else "(all)"
                sel = sorted([f for f in files if which == "(all)" or f["folder"] == which], key=lambda f: f["name"])
                n = c2.number_input("How many frames", 1, min(MAX_N, len(sel)), min(120, len(sel)))
                part = c3.selectbox("Which", ["latest (by name)", "first (by name)", "evenly spread"])
                if part.startswith("latest"):
                    sel = sel[-n:]
                elif part.startswith("first"):
                    sel = sel[:n]
                else:
                    sel = [sel[i] for i in np.linspace(0, len(sel) - 1, n).astype(int)]
                if st.button(f"Download & analyse {len(sel)} frames", type="primary"):
                    items, bar = [], st.progress(0.0, "Downloading…")
                    for i, f in enumerate(sel):
                        try:
                            items.append((f["name"], src.download_drive_file(f["id"])))
                        except Exception as e:
                            st.warning(f"Skipped {f['name']}: {e}")
                        bar.progress((i + 1) / len(sel), f"Downloading {i + 1}/{len(sel)}…")
                    bar.empty()
                    run_batch(items, ("drive", fid, which, n, part))
    elif kind == "file":
        if st.button("Download & analyse", type="primary"):
            with st.spinner("Downloading…"):
                try:
                    data = src.download_drive_file(fid)
                except Exception as e:
                    st.error(str(e)); st.stop()
            items = src.images_from_zip(data, MAX_N) if data[:2] == b"PK" else [(f"{fid}.jpg", data)]
            run_batch(items, ("drivefile", fid))

else:
    st.markdown("Works when the app runs **on your own computer**, e.g. the folder where the camera software saves "
                "its frames.")
    folder = st.text_input("Folder path", placeholder=r"C:\AllSky\images   or   /home/user/allsky/images")
    if folder:
        imgs = src.list_local_images(folder)
        if not imgs:
            st.error("No images found there (or the folder does not exist).")
        else:
            n = st.number_input(f"Found {len(imgs):,} images — analyse the latest", 1, min(MAX_N, len(imgs)),
                                min(120, len(imgs)))
            if st.button("Analyse", type="primary"):
                sel = imgs[-n:]
                run_batch([(p.name, p.read_bytes()) for p in sel], ("local", folder, n))

if source != "Test nights":
    if "batch_df" not in ss:
        st.stop()
    df = ss["batch_df"].copy()
    if df.attrs.get("synthetic_time") or ss["batch_df"].attrs.get("synthetic_time"):
        st.info("File names have no timestamp (YYMMDDhhmmss), so frames are shown in file-name order.")

# ---------------------------------------------------------------- summary of the conditions
counts = df.assign(category=[sky_verdict(p, c, s, cfg, DEFAULT_THR) for p, c, s in
                             zip(df.p_ensemble, df.cloud_cover, df.sky_type)]).category.value_counts()
cols = st.columns(len(CATEGORIES))
for col, (k, v) in zip(cols, CATEGORIES.items()):
    col.metric(f"{v['icon']} {k}", f"{counts.get(k, 0)}", f"{counts.get(k, 0) / len(df):.0%} of frames",
               delta_color="off")

# ---------------------------------------------------------------- controls
c1, c2, c3, c4 = st.columns(4)
model = c1.selectbox("Model", ["ensemble"] + MODEL_KEYS + (["handcrafted"] if has_labels else []),
                     format_func=lambda k: MODEL_LABELS[k])
window = c2.slider("Smoothing window (frames ≈ minutes)", 1, 21, 5, 2)
centred = c3.radio("Window", ["centred", "trailing (real-time)"], horizontal=True) == "centred"
thr = c4.slider("Decision threshold on P(clear)", 0.1, 0.9, 0.5, 0.05)

col = f"p_{model}"
df["p_s"] = smooth(df, col, window, centred).to_numpy()
df["cover_s"] = smooth(df, "cloud_cover", window, centred).to_numpy()
df["category"] = [sky_verdict(p, c, s, cfg, DEFAULT_THR) for p, c, s in zip(df.p_s, df.cover_s, df.sky_type)]

if has_labels:
    y = (df.label == FINE).to_numpy()
    a_raw, a_s = ((df[col] >= thr) == y).mean(), ((df.p_s >= thr) == y).mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Frames", len(df))
    k2.metric("Agreement with labels (raw)", f"{a_raw:.1%}")
    k3.metric("Agreement after smoothing", f"{a_s:.1%}", f"{(a_s - a_raw) * 100:+.1f} pts")
    k4.metric("Moon above horizon", f"{(df.moon_alt > 0).mean():.0%} of frames")

# ---------------------------------------------------------------- figure
fig = go.Figure()
t = df.time
runs = (df.category != df.category.shift()).cumsum()
for _, g in df.groupby(runs):
    fig.add_vrect(x0=g.time.iloc[0] - pd.Timedelta(seconds=30), x1=g.time.iloc[-1] + pd.Timedelta(seconds=30),
                  fillcolor=CATEGORIES[g.category.iloc[0]]["color"], opacity=0.13, line_width=0, layer="below")
fig.add_trace(go.Scatter(x=t, y=df[col], mode="markers", name=f"P(clear), {MODEL_LABELS[model]}",
                         marker=dict(size=4, color="#5b8def", opacity=0.55)))
fig.add_trace(go.Scatter(x=t, y=df.p_s, mode="lines", name=f"smoothed ({window} frames)",
                         line=dict(color="#5b8def", width=2.5)))
if has_labels:
    fig.add_trace(go.Scatter(x=t, y=(df.label == FINE).astype(float), mode="lines", name="label (1 = fine)",
                             line=dict(color="#e6e9f2", width=2, shape="hv", dash="dot")))
if df.moon_alt.notna().any():
    fig.add_trace(go.Scatter(x=t, y=df.moon_alt, mode="lines", name="moon altitude (°)", yaxis="y2",
                             line=dict(color="#f0a530", width=1.5)))
fig.add_hline(y=thr, line_dash="dash", line_color="gray", line_width=1)
fig.update_layout(yaxis=dict(title="P(clear)", range=[-0.05, 1.05]),
                  yaxis2=dict(title="moon altitude (°)", overlaying="y", side="right", showgrid=False),
                  hovermode="x unified")
st.plotly_chart(dark_layout(fig, height=430), width="stretch")
st.markdown("Background colour = condition after smoothing: " +
            " · ".join(f"{v['icon']} {k}" for k, v in CATEGORIES.items()))

# ---------------------------------------------------------------- frame viewer
st.subheader("Frame viewer")
idx = st.select_slider("Time", options=list(range(len(df))), value=len(df) // 2,
                       format_func=lambda i: df.time.iloc[i].strftime("%m-%d %H:%M"))
row = df.iloc[idx]
v1, v2 = st.columns([1, 1.6])
with v1:
    if source == "Test nights":
        tp = thumb_path(row.stem, "test")
        if tp:
            st.image(str(tp), width=300)
        else:
            st.caption("(no thumbnail)")
    elif row.stem in ss.get("batch_imgs", {}):
        st.image(ss["batch_imgs"][row.stem], width=300)
with v2:
    c = CATEGORIES[row.category]
    st.markdown(f"### {c['icon']} {row.category}")
    st.markdown(f"**{row.time:%Y-%m-%d %H:%M}** · frame `{row.stem}`")
    lines = [f"- P(clear), {MODEL_LABELS[model]}: **{row[col]:.2f}** (smoothed {row.p_s:.2f})",
             f"- estimated cloud cover: **{row.cloud_cover:.0%}**",
             f"- sky type: **{cfg['cluster_names'][int(row.sky_type)]}**"]
    if has_labels:
        lines.insert(0, f"- dataset label: **{cfg['classes'][int(row.label)].replace('Data_', '')}**")
    if pd.notna(row.moon_alt):
        lines.append(f"- moon altitude: **{row.moon_alt:.0f}°**")
    st.markdown("\n".join(lines))
    st.caption(c["advice"])

out = df.drop(columns=["p_s", "cover_s"]).assign(sky_type_name=[cfg["cluster_names"][int(s)] for s in df.sky_type])
st.download_button("Download results (CSV)", out.to_csv(index=False), "sky_results.csv", "text/csv")
