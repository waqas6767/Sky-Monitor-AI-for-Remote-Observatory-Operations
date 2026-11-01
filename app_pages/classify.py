import numpy as np
import plotly.graph_objects as go
import streamlit as st

from skyapp.core import (ASSETS, CATEGORIES, DEFAULT_THR, MODEL_KEYS, MODEL_LABELS, analyse_bytes,
                         is_unusual, load_config, parse_timestamp, moon_altitude, sky_mask, sky_verdict)
from skyapp.ui import dark_layout, evidence_figure, verdict_card

cfg = load_config()
st.title("🔭 Classify a sky")
st.caption("Classify one image: a sample, an upload, a Google Drive / web link, or a camera snapshot. Works best "
           "with frames from the observatory's all-sky camera. For many images use *Night timeline*, for continuous "
           "monitoring use *Live camera*.")

left, right = st.columns([1, 1.35], gap="large")

with left:
    source = st.radio("Image source", ["Sample frames", "Upload", "Link", "Camera snapshot"], horizontal=True,
                      label_visibility="collapsed")
    data, name = None, None
    if source == "Link":
        from skyapp import sources as src
        link = st.text_input("Google Drive file link or direct image URL",
                             placeholder="https://drive.google.com/file/d/…/view   or   https://…/latest.jpg")
        if link:
            if st.session_state.get("cls_link") != link:
                with st.spinner("Downloading…"):
                    try:
                        st.session_state["cls_link_data"] = src.link_bytes(link)
                        st.session_state["cls_link"] = link
                    except Exception as e:
                        st.error(f"Could not get the image: {e}")
            if st.session_state.get("cls_link") == link:
                name, data = st.session_state["cls_link_data"]
    elif source == "Camera snapshot":
        shot = st.camera_input("Take a picture with this device's camera")
        st.caption("Uses the browser camera (webcam or phone). For an observatory camera running continuously, "
                   "use the *Live camera* page.")
        if shot is not None:
            data, name = shot.getvalue(), "camera_snapshot.jpg"
    if source == "Sample frames":
        samples = cfg.get("samples", [])
        if not samples:
            st.info("No sample frames in the assets — upload an image instead.")
        else:
            caps = [s["caption"] for s in samples]
            i = st.selectbox("Sample", range(len(caps)), format_func=lambda j: caps[j])
            path = ASSETS / "samples" / samples[i]["file"]
            data, name = path.read_bytes(), samples[i]["file"]
            st.caption(f"Dataset label: **{samples[i]['label'].replace('Data_', '')}**")
    elif source == "Upload":
        up = st.file_uploader("All-sky image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up is not None:
            data, name = up.getvalue(), up.name

    with st.expander("Verdict thresholds"):
        thr = dict(DEFAULT_THR)
        thr["clear_p"] = st.slider("Clear if P(clear) ≥", 0.5, 0.95, DEFAULT_THR["clear_p"], 0.05)
        thr["clear_cover"] = st.slider("…and cloud cover ≤", 0.05, 0.6, DEFAULT_THR["clear_cover"], 0.05)
        thr["cloudy_p"] = st.slider("Overcast if P(clear) ≤", 0.05, 0.5, DEFAULT_THR["cloudy_p"], 0.05)
        thr["cloudy_cover"] = st.slider("…and cloud cover ≥", 0.4, 0.95, DEFAULT_THR["cloudy_cover"], 0.05)
        st.caption("Everything in between is *mostly clear* or *mostly cloudy*, decided by "
                   "0.5·(1 − P(clear)) + 0.5·cloud cover.")

if data is None:
    right.info("⬅️ Choose a sample or upload an image to start.")
    st.stop()

with st.spinner("Analysing…"):
    r = analyse_bytes(data)

p = float(r["ensemble"]); cover = float(r["cloud_cover"]); stype = int(r["sky_type"])
cat = sky_verdict(p, cover, stype, cfg, thr)
s = cfg["img_size"]
mask = sky_mask(s, s, cfg["mask_radius_frac"])

with right:
    extra = ""
    ts = parse_timestamp(name or "")
    if ts:
        alt = moon_altitude([ts], cfg)
        extra = f"Taken {ts:%Y-%m-%d %H:%M} local time"
        if alt is not None:
            extra += f" · moon altitude {alt[0]:.0f}°" + (" (below horizon)" if alt[0] < 0 else "")
    verdict_card(cat, extra)

    m1, m2, m3 = st.columns(3)
    m1.metric("P(clear), ensemble", f"{p:.0%}")
    m2.metric("Estimated cloud cover", f"{cover:.0%}")
    m3.metric("Sky type", cfg["cluster_names"][stype].split(" (")[0])

    w, h = r["size"]
    if abs(w - h) > 0.05 * max(w, h):
        st.warning("This image is not square like the training frames; the sky mask may cut it badly.")
    if is_unusual(stype, r["sky_type_dist"], cfg):
        st.warning("This image looks unlike anything in the training data, so treat the result with caution.")

    opacity = st.slider("Evidence-map opacity", 0.0, 0.9, 0.5, 0.05,
                        help="Red: this part of the sky looks clear. Blue: it looks cloudy.")
    st.plotly_chart(evidence_figure(r["display"], r["fine_map"], mask, opacity),
                    width="stretch", config={"displayModeBar": False})

st.divider()
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("What each model says")
    keys = MODEL_KEYS + ["ensemble"]
    vals = [float(r[k]) for k in keys]
    fig = go.Figure(go.Bar(
        x=vals, y=[MODEL_LABELS[k] for k in keys], orientation="h",
        marker_color=["#5b8def" if v >= 0.5 else "#8a94ad" for v in vals],
        text=[f"{v:.0%}" for v in vals], textposition="auto"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_xaxes(range=[0, 1], title="P(clear)")
    st.plotly_chart(dark_layout(fig, height=260), width="stretch", config={"displayModeBar": False})
    spread = max(vals[:3]) - min(vals[:3])
    if spread > 0.5:
        st.caption(f"⚠️ The models disagree strongly (spread {spread:.0%}) — a typical sign of an ambiguous, "
                   "partly cloudy or moonlit sky.")
with c2:
    st.subheader("How to read this")
    st.markdown(
        f"""
- **P(clear)** is the average of the three networks' probabilities that the frame is *fine*.
- **Cloud cover** is the share of the sky circle where the evidence map is blue: those regions,
  taken alone, would make the models say *cloudy*.
- **Sky type** is the nearest of five groups the network found *without labels*:
  {", ".join(cfg['cluster_names'])}.
- The models were trained on two classes only (clear / cloudy). The intermediate levels
  {CATEGORIES['Mostly clear']['icon']} and {CATEGORIES['Mostly cloudy']['icon']} combine probability and
  cloud cover, and the {CATEGORIES['Wet dome']['icon']} warning comes from the sky type.
        """
    )
