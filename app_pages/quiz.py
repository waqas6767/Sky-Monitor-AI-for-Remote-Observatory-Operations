import numpy as np
import streamlit as st

from skyapp.core import CATEGORIES, DEFAULT_THR, load_config, load_test_predictions, sky_verdict, thumb_path

cfg = load_config()
FINE = cfg["fine_index"]
df = load_test_predictions()
df = df[[thumb_path(s) is not None for s in df.stem]].reset_index(drop=True)

st.title("🎯 Sky quiz")
st.caption("Would you open the dome? Judge the sky, then see what the dataset label and the model say.")
if df.empty:
    st.info("No thumbnails available in the assets, so the quiz cannot run.")
    st.stop()

ss = st.session_state
ss.setdefault("q_scores", {"you": 0.0, "model": 0, "n": 0})
ss.setdefault("q_answer", None)
hard = st.toggle("Hard mode: only frames where the models were unsure (0.25 < P(clear) < 0.75)")
pool = df.index[(df.p_ensemble > 0.25) & (df.p_ensemble < 0.75)] if hard else df.index
if "q_idx" not in ss or ss.q_idx not in pool:
    ss.q_idx = int(np.random.choice(pool)); ss.q_answer = None

row = df.loc[ss.q_idx]
left, right = st.columns([1, 1.1], gap="large")
with left:
    st.image(str(thumb_path(row.stem)), width=380)
    st.caption(f"{row.time:%Y-%m-%d %H:%M} local time · moon {'up' if row.moon_alt > 0 else 'down'}")

CHOICES = {"☀️ Clear": 1.0, "⛅ Partly cloudy": 0.5, "☁️ Cloudy": 0.0}
with right:
    if ss.q_answer is None:
        st.markdown("### What do you see?")
        for label, val in CHOICES.items():
            if st.button(label, width="stretch"):
                truth = float(row.label == FINE)
                ss.q_answer = label
                ss.q_scores["you"] += 1 - abs(val - truth)          # partly cloudy = half a point
                ss.q_scores["model"] += int((row.p_ensemble >= 0.5) == bool(truth))
                ss.q_scores["n"] += 1
                st.rerun()
    else:
        truth = cfg["classes"][int(row.label)].replace("Data_", "")
        cat = sky_verdict(row.p_ensemble, row.cloud_cover, row.sky_type, cfg, DEFAULT_THR)
        c = CATEGORIES[cat]
        st.markdown(f"**You said:** {ss.q_answer}")
        st.markdown(f"**Dataset label:** {truth}")
        st.markdown(f"**Model:** {c['icon']} **{cat}** — P(clear) = {row.p_ensemble:.2f}, "
                    f"cloud cover ≈ {row.cloud_cover:.0%}")
        st.markdown(f"**Sky type:** {cfg['cluster_names'][int(row.sky_type)]}")
        if (row.p_ensemble >= 0.5) != (row.label == FINE):
            st.info("The model disagrees with the label here. Look closely: many such frames are partly "
                    "cloudy, or sit inside a time block that was labelled as a whole.")
        if st.button("Next sky ➜", type="primary"):
            ss.q_idx = int(np.random.choice(pool)); ss.q_answer = None
            st.rerun()

    n = ss.q_scores["n"]
    if n:
        st.divider()
        s1, s2 = st.columns(2)
        s1.metric("Your score vs labels", f"{ss.q_scores['you']:.1f} / {n}")
        s2.metric("Model score vs labels", f"{ss.q_scores['model']} / {n}")
        st.caption("Scoring: an exact match with the clear/cloudy label earns 1 point; 'Partly cloudy' earns ½.")
