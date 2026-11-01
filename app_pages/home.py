import streamlit as st

from skyapp.core import CATEGORIES, load_config, load_test_predictions

cfg = load_config()
df = load_test_predictions()

st.title("🌌 Sky Monitor AI")
st.markdown(
    f"#### Automatic sky-condition classification for the all-sky camera at {cfg['site']['name']}"
)
st.markdown(
    "A small camera at an astronomical observatory photographs the whole sky every minute. "
    "This app uses convolutional neural networks (CNNs) trained on **8,806 frames from 32 nights** "
    "to decide whether the sky is clear enough to observe — one step towards automating "
    "observatory operations such as opening and closing the dome."
)

acc = ((df.p_ensemble >= 0.5) == (df.label == cfg["fine_index"])).mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test accuracy (unseen nights)", f"{acc:.1%}")
c2.metric("With 15-min smoothing", "87.8%")
c3.metric("Hand-crafted baseline", "77.1%")
c4.metric("Unsupervised cluster purity", "87%")

st.divider()
st.subheader("Explore")
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
cards = [
    ("app_pages/classify.py", "🔭 Classify a sky", "Upload a frame or pick a sample: verdict, confidence, "
     "and a map of which parts of the sky look clear or cloudy."),
    ("app_pages/timeline.py", "🌙 Night timeline & batch", "Classify many frames at once — uploads, a zip, a Google "
     "Drive folder or a local folder — and follow the night minute by minute."),
    ("app_pages/live.py", "📡 Live camera", "Connect to the observatory camera (folder, image URL or video stream) "
     "and classify every new frame automatically, with alerts."),
    ("app_pages/weather.py", "🛰️ Weather & satellite", "Forecast and live satellite picture for any observatory "
     "or city, at any time — and a cross-check with what the camera saw."),
    ("app_pages/quiz.py", "🎯 Sky quiz", "Clear, partly cloudy or cloudy? Test your eye against the model."),
    ("app_pages/clusters.py", "🧭 Explore sky types", "Without any labels, the network sorts the sky into dark clear, "
     "moonlit clear, broken cloud, moonlit overcast — and rain on the dome."),
    ("app_pages/results.py", "📊 Results", "Model comparison, per-night accuracy, the moon effect, training "
     "curves and what limits performance."),
]
for (page, title, text), col in zip(cards, row1 + row2 + row3):
    with col.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)
        st.page_link(page, label="Open", icon=":material/arrow_forward:")

st.divider()
left, right = st.columns([1.2, 1])
with left:
    st.subheader("How it works")
    st.markdown(
        """
1. **Mask** — everything outside the sky circle (date stamp, labels) is blacked out, so the network
   cannot read the date instead of the sky.
2. **Three CNNs** — a small CNN trained from scratch and two ResNet-18 variants (ImageNet-pretrained)
   each estimate *P(clear)*. Their average is the final prediction.
3. **Evidence map** — every model ends in global average pooling + a linear layer, so each sky cell's
   contribution to the decision can be read off exactly. The share of the sky voting *cloudy* gives a
   cloud-cover estimate.
4. **Sky type** — nearest of five clusters discovered *without labels* in CNN feature space.
5. **Verdict** — probability + cloud cover + sky type → one of five conditions:
        """
    )
    st.markdown(" ".join(f"{v['icon']} **{k}**" for k, v in CATEGORIES.items()))
with right:
    st.subheader("Key findings")
    st.markdown(
        """
- An **ensemble** of the three CNNs is the most accurate *and* the most stable from night to night.
- **Moonlight** is the main difficulty; the ensemble loses only 5 points across moon conditions,
  hand-crafted features lose 20.
- Many remaining "errors" are **labelling artefacts**: labels were assigned in 1–2 h blocks, while
  clouds change within minutes.
- Unsupervised clustering found a class missing from the labels: **water droplets on the camera dome**.
        """
    )

st.divider()
st.caption(
    "Images: all-sky camera of the Higashi-Hiroshima Observatory, Hiroshima University (dataset provided "
    "for a course project). The models were trained only on *clear* vs *cloudy*; the intermediate levels "
    "are derived from the evidence map and are indicative only. Not for safety-critical use."
)
