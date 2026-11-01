import datetime as dt
import hashlib
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skyapp import sources as src
from skyapp.core import (CATEGORIES, DEFAULT_THR, analyse_bytes, is_unusual, load_config, moon_altitude,
                         parse_timestamp, sky_mask, sky_verdict)
from skyapp.ui import dark_layout, evidence_figure, verdict_card

cfg = load_config()
ss = st.session_state
ss.setdefault("live_log", [])
ss.setdefault("live_last_sig", None)

st.title("📡 Live camera")
st.markdown(
    "Connect the app to the observatory camera and classify each new frame automatically. "
    "Choose how the camera delivers its images:"
)
with st.expander("Which option do I need?"):
    st.markdown(
        """
| Your camera setup | Choose |
|---|---|
| The camera software saves a JPG to a folder every minute (most all-sky cameras) | **Watch a folder** |
| The camera has a web page / URL that always shows the latest image | **Latest-image URL** |
| A USB camera, or an IP camera with an RTSP / MJPEG video stream | **Camera or video stream** |

*Folder* and *USB camera* only work when this app runs **on the computer connected to the camera**
(start it with `run_windows.bat`, `run_mac.command` or `run_linux.sh`). A cloud-hosted copy of the app can only reach cameras
with a public URL.
        """
    )

c1, c2 = st.columns([1.3, 1])
with c1:
    mode = st.radio("Camera source", ["Watch a folder", "Latest-image URL", "Camera or video stream"], horizontal=True)
    if mode == "Watch a folder":
        target = st.text_input("Folder where the camera saves its images",
                               placeholder=r"C:\AllSky\images   or   /home/user/allsky/images")
    elif mode == "Latest-image URL":
        target = st.text_input("URL of the latest image", placeholder="http://192.168.1.50/latest.jpg")
    else:
        target = st.text_input("Camera index or stream URL", value="0",
                               help="0 = first USB/built-in camera, 1 = second… or rtsp://user:pass@ip:554/stream")
with c2:
    interval = st.number_input("Check every (seconds)", 5, 3600, 60, 5)
    save_dir = st.text_input("Save grabbed frames to (optional, local folder)", "")
    alerts = st.toggle("Pop-up alert when the condition changes", True)

b1, b2, b3 = st.columns([1, 1, 2])
running = b1.toggle("▶ Monitoring", key="live_running", disabled=not target)
grab_now = b2.button("Grab one frame now", disabled=not target)
if b3.button("Clear history"):
    ss.live_log, ss.live_last_sig = [], None


def grab() -> tuple[bytes, str, dt.datetime] | None:
    """Get the newest frame as (jpeg bytes, name, timestamp); None if nothing new."""
    if mode == "Watch a folder":
        f = src.newest_image(target)
        if f is None:
            raise RuntimeError("No images in that folder (or it does not exist).")
        sig = f"{f}|{f.stat().st_mtime}"
        if sig == ss.live_last_sig:
            return None
        data, name = f.read_bytes(), f.name
        ts = parse_timestamp(f.name) or dt.datetime.fromtimestamp(f.stat().st_mtime)
    else:
        img = src.grab_from_url(target) if mode == "Latest-image URL" else src.grab_from_stream(target)
        buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=92); data = buf.getvalue()
        sig = hashlib.md5(data).hexdigest()
        if sig == ss.live_last_sig:
            return None
        ts = dt.datetime.now(); name = ts.strftime("%y%m%d%H%M%S") + ".jpg"
    ss.live_last_sig = sig
    if save_dir:
        out = Path(save_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    return data, name, ts


def step(show_errors: bool = True) -> None:
    try:
        got = grab()
    except Exception as e:
        if show_errors:
            st.error(f"Could not get a frame: {e}")
        return
    if got is None:
        return
    data, name, ts = got
    r = analyse_bytes(data)
    cat = sky_verdict(float(r["ensemble"]), float(r["cloud_cover"]), int(r["sky_type"]), cfg, DEFAULT_THR)
    prev = ss.live_log[-1]["category"] if ss.live_log else None
    ss.live_log.append({"time": ts, "frame": name, "p_clear": float(r["ensemble"]),
                        "cloud_cover": float(r["cloud_cover"]), "sky_type": cfg["cluster_names"][int(r["sky_type"])],
                        "category": cat, "unusual": is_unusual(int(r["sky_type"]), r["sky_type_dist"], cfg)})
    ss.live_log = ss.live_log[-2880:]                       # keep ~2 days at one frame per minute
    ss.live_current = (data, r, cat, ts)
    if alerts and prev and prev != cat:
        st.toast(f"{CATEGORIES[cat]['icon']} Sky changed: {prev} → {cat}", icon="🔔")


def show() -> None:
    if "live_current" not in ss:
        st.info("No frame yet. Press **Grab one frame now** to test the connection, then switch **Monitoring** on.")
        return
    data, r, cat, ts = ss.live_current
    left, right = st.columns([1.35, 1], gap="large")
    with right:
        alt = moon_altitude([ts], cfg)
        extra = f"Frame time {ts:%Y-%m-%d %H:%M:%S}" + (f" · moon {alt[0]:.0f}°" if alt is not None else "")
        verdict_card(cat, extra)
        m1, m2 = st.columns(2)
        m1.metric("P(clear)", f"{float(r['ensemble']):.0%}")
        m2.metric("Cloud cover", f"{float(r['cloud_cover']):.0%}")
        st.markdown(f"**Sky type:** {cfg['cluster_names'][int(r['sky_type'])]}")
        if is_unusual(int(r["sky_type"]), r["sky_type_dist"], cfg):
            st.warning("This frame looks unlike the training data (different camera or unusual conditions).")
        if running:
            st.caption(f"Monitoring: next check in ≤ {interval} s · {len(ss.live_log)} frames logged")
    with left:
        s = cfg["img_size"]
        st.plotly_chart(evidence_figure(r["display"], r["fine_map"], sky_mask(s, s, cfg["mask_radius_frac"]), 0.45,
                                        height=420), width="stretch", config={"displayModeBar": False})

    if len(ss.live_log) >= 2:
        log = pd.DataFrame(ss.live_log)
        fig = go.Figure()
        runs = (log.category != log.category.shift()).cumsum()
        for _, g in log.groupby(runs):
            fig.add_vrect(x0=g.time.iloc[0], x1=g.time.iloc[-1] + pd.Timedelta(seconds=interval),
                          fillcolor=CATEGORIES[g.category.iloc[0]]["color"], opacity=0.15, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=log.time, y=log.p_clear, mode="lines+markers", name="P(clear)",
                                 line=dict(color="#5b8def")))
        fig.add_trace(go.Scatter(x=log.time, y=log.cloud_cover, mode="lines", name="cloud cover",
                                 line=dict(color="#e6e9f2", dash="dot")))
        fig.update_yaxes(range=[-0.05, 1.05])
        st.plotly_chart(dark_layout(fig, height=300, title="Since monitoring started"), width="stretch")
    if ss.live_log:
        with st.expander(f"Log ({len(ss.live_log)} frames)"):
            log = pd.DataFrame(ss.live_log)
            st.dataframe(log.iloc[::-1], hide_index=True, width="stretch",
                         column_config={"p_clear": st.column_config.ProgressColumn("P(clear)", min_value=0, max_value=1),
                                        "cloud_cover": st.column_config.ProgressColumn("cloud cover", min_value=0, max_value=1)})
            st.download_button("Download log (CSV)", log.to_csv(index=False), "sky_live_log.csv", "text/csv")


if grab_now:
    ss.live_last_sig = None                                  # force a fresh analysis
    step()


@st.fragment(run_every=f"{int(interval)}s" if running else None)
def live_panel():
    if st.session_state.get("live_running"):
        step(show_errors=True)
    show()


live_panel()
