import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skyapp import weather as wx
from skyapp.core import CATEGORIES, analyse_bytes, load_config, parse_timestamp, sky_verdict
from skyapp.ui import dark_layout

cfg = load_config()
ss = st.session_state
ss.setdefault("wx_place", None)

st.title("🛰️ Weather & satellite")
st.markdown(
    "An independent check on the sky: the **weather forecast** for a place and time (left) and the "
    "**satellite picture** of the same area (right). Optionally compare both with what the camera saw. "
    "Data: [Open-Meteo](https://open-meteo.com) forecasts and NASA [GIBS](https://www.earthdata.nasa.gov/gibs) "
    "satellite imagery — free, no account needed, internet required."
)

# ----------------------------------------------------------------------------- 1. location
st.subheader("1 · Location")
how = st.radio("Choose the location", ["From the list", "Search by name", "Enter coordinates"],
               horizontal=True, label_visibility="collapsed")

if how == "From the list":
    user = wx.load_user_places()
    options = ([f"🔭 {n}" if k == "observatory" else f"🏙️ {n}" for n, _, _, k in wx.PRESET_PLACES]
               + [f"📌 {p['name']}" for p in user])
    coords = ([(n, la, lo) for n, la, lo, _ in wx.PRESET_PLACES] + [(p["name"], p["lat"], p["lon"]) for p in user])
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Observatory or city", options, index=None,
                        placeholder="Choose an observatory or city…")
    if pick is not None:
        n, la, lo = coords[options.index(pick)]
        ss.wx_place = {"name": n, "lat": la, "lon": lo}
        if pick.startswith("📌") and c2.button("Remove from my list", width="stretch"):
            wx.delete_user_place(n)
            ss.wx_place = None
            st.rerun()
    st.caption("🔭 observatory · 🏙️ city · 📌 places you added yourself (add them with *Search by name* "
               "or *Enter coordinates*).")

elif how == "Search by name":
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Place name", placeholder="e.g. Bosscha Observatory, Surabaya, Kyoto…")
    if q.strip():
        try:
            hits = wx.geocode(q.strip())
        except Exception as e:  # noqa: BLE001
            hits = []
            st.error(f"Place search failed — check the internet connection ({e.__class__.__name__}).")
        if hits:
            labels = [f"{h['name']}  ({h['lat']:.3f}, {h['lon']:.3f})" for h in hits]
            sel = st.selectbox("Matching places", labels, index=None, placeholder="Choose the right one…")
            if sel is not None:
                h = hits[labels.index(sel)]
                ss.wx_place = dict(h)
                if c2.button("Add to my list", width="stretch"):
                    st.toast("Saved to your list" if wx.save_user_place(h["name"], h["lat"], h["lon"])
                             else "Could not save the list", icon="📌")
        elif q.strip():
            st.info("No place found. Observatories are often not in the place database — search the nearest "
                    "town, or enter the coordinates.")

else:
    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.text_input("Name", placeholder="e.g. My rooftop telescope")
    lat = c2.number_input("Latitude (°, north +)", -90.0, 90.0, value=None, format="%.4f",
                          placeholder="e.g. 34.3776")
    lon = c3.number_input("Longitude (°, east +)", -180.0, 180.0, value=None, format="%.4f",
                          placeholder="e.g. 132.7769")
    b1, b2, _ = st.columns([1, 1, 2])
    if lat is not None and lon is not None:
        label = name.strip() or f"{lat:.4f}, {lon:.4f}"
        if b1.button("Use this location", type="primary", width="stretch"):
            ss.wx_place = {"name": label, "lat": lat, "lon": lon}
        if b2.button("Save to my list", width="stretch"):
            ok = wx.save_user_place(label, lat, lon)
            ss.wx_place = {"name": label, "lat": lat, "lon": lon}
            st.toast("Saved — it now appears under *From the list*" if ok else "Could not save the list", icon="📌")
    st.caption("Tip: right-click a spot in Google Maps to copy its latitude and longitude.")

place = ss.wx_place
if not place:
    st.info("Choose a location to see its forecast and satellite picture.")
    st.stop()

# ----------------------------------------------------------------------------- 2. time
st.subheader("2 · Time")
try:
    with st.spinner("Loading the forecast…"):
        fc = wx.forecast(place["lat"], place["lon"])
except Exception as e:  # noqa: BLE001
    st.error(f"Could not reach the weather service ({e.__class__.__name__}). This page needs internet.")
    st.stop()
offset = dt.timedelta(seconds=fc["utc_offset_s"])
now_local = (dt.datetime.now(dt.timezone.utc) + offset).replace(tzinfo=None)

opts = ["Now", "Choose date & time", "Time of a camera frame"]
if ss.get("live_log"):
    opts.append("Latest live-camera frame")
when = st.radio("Time", opts, horizontal=True, label_visibility="collapsed")
frame_result = None

if when == "Now":
    t_local = now_local
elif when == "Choose date & time":
    c1, c2 = st.columns(2)
    d = c1.date_input("Date", value=now_local.date(), min_value=dt.date(1950, 1, 1),
                      max_value=(now_local + dt.timedelta(days=15)).date())
    t = c2.time_input("Local time at the location", value=now_local.time().replace(second=0, microsecond=0),
                      step=dt.timedelta(minutes=10))
    t_local = dt.datetime.combine(d, t)
elif when == "Time of a camera frame":
    up = st.file_uploader("Camera frame (the time is read from a file name like 211124194628.jpg)",
                          type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload a frame, or choose another time option.")
        st.stop()
    data = up.getvalue()
    ts = parse_timestamp(up.name)
    if ts is None:
        st.warning("No timestamp in the file name — enter the time the frame was taken.")
        c1, c2 = st.columns(2)
        d = c1.date_input("Date", value=now_local.date())
        t = c2.time_input("Local time", value=dt.time(21, 0))
        ts = dt.datetime.combine(d, t)
    t_local = ts
    r = analyse_bytes(data)
    frame_result = {"category": sky_verdict(float(r["ensemble"]), float(r["cloud_cover"]),
                                            int(r["sky_type"]), cfg),
                    "p_clear": float(r["ensemble"]), "cloud_cover": float(r["cloud_cover"]),
                    "display": r["display"], "name": up.name}
else:
    last = ss.live_log[-1]
    t_local = pd.Timestamp(last["time"]).to_pydatetime()
    frame_result = {"category": last["category"], "p_clear": last["p_clear"],
                    "cloud_cover": last["cloud_cover"], "name": last["frame"]}

t_utc = t_local - offset
st.caption(f"📍 **{place['name']}** ({place['lat']:.4f}, {place['lon']:.4f}) · "
           f"🕒 **{t_local:%Y-%m-%d %H:%M}** local time ({fc['timezone']}, UTC{fc['utc_offset_s'] / 3600:+g})")

# ----------------------------------------------------------------------------- 3. forecast | satellite
left, right = st.columns(2, gap="large")

with left:
    st.subheader("🌦️ Forecast")
    try:
        hours, source = wx.hourly_around(place["lat"], place["lon"], t_local, hours=12)
    except Exception as e:  # noqa: BLE001
        hours, source = pd.DataFrame(), "none"
        st.error(f"Could not load weather data ({e.__class__.__name__}).")
    row = wx.row_at(hours, t_local)
    if row is None:
        st.warning("No weather data for this time.")
    else:
        use_current = when == "Now" and fc["current"]
        cur = fc["current"] if use_current else row
        icon, text = wx.wmo(cur["weather_code"])
        pp = row.get("precipitation_probability") if "precipitation_probability" in row else None
        rain_txt = (f"🌧️ Chance of rain: <b>{pp:.0f}%</b><br>" if pp is not None and pd.notna(pp) else "")
        card = (
            '<div class="card" style="display:flex;gap:1.2rem;align-items:center">'
            f'<div style="font-size:3.4rem;line-height:1">{icon}</div>'
            f'<div><div style="font-size:2.6rem;font-weight:700;line-height:1">{cur["temperature_2m"]:.0f}°C</div>'
            f'<div class="small">{text}</div></div>'
            '<div class="small" style="margin-left:auto;line-height:1.7">'
            f'☁️ Cloud cover: <b>{cur["cloud_cover"]:.0f}%</b><br>{rain_txt}'
            f'💧 Humidity: <b>{cur["relative_humidity_2m"]:.0f}%</b><br>'
            f'💨 Wind: <b>{cur["wind_speed_10m"]:.0f} km/h</b></div></div>'
        )
        st.markdown(card, unsafe_allow_html=True)
        st.caption({"forecast": "Source: Open-Meteo forecast models (≈ 10 km grid).",
                    "archive": "Source: ERA5 reanalysis — the weather that actually occurred (≈ 30 km grid); "
                               "forecasts are only kept for the last 3 months."}.get(source, ""))
        st.markdown(f"**Cloud layers at {pd.Timestamp(row['time']):%H:%M}:** low {row['cloud_cover_low']:.0f}% · "
                    f"mid {row['cloud_cover_mid']:.0f}% · high {row['cloud_cover_high']:.0f}%")

        tab_c, tab_t, tab_p, tab_w = st.tabs(["Cloud cover", "Temperature", "Precipitation", "Wind"])
        mark = dict(x=pd.Timestamp(t_local), line_dash="dot", line_color="#ffffff")

        def chart(fig, title):
            fig.add_vline(**mark)
            return dark_layout(fig, height=280, yaxis_title=title)

        with tab_c:
            fig = go.Figure()
            for col, name, colour in [("cloud_cover", "total", "#e8edf5"), ("cloud_cover_low", "low", "#7aa7ff"),
                                      ("cloud_cover_mid", "mid", "#9ccc3c"), ("cloud_cover_high", "high", "#f0a530")]:
                fig.add_trace(go.Scatter(x=hours.time, y=hours[col], name=name, mode="lines",
                                         line=dict(color=colour, width=3 if col == "cloud_cover" else 1.5)))
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(chart(fig, "cloud cover (%)"), width="stretch", config={"displayModeBar": False})
        with tab_t:
            fig = go.Figure(go.Scatter(x=hours.time, y=hours.temperature_2m, mode="lines", fill="tozeroy",
                                       line=dict(color="#f5c542")))
            st.plotly_chart(chart(fig, "°C"), width="stretch", config={"displayModeBar": False})
        with tab_p:
            fig = go.Figure(go.Bar(x=hours.time, y=hours.precipitation, marker_color="#5aa0ff", name="mm"))
            if "precipitation_probability" in hours:
                fig.add_trace(go.Scatter(x=hours.time, y=hours.precipitation_probability, yaxis="y2",
                                         mode="lines", line=dict(color="#e8edf5"), name="chance %"))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False))
            st.plotly_chart(chart(fig, "mm per hour"), width="stretch", config={"displayModeBar": False})
        with tab_w:
            fig = go.Figure(go.Scatter(x=hours.time, y=hours.wind_speed_10m, mode="lines",
                                       line=dict(color="#9ad0c2")))
            st.plotly_chart(chart(fig, "km/h"), width="stretch", config={"displayModeBar": False})

        daily = fc["daily"]
        if source == "forecast" and not daily.empty:
            daily = daily[daily.time.dt.date >= now_local.date()].head(8)
            cols = st.columns(len(daily))
            for c, (_, dd) in zip(cols, daily.iterrows()):
                ic, tx = wx.wmo(dd.weather_code)
                c.markdown(f"<div style='text-align:center' title='{tx}'><b>{dd.time:%a}</b><br>"
                           f"<span style='font-size:1.6rem'>{ic}</span><br>"
                           f"<span class='small'>{dd.temperature_2m_max:.0f}° / {dd.temperature_2m_min:.0f}°</span>"
                           f"</div>", unsafe_allow_html=True)

with right:
    st.subheader("🛰️ Satellite")
    band = st.radio("Picture", ["infrared", "visible"], horizontal=True,
                    format_func=lambda b: {"infrared": "Infrared (day & night)", "visible": "Visible (daytime)"}[b])
    span = st.select_slider("Area shown", [4, 8, 12, 20, 30], value=12, format_func=lambda s: f"{s}° wide")
    with st.spinner("Loading the satellite picture…"):
        sat = wx.satellite_image(place["lat"], place["lon"], t_utc, band=band, span=span)
    if "error" in sat:
        st.error(sat["error"])
    else:
        st.image(sat["image"], width="stretch")
        local_txt = "" if sat.get("daily") else f" ({sat['time'] + offset:%H:%M} local)"
        when_txt = f"{sat['time']:%Y-%m-%d}" if sat.get("daily") else f"{sat['time']:%Y-%m-%d %H:%M} UTC{local_txt}"
        st.caption(f"{sat['source']} · {when_txt} · red circle = {place['name']}. {sat['note']}")
        st.link_button("Open the interactive map (NASA Worldview)",
                       wx.worldview_link(place["lat"], place["lon"], sat["layer"], sat["time"], span),
                       icon=":material/open_in_new:")

# ----------------------------------------------------------------------------- 4. comparison
if frame_result is not None and row is not None:
    st.divider()
    st.subheader("3 · Camera vs forecast")
    cat = frame_result["category"]
    c0, c1, c2, c3 = st.columns([1.1, 1, 1, 1.4])
    if "display" in frame_result:
        c0.image(frame_result["display"], caption=frame_result["name"], width="stretch")
    c1.markdown(f"<div class='small'>Camera says</div><div style='font-size:1.5rem;font-weight:700'>"
                f"{CATEGORIES[cat]['icon']} {cat}</div>", unsafe_allow_html=True)
    c1.caption(f"P(clear) {frame_result['p_clear']:.2f}")
    c2.metric("Camera cloud cover", f"{frame_result['cloud_cover']:.0%}")
    c2.metric("Forecast cloud cover", f"{row['cloud_cover']:.0f}%")
    cam, fct = frame_result["cloud_cover"] * 100, float(row["cloud_cover"])
    if abs(cam - fct) <= 30 or (cam < 30 and fct < 40) or (cam > 70 and fct > 60):
        c3.success("✅ The camera and the forecast **agree**.")
    else:
        c3.warning("⚠️ The camera and the forecast **disagree**.")
    c3.caption("They need not match exactly: the forecast describes a grid square of 10–30 km and all cloud "
               "layers, while the camera sees the sky above one spot. Check the satellite picture to decide. "
               "Make sure the location above is where the camera stands.")
