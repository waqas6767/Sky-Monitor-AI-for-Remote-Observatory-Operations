"""Weather forecast (Open-Meteo) and satellite images (NASA GIBS) for any location.

Both services are free and need no account or API key. Everything here needs internet.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from .sources import http_get

# ----------------------------------------------------------------------------- places
PRESET_PLACES = [
    # name, lat, lon, kind
    ("Higashi-Hiroshima Observatory, Japan", 34.3776, 132.7769, "observatory"),
    ("Bosscha Observatory, Lembang, Indonesia", -6.8249, 107.6156, "observatory"),
    ("National Astronomical Observatory, Mitaka, Japan", 35.6764, 139.5386, "observatory"),
    ("Okayama Astrophysical Observatory, Japan", 34.5765, 133.5944, "observatory"),
    ("Mauna Kea Observatories, Hawaii, USA", 19.8207, -155.4681, "observatory"),
    ("Hiroshima, Japan", 34.3853, 132.4553, "city"),
    ("Tokyo, Japan", 35.6762, 139.6503, "city"),
    ("Bandung, Indonesia", -6.9175, 107.6191, "city"),
    ("Surabaya, Indonesia", -7.2575, 112.7521, "city"),
    ("Jakarta, Indonesia", -6.2088, 106.8456, "city"),
]
USER_PLACES = Path(__file__).resolve().parent.parent / "user_places.json"


def load_user_places() -> list[dict]:
    try:
        return json.loads(USER_PLACES.read_text())
    except Exception:
        return []


def save_user_place(name: str, lat: float, lon: float) -> bool:
    places = [p for p in load_user_places() if p["name"] != name]
    places.append({"name": name, "lat": round(lat, 4), "lon": round(lon, 4)})
    try:
        USER_PLACES.write_text(json.dumps(places, indent=1))
        return True
    except Exception:
        return False


def delete_user_place(name: str) -> None:
    try:
        USER_PLACES.write_text(json.dumps([p for p in load_user_places() if p["name"] != name], indent=1))
    except Exception:
        pass


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode(name: str) -> list[dict]:
    """Search a place name with the Open-Meteo geocoding service."""
    q = urllib.parse.urlencode({"name": name, "count": 8, "language": "en", "format": "json"})
    data = json.loads(http_get(f"https://geocoding-api.open-meteo.com/v1/search?{q}", timeout=20))
    out = []
    for r in data.get("results", []) or []:
        label = ", ".join(x for x in [r.get("name"), r.get("admin1"), r.get("country")] if x)
        out.append({"name": label, "lat": r["latitude"], "lon": r["longitude"]})
    return out


# ----------------------------------------------------------------------------- weather codes
WMO = {
    0: ("☀️", "Clear sky"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
    45: ("🌫️", "Fog"), 48: ("🌫️", "Freezing fog"),
    51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Drizzle"), 55: ("🌧️", "Heavy drizzle"),
    56: ("🌧️", "Freezing drizzle"), 57: ("🌧️", "Freezing drizzle"),
    61: ("🌦️", "Light rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"),
    66: ("🌧️", "Freezing rain"), 67: ("🌧️", "Freezing rain"),
    71: ("🌨️", "Light snow"), 73: ("🌨️", "Snow"), 75: ("❄️", "Heavy snow"), 77: ("🌨️", "Snow grains"),
    80: ("🌦️", "Light showers"), 81: ("🌧️", "Showers"), 82: ("⛈️", "Violent showers"),
    85: ("🌨️", "Snow showers"), 86: ("🌨️", "Heavy snow showers"),
    95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm with hail"), 99: ("⛈️", "Thunderstorm with hail"),
}


def wmo(code) -> tuple[str, str]:
    try:
        return WMO.get(int(code), ("🌡️", f"Code {int(code)}"))
    except (TypeError, ValueError):
        return ("🌡️", "—")


# ----------------------------------------------------------------------------- Open-Meteo
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "cloud_cover_low",
          "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "weather_code"]


def _frame(js: dict) -> pd.DataFrame:
    h = js.get("hourly", {})
    df = pd.DataFrame(h)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(ttl=15 * 60, show_spinner=False)
def forecast(lat: float, lon: float) -> dict:
    """Current conditions, hourly data for the last 92 days and next 16 days, and an 8-day daily outlook.
    Times are local to the location."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "auto", "past_days": 92, "forecast_days": 16,
        "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,weather_code",
        "hourly": ",".join(HOURLY + ["precipitation_probability"]),
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    js = json.loads(http_get(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=30))
    daily = pd.DataFrame(js.get("daily", {}))
    if not daily.empty:
        daily["time"] = pd.to_datetime(daily["time"])
    return {"hourly": _frame(js), "daily": daily, "current": js.get("current", {}),
            "utc_offset_s": int(js.get("utc_offset_seconds", 0)), "timezone": js.get("timezone", "UTC"),
            "elevation": js.get("elevation")}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def archive(lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Hourly weather that actually occurred (ERA5 reanalysis), for dates older than ~3 months."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "start_date": start.isoformat(), "end_date": end.isoformat(), "hourly": ",".join(HOURLY),
    })
    return _frame(json.loads(http_get(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=30)))


def hourly_around(lat: float, lon: float, t_local: dt.datetime, hours: int = 12) -> tuple[pd.DataFrame, str]:
    """Hourly rows from t-hours to t+hours (local time) and the data source used."""
    fc = forecast(lat, lon)
    h = fc["hourly"]
    lo, hi = pd.Timestamp(t_local) - pd.Timedelta(hours=hours), pd.Timestamp(t_local) + pd.Timedelta(hours=hours)
    if not h.empty and h.time.min() <= pd.Timestamp(t_local) <= h.time.max():
        return h[(h.time >= lo) & (h.time <= hi)].reset_index(drop=True), "forecast"
    a = archive(lat, lon, (t_local - dt.timedelta(days=1)).date(), (t_local + dt.timedelta(days=1)).date())
    if a.empty:
        return a, "none"
    return a[(a.time >= lo) & (a.time <= hi)].reset_index(drop=True), "archive"


def row_at(df: pd.DataFrame, t_local: dt.datetime) -> pd.Series | None:
    if df.empty:
        return None
    i = (df.time - pd.Timestamp(t_local)).abs().idxmin()
    return df.loc[i]


# ----------------------------------------------------------------------------- satellite (NASA GIBS)
GIBS_WMS = ["https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
            "https://gibs.earthdata.nasa.gov/wms/epsg4326/nrt/wms.cgi"]
GEO = {
    # satellite: (infrared layer, visible layer)
    "Himawari-9 (Japan)": ("Himawari_AHI_Band13_Clean_Infrared", "Himawari_AHI_Band3_Red_Visible_1km"),
    "GOES-West (USA)": ("GOES-West_ABI_Band13_Clean_Infrared", "GOES-West_ABI_Band2_Red_Visible_1km"),
    "GOES-East (USA)": ("GOES-East_ABI_Band13_Clean_Infrared", "GOES-East_ABI_Band2_Red_Visible_1km"),
}
DAILY_LAYER = ("VIIRS_SNPP_CorrectedReflectance_TrueColor", "VIIRS (Suomi NPP), one daytime pass per day")


def satellite_for(lon: float) -> str | None:
    if lon >= 80 or lon <= -160:
        return "Himawari-9 (Japan)"
    if -160 < lon <= -105:
        return "GOES-West (USA)"
    if -105 < lon <= -20:
        return "GOES-East (USA)"
    return None          # Europe, Africa, West & South Asia: no geostationary imagery in GIBS


def _bbox(lat: float, lon: float, span: float) -> tuple[float, float, float, float]:
    half_lat = span / 2
    half_lon = span / 2 * 1.4
    return (max(-90, lat - half_lat), max(-180, lon - half_lon), min(90, lat + half_lat), min(180, lon + half_lon))


@st.cache_data(ttl=10 * 60, show_spinner=False, max_entries=64)
def _wms(layers: str, time_str: str, bbox: tuple, width: int, height: int) -> bytes:
    q = urllib.parse.urlencode({
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": layers, "STYLES": "",
        "CRS": "EPSG:4326", "BBOX": ",".join(f"{v:.4f}" for v in bbox), "WIDTH": width, "HEIGHT": height,
        "FORMAT": "image/png", "TRANSPARENT": "TRUE", "TIME": time_str,
    })
    err = None
    for base in GIBS_WMS:                      # 'best' collection first, near-real-time collection as backup
        try:
            data = http_get(f"{base}?{q}", timeout=40)
            if data[:4] == b"\x89PNG":
                return data
            err = RuntimeError(data[:200].decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


def _usable(img: Image.Image) -> bool:
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    covered = (a[..., 3] > 0).mean()
    return covered > 0.6 and a[..., :3][a[..., 3] > 0].std() > 3


def _compose(sat_png: bytes, lines_png: bytes | None, lat: float, lon: float, bbox: tuple) -> Image.Image:
    sat = Image.open(io.BytesIO(sat_png)).convert("RGBA")
    base = Image.new("RGBA", sat.size, (8, 12, 24, 255))
    base.alpha_composite(sat)
    if lines_png:
        try:
            base.alpha_composite(Image.open(io.BytesIO(lines_png)).convert("RGBA").resize(sat.size))
        except Exception:
            pass
    w, h = base.size
    x = (lon - bbox[1]) / (bbox[3] - bbox[1]) * w
    y = (bbox[2] - lat) / (bbox[2] - bbox[0]) * h
    d = ImageDraw.Draw(base)
    for r, col in [(13, (255, 255, 255, 255)), (9, (230, 60, 60, 255))]:
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
    d.line([x - 22, y, x - 14, y], fill=(255, 255, 255, 255), width=2)
    d.line([x + 14, y, x + 22, y], fill=(255, 255, 255, 255), width=2)
    return base.convert("RGB")


def satellite_image(lat: float, lon: float, t_utc: dt.datetime, band: str = "infrared",
                    span: float = 12.0, width: int = 900) -> dict:
    """Best available satellite picture near t_utc. Returns {image, layer, time, source, note} or {error}."""
    bbox = _bbox(lat, lon, span)
    height = int(width * (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]))
    sat = satellite_for(lon)
    tried = []
    if sat:
        layer = GEO[sat][0 if band == "infrared" else 1]
        t0 = t_utc.replace(second=0, microsecond=0, minute=t_utc.minute - t_utc.minute % 10)
        # live images appear with a delay, so step back in 10-minute steps; for past times look both ways
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        steps = [-10 * k for k in range(0, 13)] if (now - t0) < dt.timedelta(hours=3) else \
            [0, -10, 10, -20, 20, -30, 30]
        for s in steps:
            t = t0 + dt.timedelta(minutes=s)
            if t > now:
                continue
            ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                png = _wms(layer, ts, bbox, width, height)
                if _usable(Image.open(io.BytesIO(png))):
                    lines = None
                    try:
                        lines = _wms("Coastlines_15m", t.strftime("%Y-%m-%d"), bbox, width, height)
                    except Exception:
                        pass
                    return {"image": _compose(png, lines, lat, lon, bbox), "layer": layer, "time": t,
                            "source": sat, "bbox": bbox,
                            "note": "Infrared: cold, high clouds look bright; works day and night."
                            if band == "infrared" else "Visible light: only useful in daytime."}
                tried.append(ts)
            except Exception as e:      # noqa: BLE001
                tried.append(f"{ts} ({e.__class__.__name__})")
    # fallback: daily true-colour pass (also covers dates before geostationary archives start)
    day = t_utc.date()
    for dd in (0, -1):
        d = (day + dt.timedelta(days=dd)).isoformat()
        try:
            png = _wms(DAILY_LAYER[0], d, bbox, width, height)
            if _usable(Image.open(io.BytesIO(png))):
                lines = None
                try:
                    lines = _wms("Coastlines_15m", d, bbox, width, height)
                except Exception:
                    pass
                why = ("No geostationary weather satellite covers this longitude in NASA's service, so this is"
                       if not sat else "No geostationary image was available for this time, so this is")
                return {"image": _compose(png, lines, lat, lon, bbox), "layer": DAILY_LAYER[0],
                        "time": dt.datetime.fromisoformat(d), "source": DAILY_LAYER[1], "bbox": bbox, "daily": True,
                        "note": f"{why} the daily daytime picture from the polar-orbiting VIIRS satellite."}
        except Exception as e:          # noqa: BLE001
            tried.append(f"{d} ({e.__class__.__name__})")
    return {"error": "No satellite image could be loaded (no internet, or no image for this time).",
            "tried": tried}


def worldview_link(lat: float, lon: float, layer: str, t: dt.datetime, span: float = 12.0) -> str:
    b = _bbox(lat, lon, span)
    tstr = t.strftime("%Y-%m-%d-T%H:%M:%SZ")
    return (f"https://worldview.earthdata.nasa.gov/?v={b[1]:.2f},{b[0]:.2f},{b[3]:.2f},{b[2]:.2f}"
            f"&l={layer},Coastlines_15m&t={tstr}")
