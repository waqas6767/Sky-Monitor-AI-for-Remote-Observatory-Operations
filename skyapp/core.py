"""Core logic: asset loading, preprocessing, ONNX inference, evidence maps and sky verdicts."""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
STATIC = ROOT / "static"

MODEL_KEYS = ["smallcnn", "resnet_v2", "resnet_v3"]
MODEL_LABELS = {
    "ensemble": "Ensemble (3 models)",
    "smallcnn": "Small CNN",
    "resnet_v2": "ResNet-18 v2",
    "resnet_v3": "ResNet-18 v3",
    "handcrafted": "Hand-crafted features",
}

# Sky categories shown to the user (the models were trained on 2 classes; the 3 middle levels
# combine the ensemble probability with the estimated share of the sky voting "cloudy").
CATEGORIES = {
    "Clear":         dict(color="#2fb37a", icon="☀️", advice="Open the dome — good observing conditions."),
    "Mostly clear":  dict(color="#9ccc3c", icon="🌤️", advice="Observe, but monitor passing clouds."),
    "Mostly cloudy": dict(color="#f0a530", icon="⛅", advice="Stand by — wait for a clear gap."),
    "Overcast":      dict(color="#7d8aa5", icon="☁️", advice="Keep the dome closed."),
    "Wet dome":      dict(color="#e0524f", icon="🌧️", advice="Close the dome — water droplets on the camera dome (rain or dew likely)."),
}
DEFAULT_THR = dict(clear_p=0.70, clear_cover=0.30, cloudy_p=0.30, cloudy_cover=0.70)


# ----------------------------------------------------------------------------- assets
def _assets_url() -> str:
    url = os.environ.get("SKYAPP_ASSETS_URL", "")
    if not url:
        try:
            url = st.secrets.get("assets_url", "")
        except Exception:
            url = ""
    if not url and (ROOT / "assets_url.txt").exists():
        url = (ROOT / "assets_url.txt").read_text().strip()
    return url


@st.cache_resource(show_spinner=False)
def _download_assets(url: str) -> bool:
    ASSETS.mkdir(exist_ok=True)
    tmp = ASSETS / "_download.zip"
    urllib.request.urlretrieve(url, tmp)
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():                       # refuse paths that escape the folder
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in assets zip: {name}")
        z.extractall(ASSETS)
    tmp.unlink()
    if not (ASSETS / "config.json").exists():               # e.g. the standalone zip: SkyMonitorAI/assets/...
        hits = sorted(ASSETS.rglob("config.json"), key=lambda p: len(p.parts))
        if hits:
            import shutil
            for item in hits[0].parent.iterdir():
                shutil.move(str(item), str(ASSETS / item.name))
    return True


def ensure_assets() -> None:
    """Make sure models and data are present; download them from the release URL on first start."""
    if (ASSETS / "config.json").exists():
        return
    url = _assets_url()
    if not url or "YOUR-" in url:
        st.error("Model files not found. Put the contents of `skyapp_assets.zip` into an `assets/` "
                 "folder, or set its download URL in `assets_url.txt` (see README).")
        st.stop()
    with st.spinner("Downloading model files (first start only, about a minute)…"):
        _download_assets(url)
    if not (ASSETS / "config.json").exists():
        st.error("The downloaded zip does not contain config.json at its top level.")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_config() -> dict:
    return json.loads((ASSETS / "config.json").read_text())


@st.cache_resource(show_spinner=False)
def load_static() -> dict:
    return json.loads((STATIC / "static_results.json").read_text())


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sessions = {k: ort.InferenceSession(str(ASSETS / "models" / f"{k}.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
                for k in MODEL_KEYS + ["embedder"]}
    params = {k: v for k, v in np.load(ASSETS / "models" / "params.npz").items()}
    return sessions, params


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    df = pd.read_csv(ASSETS / "data" / "test_predictions.csv", dtype={"stem": str, "night": str})
    df["time"] = pd.to_datetime(df["time"])
    df["p_ensemble"] = df[[f"p_{k}" for k in MODEL_KEYS]].mean(axis=1)
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_tsne() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "data" / "tsne.csv", dtype={"stem": str, "night": str})


def thumb_path(stem: str, kind: str = "test") -> Path | None:
    p = ASSETS / "thumbs" / kind / f"{stem}.jpg"
    return p if p.exists() else None


# ----------------------------------------------------------------------------- preprocessing
def sky_mask(w: int, h: int, frac: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= (frac * min(w, h)) ** 2


def preprocess(img: Image.Image, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mask everything outside the sky circle, resize, normalise. Returns (CHW tensor, HWC display image)."""
    a = np.asarray(img.convert("RGB"))
    a = (a * sky_mask(a.shape[1], a.shape[0], cfg["mask_radius_frac"])[..., None]).astype(np.uint8)
    s = cfg["img_size"]
    disp = np.asarray(Image.fromarray(a).resize((s, s), Image.BILINEAR))
    x = (disp.astype(np.float32) / 255.0 - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return x.transpose(2, 0, 1), disp


def parse_timestamp(name: str) -> dt.datetime | None:
    m = re.search(r"(\d{12})", name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%y%m%d%H%M%S")
    except ValueError:
        return None


def moon_altitude(times_local: list[dt.datetime], cfg: dict) -> np.ndarray | None:
    """Moon altitude (deg) at the observatory for local timestamps; None if astropy is unavailable."""
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, get_body
        from astropy.time import Time
        from astropy.utils import iers
        iers.conf.auto_download = False
    except ImportError:
        return None
    site = cfg["site"]
    loc = EarthLocation(lat=site["lat"] * u.deg, lon=site["lon"] * u.deg, height=site["height"] * u.m)
    t = Time([tl - dt.timedelta(hours=site["utc_offset_hours"]) for tl in times_local], scale="utc")
    return get_body("moon", t, loc).transform_to(AltAz(obstime=t, location=loc)).alt.deg


# ----------------------------------------------------------------------------- inference
def _run(sess, x: np.ndarray) -> list[np.ndarray]:
    try:
        return sess.run(None, {"input": x})
    except Exception:                                  # model exported without a dynamic batch axis
        outs = [sess.run(None, {"input": x[i:i + 1]}) for i in range(len(x))]
        return [np.concatenate([o[j] for o in outs]) for j in range(len(outs[0]))]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _upsample(m: np.ndarray, s: int) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.float32)).resize((s, s), Image.BILINEAR))


def analyse(X: np.ndarray, keep_maps: bool = True) -> dict:
    """Run all models on a batch (N,3,S,S).

    For every model the classifier is global-average-pool + linear layer, so the logit difference
    z_fine - z_cloudy is exactly the mean over the final feature map of a *local logit*
    dw·A(x, y) + db. sigmoid(local logit) is therefore the verdict each sky cell would give on its own;
    the ensemble map is the mean of the three models' maps, and the cloud cover is the share of sky
    pixels whose map value is below 0.5.
    """
    cfg = load_config()
    sessions, P = load_models()
    fine, cloudy = cfg["fine_index"], 1 - cfg["fine_index"]
    s = cfg["img_size"]
    mask = sky_mask(s, s, cfg["mask_radius_frac"])
    out = {k: [] for k in MODEL_KEYS}
    maps_all, cover = [], []
    for b in range(0, len(X), 16):
        xb = np.ascontiguousarray(X[b:b + 16], dtype=np.float32)
        maps = []
        for k in MODEL_KEYS:
            logits, feat = _run(sessions[k], xb)
            out[k].append(_softmax(logits)[:, fine])
            dw = P[f"{k}_w"][fine] - P[f"{k}_w"][cloudy]
            db = float(P[f"{k}_b"][fine] - P[f"{k}_b"][cloudy])
            loc = np.einsum("c,nchw->nhw", dw, feat) + db
            maps.append(np.stack([_upsample(1 / (1 + np.exp(-m)), s) for m in loc]))
        fm = np.mean(maps, axis=0)
        cover.append((((fm < 0.5) & mask).sum((1, 2)) / mask.sum()))
        if keep_maps:
            maps_all.append(fm)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["ensemble"] = np.mean([res[k] for k in MODEL_KEYS], axis=0)
    res["cloud_cover"] = np.concatenate(cover)
    res["sky_type"], res["sky_type_dist"] = sky_type(X)
    if keep_maps:
        res["fine_map"] = np.concatenate(maps_all)
    return res


def sky_type(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest unsupervised cluster (K-means on ImageNet ResNet-18 embeddings)."""
    sessions, P = load_models()
    e = np.concatenate([_run(sessions["embedder"], np.ascontiguousarray(X[b:b + 16], dtype=np.float32))[0]
                        for b in range(0, len(X), 16)])
    z = ((e - P["scaler_mean"]) / P["scaler_scale"] - P["pca_mean"]) @ P["pca_components"].T
    d = np.sqrt(((z[:, None, :] - P["km_centers"][None]) ** 2).sum(-1))
    return d.argmin(1), d.min(1)


@st.cache_data(show_spinner=False, max_entries=64)
def analyse_bytes(data: bytes) -> dict:
    """Analyse one image given as file bytes (cached, so re-runs of the page are instant)."""
    cfg = load_config()
    img = Image.open(io.BytesIO(data))
    x, disp = preprocess(img, cfg)
    res = analyse(x[None], keep_maps=True)
    out = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in res.items()}
    out["display"] = disp
    out["size"] = img.size
    return out


def analyse_items(items: list[tuple[str, bytes]], progress=None) -> pd.DataFrame:
    """Analyse many images given as (name, bytes). Returns one row per readable image, sorted by time."""
    cfg = load_config()
    rows, X = [], []
    for i, (name, data) in enumerate(items):
        try:
            x, _ = preprocess(Image.open(io.BytesIO(data)), cfg)
        except Exception:
            continue                                           # skip unreadable files
        X.append(x)
        rows.append({"stem": Path(name).stem, "time": parse_timestamp(name)})
        if progress:
            progress((i + 1) / len(items) * 0.3, "Reading images…")
    if not X:
        return pd.DataFrame()
    X = np.stack(X)
    res: dict[str, list] = {}
    for b in range(0, len(X), 32):
        r = analyse(X[b:b + 32], keep_maps=False)
        for k, v in r.items():
            res.setdefault(k, []).append(v)
        if progress:
            progress(0.3 + 0.7 * min(1.0, (b + 32) / len(X)), "Running the models…")
    df = pd.DataFrame(rows)
    for k in MODEL_KEYS + ["ensemble"]:
        df[f"p_{k}"] = np.concatenate(res[k])
    for k in ["cloud_cover", "sky_type", "sky_type_dist"]:
        df[k] = np.concatenate(res[k])
    synthetic = bool(df.time.isna().any())
    if synthetic:
        df["time"] = pd.date_range("2000-01-01 20:00", periods=len(df), freq="min")
    df["time"] = pd.to_datetime(df.time)
    df = df.sort_values("time").reset_index(drop=True)
    alt = None if synthetic else moon_altitude(list(df.time.dt.to_pydatetime()), cfg)
    df["moon_alt"] = alt if alt is not None else np.nan
    df["night"] = "input"
    df["category"] = [sky_verdict(p, c, s, cfg) for p, c, s in zip(df.p_ensemble, df.cloud_cover, df.sky_type)]
    df.attrs["synthetic_time"] = synthetic
    return df


# ----------------------------------------------------------------------------- verdicts
def sky_verdict(p_fine: float, cover: float, stype: int | None, cfg: dict, thr: dict = DEFAULT_THR) -> str:
    if stype is not None and int(stype) == int(cfg["wet_cluster"]):
        return "Wet dome"
    if p_fine >= thr["clear_p"] and cover <= thr["clear_cover"]:
        return "Clear"
    if p_fine <= thr["cloudy_p"] and cover >= thr["cloudy_cover"]:
        return "Overcast"
    cloudiness = 0.5 * (1 - p_fine) + 0.5 * cover
    return "Mostly clear" if cloudiness < 0.5 else "Mostly cloudy"


def is_unusual(stype: int, dist: float, cfg: dict) -> bool:
    """True if the image is farther from its cluster centre than 99% of the clustering images (x1.2)."""
    return float(dist) > 1.2 * float(cfg["cluster_p99"][int(stype)])


# ----------------------------------------------------------------------------- metrics
def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def smooth(df: pd.DataFrame, col: str, window: int, centred: bool = True) -> pd.Series:
    """Rolling median within each night (frames are ~1 minute apart)."""
    if window <= 1:
        return df[col]
    return df.groupby("night")[col].transform(
        lambda s: s.rolling(window, center=centred, min_periods=1).median())
