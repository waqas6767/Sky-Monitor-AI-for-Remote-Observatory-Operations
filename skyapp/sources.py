"""Image sources: Google Drive links, image URLs, local folders and live cameras."""
from __future__ import annotations

import html
import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
UA = {"User-Agent": "Mozilla/5.0 (SkyMonitorAI)"}


# ----------------------------------------------------------------------------- generic HTTP
def http_get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def open_image_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


# ----------------------------------------------------------------------------- Google Drive
def drive_id(link: str) -> tuple[str | None, str | None]:
    """Return (kind, id) for a Drive link: kind is 'folder' or 'file'."""
    link = link.strip()
    m = re.search(r"/folders/([\w-]{10,})", link)
    if m:
        return "folder", m.group(1)
    m = re.search(r"/file/d/([\w-]{10,})", link) or re.search(r"[?&]id=([\w-]{10,})", link)
    if m:
        return "file", m.group(1)
    if re.fullmatch(r"[\w-]{20,}", link):
        return "file", link
    return None, None


_ENTRY = re.compile(r'<div class="flip-entry" id="entry-([\w-]+)".*?<a href="([^"]+)".*?'
                    r'<div class="flip-entry-title">(.*?)</div>', re.S)


def parse_drive_listing(page: str) -> list[dict]:
    """Parse Drive's embedded folder view (lists all entries of a public folder on one page)."""
    out = []
    for fid, href, title in _ENTRY.findall(page):
        out.append({"id": fid, "name": html.unescape(title).strip(),
                    "is_folder": "/folders/" in href})
    return out


def list_drive_folder(folder_id: str, recursive: bool = True, max_depth: int = 2, _depth: int = 0,
                      _prefix: str = "") -> list[dict]:
    """List image files in a shared ('anyone with the link') Drive folder, optionally with subfolders."""
    page = http_get(f"https://drive.google.com/embeddedfolderview?id={folder_id}").decode("utf-8", "replace")
    files = []
    for e in parse_drive_listing(page):
        if e["is_folder"]:
            if recursive and _depth < max_depth:
                files += list_drive_folder(e["id"], recursive, max_depth, _depth + 1, f"{_prefix}{e['name']}/")
        elif e["name"].lower().endswith(IMG_EXT):
            files.append({"id": e["id"], "name": e["name"], "folder": _prefix.rstrip("/")})
    return files


def download_drive_file(file_id: str) -> bytes:
    """Download one public Drive file (uses gdown when available: it handles Google's confirm pages)."""
    try:
        import gdown
    except ImportError:
        return http_get(f"https://drive.google.com/uc?export=download&id={file_id}")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = gdown.download(id=file_id, output=str(Path(d) / "file"), quiet=True)
        if not out or not Path(out).exists():
            raise RuntimeError("Download failed: is the file shared as 'Anyone with the link'?")
        return Path(out).read_bytes()


def link_bytes(link: str) -> tuple[str, bytes]:
    """Fetch a single file given a Drive file link or any direct URL. Returns (name, bytes)."""
    kind, fid = drive_id(link)
    if kind == "file":
        return f"{fid}", download_drive_file(fid)
    if kind == "folder":
        raise ValueError("This is a folder link: use the batch / timeline page for folders.")
    name = Path(link.split("?")[0]).name or "image"
    return name, http_get(link)


def images_from_zip(data: bytes, limit: int = 2000) -> list[tuple[str, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in sorted(z.namelist()):
            if n.lower().endswith(IMG_EXT) and not Path(n).name.startswith("."):
                out.append((Path(n).name, z.read(n)))
                if len(out) >= limit:
                    break
    return out


# ----------------------------------------------------------------------------- local folders
def list_local_images(folder: str) -> list[Path]:
    p = Path(folder).expanduser()
    if not p.is_dir():
        return []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT and not f.name.startswith("."))


def newest_image(folder: str) -> Path | None:
    p = Path(folder).expanduser()
    if not p.is_dir():
        return None
    files = [f for f in p.iterdir() if f.suffix.lower() in IMG_EXT and not f.name.startswith(".")]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


# ----------------------------------------------------------------------------- live cameras
def grab_from_url(url: str) -> Image.Image:
    """Snapshot from a URL that serves the camera's latest image (JPEG/PNG)."""
    sep = "&" if "?" in url else "?"
    return open_image_bytes(http_get(f"{url}{sep}_t={int(time.time())}", timeout=20))   # defeat caching


def grab_from_stream(source: str, warmup: int = 3) -> Image.Image:
    """One frame from a USB camera (index '0', '1', …, or a device path such as
    /dev/v4l/by-id/…) or a video stream (rtsp://, http://…mjpg)."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("OpenCV is not installed: pip install opencv-python-headless") from e
    s = source.strip()
    src = int(s) if s.isdigit() else s
    local = isinstance(src, int) or s.startswith("/dev/")
    if local and sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    if local:
        # Ask USB cameras for MJPEG: many cheap webcams (e.g. Sonix) send raw YUYV with
        # swapped colour order, which shows up as a purple/green picture.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        warmup = max(warmup, 10)                 # let auto-exposure / white balance settle
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera/stream: {source}")
        frame = None
        for _ in range(max(1, warmup)):          # first frames are often dark or stale
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("The camera opened but returned no frame.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
