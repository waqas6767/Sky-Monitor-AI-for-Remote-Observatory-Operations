"""Sky Monitor AI — entry point. Run locally with:  streamlit run streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Sky Monitor AI", page_icon="🌌", layout="wide")

from skyapp.core import ensure_assets  # noqa: E402
from skyapp.ui import inject_css       # noqa: E402

inject_css()
ensure_assets()

pages = [
    st.Page("app_pages/home.py",     title="Home",              icon=":material/home:", default=True),
    st.Page("app_pages/classify.py", title="Classify a sky",    icon=":material/photo_camera:"),
    st.Page("app_pages/timeline.py", title="Night timeline & batch", icon=":material/timeline:"),
    st.Page("app_pages/live.py",     title="Live camera",       icon=":material/videocam:"),
    st.Page("app_pages/weather.py",  title="Weather & satellite", icon=":material/satellite_alt:"),
    st.Page("app_pages/quiz.py",     title="Sky quiz",          icon=":material/quiz:"),
    st.Page("app_pages/clusters.py", title="Explore sky types", icon=":material/bubble_chart:"),
    st.Page("app_pages/results.py",  title="Results",           icon=":material/analytics:"),
]
st.navigation(pages).run()
