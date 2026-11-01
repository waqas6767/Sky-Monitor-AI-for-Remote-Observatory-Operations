#!/usr/bin/env bash
# Sky Monitor AI launcher for macOS and Linux (double-click on macOS, or: bash run_mac.command)
cd "$(dirname "$0")" || exit 1
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close…"; exit 1
fi
mkdir -p "$HOME/.streamlit"                                   # skip Streamlit's one-time e-mail question
[ -f "$HOME/.streamlit/credentials.toml" ] || printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
if [ ! -x .venv/bin/python ]; then
  echo "First start: installing Sky Monitor AI. This takes a few minutes and needs internet once."
  "$PY" -m venv .venv || { echo "Could not create a virtual environment (Linux: sudo apt install python3-venv)."; read -r -p "Press Enter…"; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { rm -rf .venv; echo "Installation failed - check the internet connection and try again."; read -r -p "Press Enter…"; exit 1; }
fi
echo "Starting Sky Monitor AI - your browser will open at http://localhost:8501  (Ctrl+C to stop)"
.venv/bin/python -m streamlit run streamlit_app.py
