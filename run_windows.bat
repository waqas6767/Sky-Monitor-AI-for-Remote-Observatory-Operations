@echo off
title Sky Monitor AI
cd /d "%~dp0"

rem ---- find Python (the "py" launcher first, then "python")
set "PY=py -3"
py -3 --version >nul 2>nul
if errorlevel 1 set "PY=python"
%PY% --version >nul 2>nul
if errorlevel 1 goto nopython

rem ---- skip Streamlit's one-time e-mail question
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

if exist ".venv\Scripts\python.exe" goto run
echo.
echo  First start: installing Sky Monitor AI. This takes a few minutes and needs internet once.
echo.
%PY% -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

:run
echo.
echo  Starting Sky Monitor AI - your browser will open at http://localhost:8501
echo  Keep this window open while you use the app. Close it to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
pause
exit /b

:nopython
echo.
echo  Python was not found on this computer.
echo  1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
echo  2. During installation tick "Add python.exe to PATH"
echo  3. Double-click run_windows.bat again
start https://www.python.org/downloads/
pause
exit /b

:fail
echo.
echo  Installation failed. Check the internet connection and that Python is version 3.10 to 3.13,
echo  then double-click run_windows.bat again.
rmdir /s /q .venv 2>nul
pause
