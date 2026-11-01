SKY MONITOR AI - QUICK START
============================

Everything is in this folder: the application, the trained models and the data.

1. ONE-TIME: install Python 3.11 or 3.12 from https://www.python.org/downloads/
   (Windows: tick "Add python.exe to PATH" on the first installer screen.)

2. START THE APP
   Windows : double-click  run_windows.bat
   macOS   : double-click  run_mac.command
             (if macOS blocks it: right-click > Open > Open)
   Linux   : bash run_linux.sh

   The first start installs the required libraries (a few minutes, internet needed once).
   Later starts take a few seconds and work offline.
   Your browser opens at http://localhost:8501 . Keep the black window open while you use
   the app; close it to stop the app.

3. GIVE IT IMAGES - three ways
   a) ONE IMAGE         page "Classify a sky"   -> Upload, Link (Google Drive / web) or Camera snapshot
   b) MANY IMAGES       page "Night timeline & batch" -> upload files or a zip, paste a
                        Google Drive folder link (shared as "Anyone with the link"), or give
                        a folder on this computer
   c) LIVE CAMERA       page "Live camera" -> watch the folder where the camera saves frames,
                        a URL that shows the latest image, or a USB / RTSP camera stream.
                        Switch "Monitoring" on; every new frame is classified automatically.
                        Linux USB camera: enter the path shown by  ls /dev/v4l/by-id/  (…video-index0)

4. WEATHER & SATELLITE (needs internet)
   Choose a place (list, name search or coordinates) and a time: forecast on the left,
   satellite picture on the right, and a comparison with a camera frame.

OUTPUT: Clear / Mostly clear / Mostly cloudy / Overcast / Wet dome, the probability of a clear
sky, estimated cloud cover, sky type, a map of where the sky looks clear or cloudy, a timeline,
and CSV files you can download.

PROBLEMS
- "Python was not found": install Python (step 1), then start again.
- Installation failed: check the internet connection, delete the ".venv" folder, start again.
- Port 8501 busy: close other Streamlit apps.
- Camera not found: try index 1 instead of 0, or check the stream URL in VLC first.

Images: all-sky camera of the Higashi-Hiroshima Observatory, Hiroshima University.
Not for safety-critical use - always check the sky before operating equipment.
