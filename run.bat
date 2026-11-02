@echo off
REM Starts the Streamlit backend (port 8501) and the Vite frontend (Windows).
REM Usage: run.bat  (or .\run.bat in PowerShell)
REM Each server opens in its own window; close them to stop.
REM Needs Python 3.10+ and Node.js 20+ installed and on PATH.

set "BACKEND_DIR=%~dp0"
set "FRONTEND_DIR=%~dp0..\heart-mri-web"
REM Separate from the macOS "venv" folder so the two never clash.
set "VENV=%BACKEND_DIR%venv-win"

if not exist "%FRONTEND_DIR%\package.json" (
  echo Frontend not found at "%FRONTEND_DIR%".
  echo Put the heart-mri-web folder next to this project folder.
  pause
  exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating Python environment - first run only, this takes a while...
  python -m venv "%VENV%" || (echo Python not found. Install it from python.org & pause & exit /b 1)
  "%VENV%\Scripts\python" -m pip install --upgrade pip
  "%VENV%\Scripts\python" -m pip install -r "%BACKEND_DIR%requirements.txt" streamlit
)

REM node_modules copied from a Mac has no Windows binaries (no vite.cmd), so reinstall.
if not exist "%FRONTEND_DIR%\node_modules\.bin\vite.cmd" (
  echo Installing frontend dependencies - first run only...
  if exist "%FRONTEND_DIR%\node_modules" rmdir /s /q "%FRONTEND_DIR%\node_modules"
  if exist "%FRONTEND_DIR%\node_modules.nosync" rmdir /s /q "%FRONTEND_DIR%\node_modules.nosync"
  pushd "%FRONTEND_DIR%"
  call npm install || (echo npm failed. Install Node.js from nodejs.org & popd & pause & exit /b 1)
  popd
)

start "Heart MRI backend" /d "%BACKEND_DIR%" cmd /k ""%VENV%\Scripts\python" -m streamlit run app.py --server.port 8501 --server.headless true --server.fileWatcherType none"
start "Heart MRI frontend" /d "%FRONTEND_DIR%" cmd /k "npx vite --host 127.0.0.1 --port 5173 --strictPort --open"

echo.
echo Backend:  http://localhost:8501
echo Frontend: http://localhost:5173  (opens in your browser)
