#!/usr/bin/env bash
# Starts the Streamlit backend (port 8501) and the Vite frontend together.
# Usage: ./run.sh      Stop both with Ctrl+C.

set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$(cd "$BACKEND_DIR/../heart-mri-web" && pwd)"

# --- Backend setup (first run only) ---
if [ ! -x "$BACKEND_DIR/venv/bin/python" ]; then
  echo "Creating Python venv..."
  python3 -m venv "$BACKEND_DIR/venv"
  "$BACKEND_DIR/venv/bin/python" -m pip install -r "$BACKEND_DIR/requirements.txt" streamlit
fi

# --- Frontend setup (first run only) ---
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

# Stop both servers (and everything they spawned) when this script exits.
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Shutting down..."
  kill 0 2>/dev/null
}
trap cleanup INT TERM EXIT

echo "Starting backend  -> http://localhost:8501"
(cd "$BACKEND_DIR" && exec ./venv/bin/python -m streamlit run app.py \
  --server.port 8501 --server.headless true \
  --server.fileWatcherType none --browser.gatherUsageStats false) &

echo "Starting frontend -> http://localhost:5173"
(cd "$FRONTEND_DIR" && exec npx vite --host 127.0.0.1 --port 5173 --strictPort) < /dev/null &

# Wait until both servers answer, then open the site.
echo "Waiting for servers..."
until curl -s -o /dev/null http://127.0.0.1:8501 && curl -s -o /dev/null http://127.0.0.1:5173; do
  sleep 1
done
echo
echo "Ready!  Website: http://localhost:5173   Classifier: http://localhost:8501"
echo "Press Ctrl+C to stop."
open http://localhost:5173

wait
