#!/usr/bin/env bash
# run_demo.sh
#
# Convenience script to start the MedAI API server and Streamlit demo
# interface together.
#
# Usage:
#   bash scripts/run_demo.sh
#
# Environment variables:
#   MEDAI_API_KEY  — optional API key for auth (default: unset / disabled)
#   GOOGLE_API_KEY — optional Gemini key for LLM mode (default: unset)
#   API_PORT       — FastAPI server port (default: 8000)
#   DEMO_PORT      — Streamlit server port (default: 8501)

set -euo pipefail

API_PORT="${API_PORT:-8000}"
DEMO_PORT="${DEMO_PORT:-8501}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "  MedAI Clinical Intelligence Platform"
echo "============================================================"
echo "  Repo root : $REPO_ROOT"
echo "  API port  : $API_PORT"
echo "  Demo port : $DEMO_PORT"
echo "============================================================"
echo ""

# ── Start API server in background ──────────────────────────────────────────
echo "Starting MedAI API server on port $API_PORT ..."
cd "$REPO_ROOT"
python -m uvicorn medai.src.api.main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload \
    &
API_PID=$!
echo "  API server PID: $API_PID"

# Wait for the API to become ready
echo "  Waiting for API to be ready..."
for i in $(seq 1 15); do
    if curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "  API is ready."
        break
    fi
    sleep 1
done

echo ""

# ── Start Streamlit demo ─────────────────────────────────────────────────────
echo "Starting MedAI Streamlit demo on port $DEMO_PORT ..."
echo "  Open: http://localhost:$DEMO_PORT"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Trap SIGINT/SIGTERM so the API process is also killed on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$API_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

python -m streamlit run \
    "$REPO_ROOT/medai/src/demo/app.py" \
    --server.port "$DEMO_PORT" \
    --server.address 0.0.0.0

# If streamlit exits normally, clean up the API process
kill "$API_PID" 2>/dev/null || true
