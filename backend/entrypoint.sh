#!/usr/bin/env sh
# -----------------------------------------------------------------------------
# Container entrypoint.
#
# Railway injects $PORT at runtime. We bind uvicorn to that port (defaulting
# to 8080 — Railway's convention — if unset, which happens for local
# `docker run` without `-e PORT=...`).
#
# Using `exec` so the uvicorn process replaces the shell — that way
# SIGTERM from Railway's shutdown handler reaches uvicorn directly and
# gives it a chance to drain in-flight requests. (Shell-form CMD swallows
# signals, causing 30-second hard kills on every redeploy.)
# -----------------------------------------------------------------------------
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "[entrypoint] starting uvicorn on ${HOST}:${PORT}"
echo "[entrypoint] PYTHONPATH=/app, cwd=$(pwd)"

# Make sure /app is on the Python path so `import backend.main` resolves
# whether or not uvicorn was invoked from /app.
export PYTHONPATH="/app:${PYTHONPATH:-}"
cd /app

exec uvicorn backend.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level info