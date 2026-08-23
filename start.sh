#!/usr/bin/env sh
# -----------------------------------------------------------------------------
# Railpack 0.37+ probe stub. Railway's Railpack runs a pre-flight analysis on
# the source tree looking for a `start.sh` (legacy Nixpacks convention). Even
# though we deploy via Dockerfile + railway.json, Railpack still complains if
# this file is missing, which causes the build to abort before our Dockerfile
# is read.
#
# This file is NOT used at runtime. The actual entrypoint is the Dockerfile's
# `CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]`
# which Railway runs inside the built container.
#
# If you ever switch off the Dockerfile builder and want Railpack / Nixpacks
# to actually run this script, replace the body with:
#     exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
# -----------------------------------------------------------------------------
echo "start.sh invoked — but the real entrypoint is the Dockerfile CMD."
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"