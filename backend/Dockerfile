# Fraud Network Mapper — backend container
# Targets: Railway / Render / Fly.io / any OCI-compatible host.
#
# Build context MUST be the repo root (so we can COPY backend/, models/,
# data/, scripts/ as siblings). Run with:
#   docker build -t fraud-mapper-api -f backend/Dockerfile .
#
# Image is ~1.2 GB (CPU-only PyTorch + reportlab + scikit-learn).
# Healthcheck hits /health which returns model_loaded + session count.

FROM python:3.11-slim

# System deps: libgomp1 = OpenMP runtime (PyTorch needs it for CPU inference),
# curl = for the HEALTHCHECK probe. --no-install-recommends keeps the image small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch FIRST so we don't accidentally grab the GPU build
# (default torch wheel pulls CUDA libs — ~2 GB wasted). Then install the rest
# of the stack from requirements.txt (which lists plain `torch>=2.0`).
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
        torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy the runtime surface (siblings of WORKDIR /app so PROJECT_ROOT
# math in backend/core.py:25 resolves correctly to /app).
COPY backend/        ./backend/
COPY models/         ./models/
COPY data/samples/   ./data/samples/
COPY scripts/        ./scripts/

# Defaults that Railway/Render override at runtime via the platform's
# $PORT injection. ALLOWED_ORIGINS=* is the hackathon-demo-friendly default
# — set it to a comma-separated URL list in production to lock CORS down.
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ALLOWED_ORIGINS="*"

EXPOSE 8000

# Healthcheck: /health returns 200 with {status:"ok", model_loaded:bool}.
# Start period 60s gives the model time to load + the sample dataset to warm.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]