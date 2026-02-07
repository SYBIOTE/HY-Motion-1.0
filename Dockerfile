# HY-Motion API (production)
# Tiny layer on top of the shared base — rebuilds in seconds when only app code changes.
#
# Build (standalone — if not using the base workflow):
#   docker build -f Dockerfile.base -t hymotion-base . && docker build -t hymotion-api .
#
# Run:
#   docker run --gpus all -p 8080:8080 hymotion-api
#
# Cloud Build:
#   gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=us-central1

# ── The base image tag is overridden by cloudbuild.yaml via --build-arg ──
ARG BASE_IMAGE=hymotion-base:latest
FROM ${BASE_IMAGE}

# ── API-specific deps (fastapi, uvicorn — lightweight) ──
COPY requirements-api.txt .
RUN uv pip install --system --no-cache -r requirements-api.txt

# ── App code ──
COPY api.py .

ENV DISABLE_WOODEN_MESH=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

CMD exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
