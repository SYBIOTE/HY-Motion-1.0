# HY-Motion API (production)
# Tiny layer on top of the shared base — rebuilds in seconds when only app code changes.
#
# Build (standalone — if not using the base workflow):
#   docker build -f Dockerfile.base -t hymotion-base . && docker build -t hymotion-api .
#
# Run:
#   docker run --gpus all -p 8080:8080 hymotion-api
#
# Run with persisted checkpoints:
#   docker run --gpus all -v hymotion-ckpts:/app/ckpts -p 8080:8080 hymotion-api
#
# Cloud Build:
#   gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=us-central1
#
# RunPod (Git build of this repo): default BASE_IMAGE is Docker Hub — no local hymotion-base tag.
# Local builds on top of a freshly built base:
#   docker build --build-arg BASE_IMAGE=hymotion-base:latest ...

# ── Base (cloudbuild.yaml overrides with Artifact Registry) ──
ARG BASE_IMAGE=docker.io/sybiote/hymotion-base:latest
FROM ${BASE_IMAGE}

# ── API-specific deps (fastapi, uvicorn — lightweight) ──
COPY requirements-api.txt .
RUN uv pip install --system --no-cache -r requirements-api.txt

# ── App code ──
COPY api.py .

ENV DISABLE_WOODEN_MESH=1

EXPOSE 8080

# Long start-period allows first-ever HF checkpoint sync into an empty volume.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1200s --retries=3 \
    CMD bash -lc 'curl -fsS "http://127.0.0.1:${PORT:-8080}/health" >/dev/null || exit 1'

CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
