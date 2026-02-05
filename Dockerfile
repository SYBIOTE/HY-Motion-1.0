# HY-Motion API (production)
# Build: docker build -t hymotion-api .
# Run:   docker run --gpus all -p 8080:8080 hymotion-api
# Cloud Build: gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=us-central1

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Dependencies (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Pre-download model (cached layer — only re-runs if this line changes)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='tencent/HY-Motion-1.0', allow_patterns='HY-Motion-1.0-Lite/*', local_dir='/app/ckpts/tencent'); \
"

# App code (changes here don't re-download deps or model)
COPY hymotion/ hymotion/
COPY stats/ stats/
COPY api.py .

ENV MODEL_PATH=/app/ckpts/tencent/HY-Motion-1.0-Lite \
    QWEN_QUANTIZATION=int4 \
    DISABLE_PROMPT_ENGINEERING=True \
    USE_HF_MODELS=1 \
    DISABLE_WOODEN_MESH=1 \
    PORT=8080

EXPOSE 8080

# Note: Cloud Run ignores HEALTHCHECK (uses its own probes). Kept for local Docker.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Cloud Run sets PORT at runtime; exec for proper SIGTERM handling
CMD exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
