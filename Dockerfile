# HY-Motion API (production) — RunPod branch
# Thin layer on Dockerfile.base: FastAPI + uvicorn.
#
# Build from HY-Motion-1.0/ (monorepo subdir) so COPY paths resolve.
# Build arg HYMOTION_BASE_IMAGE: base to extend (default: Docker Hub sybiote/hymotion-base:latest).
#
#   docker build -f Dockerfile.base -t hymotion-base .
#   docker build --build-arg HYMOTION_BASE_IMAGE=hymotion-base:latest -f Dockerfile -t hymotion-api .
#   docker run --gpus all -p 8080:8080 hymotion-api

ARG HYMOTION_BASE_IMAGE=docker.io/sybiote/hymotion-base:latest
FROM ${HYMOTION_BASE_IMAGE}

COPY requirements-api.txt .
RUN uv pip install --system --no-cache -r requirements-api.txt

COPY api.py .

ENV DISABLE_WOODEN_MESH=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=1200s --retries=3 \
    CMD bash -lc 'curl -fsS "http://127.0.0.1:${PORT:-8080}/health" >/dev/null || exit 1'

CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
