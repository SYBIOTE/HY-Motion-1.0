# HY-Motion — RunPod Serverless (queue) worker.
#
# Build from HY-Motion-1.0/ (monorepo subdir) so COPY paths resolve.
# Build arg HYMOTION_BASE_IMAGE: base to extend (default: Docker Hub sybiote/hymotion-base:latest).
#
#   docker build -f Dockerfile.base -t hymotion-base .
#   docker build --build-arg HYMOTION_BASE_IMAGE=hymotion-base:latest -f Dockerfile -t hymotion-queue .
#
# No port is exposed and there is no HTTP healthcheck: the worker pulls jobs off
# the RunPod queue rather than serving requests, so readiness is the SDK
# connecting, not a socket accepting.

ARG HYMOTION_BASE_IMAGE=docker.io/sybiote/hymotion-base:latest
FROM ${HYMOTION_BASE_IMAGE}

COPY requirements-queue.txt .
RUN uv pip install --system --no-cache -r requirements-queue.txt

# ensure_checkpoints.py is baked into the base image, so the base's stale copy
# runs at the entrypoint unless it is overwritten here — an edit on this branch
# would otherwise never reach a running container.
COPY scripts/ensure_checkpoints.py /app/scripts/ensure_checkpoints.py

COPY api.py handler.py ./

ENV DISABLE_WOODEN_MESH=1

CMD ["python", "-u", "handler.py"]
