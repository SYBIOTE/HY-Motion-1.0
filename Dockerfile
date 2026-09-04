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

# The base image bakes hymotion/ and scripts/ as of whenever it was last built,
# so anything this branch changes in them is invisible at runtime unless it is
# copied again here. Overwrite both, or edits silently never reach a worker.
COPY hymotion/ hymotion/
COPY scripts/ensure_checkpoints.py /app/scripts/ensure_checkpoints.py

COPY api.py handler.py ./

ENV DISABLE_WOODEN_MESH=1

CMD ["python", "-u", "handler.py"]
