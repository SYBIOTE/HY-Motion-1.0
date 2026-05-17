#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "${SKIP_CHECKPOINT_PREP:-0}" =~ ^(1|true|yes)$ ]]; then
  exec "$@"
fi

if ! python scripts/ensure_checkpoints.py; then
  exit 1
fi

exec "$@"
