#!/usr/bin/env bash
# Launch the RoboCup labeling app from its local virtual environment.
# Optional: ROBOCUP_DEVICE=cpu ./start_app.sh  to force the CPU instead of the Apple GPU.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Run ./setup_mac.sh first." >&2
    exit 1
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python run_app.py "$@"
