#!/usr/bin/env bash
# One-time setup of the RoboCup labeling app on macOS (Apple Silicon or Intel).
# Creates a local virtual environment in .venv with everything needed for
# SAM2-assisted mask annotation and COCO export.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
            PYTHON="$candidate"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Python 3.10+ is required. Install it with: brew install python@3.12" >&2
    exit 1
fi
echo "Using $($PYTHON --version) at $(command -v $PYTHON)"

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
    torch torchvision \
    PyQt6 \
    "hydra-core>=1.3.2" "iopath>=0.1.10" natsort tqdm \
    numpy pillow pillow-heif pycocotools

.venv/bin/python - <<'PY'
import torch
mps = getattr(torch.backends, "mps", None)
print("torch", torch.__version__, "| Apple GPU (MPS) available:", bool(mps and mps.is_available()))
PY
echo
echo "Setup done. Start the app with:  ./start_app.sh"
