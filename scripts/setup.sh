#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEMO_PYTHON="${PYTHON_BIN:-python3}"
"$DEMO_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required; 3.11 recommended"'
"$DEMO_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'numpy>=1.26,<3'
if [[ "$(uname -s)" == "Linux" ]]; then
  .venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install 'torch>=2.6,<3'
fi
.venv/bin/python -c 'import torch, numpy; print("Ready:", torch.__version__, "· CPU only in this demo")'
echo 'Start: .venv/bin/python app.py --mode slurm --host 0.0.0.0 --port 8000'
