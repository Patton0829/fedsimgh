#!/usr/bin/env bash
#SBATCH --job-name=fedsimgh-system
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.log

# This script runs inside the system's existing allocation. It never calls sbatch.
# When pasted into the system editor, set FEDSIMGH_PROJECT to the shared checkout
# and change --output to the job workspace's ${id}/output/%j.log placeholder.
set -euo pipefail
DEMO_PROJECT="${FEDSIMGH_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEMO_OUTPUT="${FEDSIMGH_OUTPUT:-${DEMO_PROJECT}/runtime/system}"
DEMO_PYTHON="${FEDSIMGH_PYTHON:-${DEMO_PROJECT}/.venv/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

if [[ "${SLURM_NTASKS:-0}" != 3 || "${SLURM_JOB_NUM_NODES:-0}" != 3 ]]; then
  echo 'This launcher requires an existing Slurm allocation: 3 nodes, 3 tasks, 1 task/node.' >&2
  exit 2
fi
cd "$DEMO_PROJECT"
DEMO_RUN_DIR="$("$DEMO_PYTHON" - "$DEMO_OUTPUT" <<'PY'
import math
import os
import sys
import time
import uuid
from pathlib import Path
from faultdemo.io import atomic_json

config = {"mode": "slurm", "rounds": int(os.environ.get("FEDSIMGH_ROUNDS", "300")),
          "epochs": int(os.environ.get("FEDSIMGH_EPOCHS", "1")),
          "lr": float(os.environ.get("FEDSIMGH_LR", "0.03")),
          "tau": float(os.environ.get("FEDSIMGH_TAU", "3")),
          "history_lambda": float(os.environ.get("FEDSIMGH_HISTORY_LAMBDA", "0.6"))}
limits = {"rounds": (1, 1000), "epochs": (1, 3), "lr": (.0001, .2),
          "tau": (.1, 10), "history_lambda": (0, .99)}
for name, (low, high) in limits.items():
    if not math.isfinite(config[name]) or not low <= config[name] <= high:
        raise SystemExit(f"Invalid {name}: expected {low}..{high}")
run = Path(sys.argv[1]).resolve() / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6])
run.mkdir(parents=True)
atomic_json(run / "config.json", config)
atomic_json(run / "state.json", {"status": "queued", "round": 0, "rounds": config["rounds"],
                                "metrics": [], "nodes": [], "predictions": 0})
atomic_json(run / "submission.json", {"mode": "slurm", "job_id": os.environ.get("SLURM_JOB_ID"),
                                     "launcher": "system.sbatch"})
print(run)
PY
)"
echo "FedSimGH | job=${SLURM_JOB_ID} | shared project=$DEMO_PROJECT"
echo "Output directory: $DEMO_RUN_DIR"
# Stream all ranks to the enclosing Slurm job log so the existing system log
# viewer can display training and per-sample diagnosis without another web server.
srun --kill-on-bad-exit=1 --cpu-bind=cores --label \
  "$DEMO_PYTHON" -u -m faultdemo.worker --run-dir "$DEMO_RUN_DIR" \
  2>&1 | tee "$DEMO_RUN_DIR/console.log"

"$DEMO_PYTHON" - "$DEMO_RUN_DIR" <<'PY'
import sys
from pathlib import Path
from faultdemo.io import read_json
run = Path(sys.argv[1])
state = read_json(run / "state.json", {})
if state.get("status") != "completed":
    raise SystemExit("Workers did not complete; inspect console.log and state.json")
metrics = state["final_metrics"]
print(f"Final accuracy: {metrics['accuracy']:.1%} | Macro-F1: {metrics['macro_f1']:.3f}")
print(f"Predictions: {run / 'predictions.csv'}")
print(f"Model: {run / 'model.pt'}")
PY
