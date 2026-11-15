#!/usr/bin/env bash
# Called inside an allocation, on its first node. No sbatch and no shared model
# directory: FEDSIMGH_WORKER_CONFIG is inherited by srun, not read on each node.
set -euo pipefail
: "${DEMO_PROJECT:?}" "${DEMO_PYTHON:?}" "${DEMO_RUN_DIR:?}"
test "${SLURM_NTASKS:-0}" -eq 3
test "${SLURM_JOB_NUM_NODES:-0}" -eq 3
DEMO_MASTER="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)"
test -n "$DEMO_MASTER"
export DEMO_MASTER DEMO_PROJECT DEMO_PYTHON DEMO_RUN_DIR
FEDSIMGH_WORKER_CONFIG="$("$DEMO_PYTHON" - <<'PY'
import json, os, secrets
from pathlib import Path
from faultdemo.io import atomic_json
config = json.loads(os.environ["FEDSIMGH_WORKER_CONFIG"])
port = int(os.environ.get("FEDSIMGH_TCP_PORT", 20000 + int(os.environ["SLURM_JOB_ID"]) % 30000))
timeout = float(os.environ.get("FEDSIMGH_TCP_TIMEOUT", config.get("tcp_timeout", 120)))
if not 1024 <= port <= 65535 or not 1 <= timeout <= 3600:
    raise SystemExit("Invalid TCP port or timeout")
config.update(transport="tcp", tcp_host=os.environ["DEMO_MASTER"], tcp_bind="0.0.0.0",
              tcp_port=port, tcp_timeout=timeout)
config.setdefault("tcp_token", secrets.token_hex(32))
config.setdefault("tcp_run_id", Path(os.environ["DEMO_RUN_DIR"]).name)
path = Path(os.environ["DEMO_RUN_DIR"]) / "config.json"
atomic_json(path, config)
path.chmod(0o600)
print(json.dumps(config))
PY
)"
export FEDSIMGH_WORKER_CONFIG
# Preserve exported config even when the enclosing sbatch used --export=NONE.
export SLURM_EXPORT_ENV=ALL
echo "FedSimGH TCP | coordinator=$DEMO_MASTER | 3 CPU nodes | no shared weight files"
echo "Rank 0 result directory: $DEMO_RUN_DIR"
# Rank 1/2 use node-local temporary directories, even if the project is shared.
# Rank 0 retains results. Code + data + a Linux Python environment must be
# installed at DEMO_PROJECT / DEMO_PYTHON separately on all three nodes.
srun --kill-on-bad-exit=1 --cpu-bind=cores --distribution=block:block \
  --nodes=3 --ntasks=3 --ntasks-per-node=1 --label \
  bash -c '
    set -euo pipefail
    cd "$DEMO_PROJECT"
    if [[ "$SLURM_PROCID" == 0 ]]; then
      task_run="$DEMO_RUN_DIR"
    else
      task_run="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/fedsimgh-${SLURM_JOB_ID}-${SLURM_PROCID}-XXXXXX")"
    fi
    exec "$DEMO_PYTHON" -u -m faultdemo.worker --run-dir "$task_run" --config-env
  ' 2>&1 | tee "$DEMO_RUN_DIR/console.log"
