#!/usr/bin/env bash
#SBATCH --job-name=faultdiag-cpu
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:20:00

set -euo pipefail
DEMO_RUN_DIR="${1:?Missing shared run directory}"
DEMO_PYTHON="${2:?Missing shared Python interpreter}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "Job=${SLURM_JOB_ID} Nodes=${SLURM_JOB_NODELIST} Tasks=${SLURM_NTASKS}"
echo "Shared working directory: ${PWD}"
test "${SLURM_NTASKS}" -eq 3
test "${SLURM_JOB_NUM_NODES}" -eq 3
if [[ -n "${FEDSIMGH_WORKER_CONFIG:-}" ]]; then
  # Auto-submit control-plane mode: only rank 0 results must be visible to the
  # controller. Rank 1/2 never read that directory; all updates travel via TCP.
  export DEMO_PROJECT="$PWD" DEMO_PYTHON DEMO_RUN_DIR
  exec bash "$DEMO_PROJECT/slurm/tcp-tasks.sh"
fi
# One process on each of the three nodes. Rank 0 also performs aggregation.
# No GPU, torch.distributed port, ssh daemon or fourth compute node is required.
srun --kill-on-bad-exit=1 --cpu-bind=cores --output="${DEMO_RUN_DIR}/rank-%t.log" \
  "$DEMO_PYTHON" -u -m faultdemo.worker --run-dir "$DEMO_RUN_DIR"
