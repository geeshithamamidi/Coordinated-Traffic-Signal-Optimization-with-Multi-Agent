#!/usr/bin/env bash
# One training run per line of hpc/jobs.txt.
# Submit from the repository root, after mkdir -p hpc/logs:
#   sbatch --array=1-$(wc -l < hpc/jobs.txt) hpc/train_array.slurm

#SBATCH --job-name=traffic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%A_%a.out

# TODO: set the partition name for this course allocation. sinfo lists the queues.
#SBATCH --partition=TODO_PARTITION

# TODO: set the SLURM account for this course allocation.
#SBATCH --account=TODO_ACCOUNT

# Placeholder. Override at submit time with --array=1-$(wc -l < hpc/jobs.txt).
#SBATCH --array=1-1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  module load anaconda/3.9
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate traffic

export LIBSUMO_AS_TRACI=1
if [[ -z "${SUMO_HOME:-}" ]]; then
  SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"
  export SUMO_HOME
fi

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/jobs.txt)"
if [[ -z "${LINE}" ]]; then
  echo "No job for array task ${SLURM_ARRAY_TASK_ID} in hpc/jobs.txt" >&2
  exit 1
fi

read -r METHOD SCENARIO DEMAND SEED <<< "${LINE}"
echo "task ${SLURM_ARRAY_TASK_ID}: method=${METHOD} scenario=${SCENARIO} demand=${DEMAND} seed=${SEED}"

python scripts/run_experiment.py \
  --method "${METHOD}" \
  --scenario "${SCENARIO}" \
  --demand "${DEMAND}" \
  --seed "${SEED}"
