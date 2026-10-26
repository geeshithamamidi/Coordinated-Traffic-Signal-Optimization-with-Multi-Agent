#!/usr/bin/env bash
# Create the traffic conda env used by the SLURM array.
# Load the cluster Anaconda module before running this script (see hpc/README.md).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  module load anaconda/3.9
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n traffic python=3.11 -y
conda activate traffic
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# LIBSUMO_AS_TRACI imports libsumo. Pin it to the same SUMO release as requirements.txt.
python -m pip install "libsumo==1.27.1"

SUMO_HOME_VALUE="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"
conda env config vars set SUMO_HOME="$SUMO_HOME_VALUE" LIBSUMO_AS_TRACI=1

echo "Created conda env traffic with SUMO_HOME=${SUMO_HOME_VALUE} and LIBSUMO_AS_TRACI=1."
echo "Deactivate and activate traffic so those variables are exported."
