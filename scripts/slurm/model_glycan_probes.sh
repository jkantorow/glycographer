#!/bin/bash

#SBATCH --job-name=glyco-model
#SBATCH --partition=simbiosyslab,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=./glyco_model_%j.out

# ====================== #
# Model minimized free
# glycan probes from
# simplified-IUPAC
# sequences (one PDB per
# probe).
# ====================== #

# A single job, not an array: GlycanSampler takes seconds to minutes per probe,
# so even a batch of dozens fits in one task. stderr goes into the .out file
# (no --error) to keep log inodes down. Quote sequences -- they contain ( ) [ ] >:
#
#   sbatch scripts/slurm/model_glycan_probes.sh probes.iupac -o probes/ \
#       --manifest probes/probes.csv
#   sbatch scripts/slurm/model_glycan_probes.sh \
#       'b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc' -o probes/
#
# Check names first with `model_glycan_probes.py probes.iupac --dry-run`
# (no PyRosetta needed). The job exits non-zero if any probe failed, so a
# dependent job (e.g. --dependency=afterok:$jid) won't start on a partial set.

CONDA_PATH="${CONDA_PATH:-}"
GLYCOGRAPHER_PATH="${GLYCOGRAPHER_PATH:-}"

module purge

## Resolve `GLYCOGRAPHER_PATH` and `CONDA_PATH`, expand ~, and set defaults
if [ -z "$GLYCOGRAPHER_PATH" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    GLYCOGRAPHER_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    case "$GLYCOGRAPHER_PATH" in
        ~*) GLYCOGRAPHER_PATH="${HOME}${GLYCOGRAPHER_PATH#\~}" ;;
    esac
fi

## Prefer an explicitly set CONDA_PATH, otherwise try to discover Conda
if [ -z "$CONDA_PATH" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_PATH="$(conda info --base 2>/dev/null || true)"
    else
        echo "Warning: conda not found; glycographer env may not activate properly"
    fi
fi

# Exit on any error
set -e

if [[ "$#" -eq 0 ]]; then
    echo "No arguments provided. See: model_glycan_probes.py --help"
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/model_glycan_probes.py "$@"

exit 0
