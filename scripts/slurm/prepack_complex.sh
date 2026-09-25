#!/bin/bash

#SBATCH --job-name=gdock-prepack
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=./gdock_prepack_%j.out
#SBATCH --error=./gdock_prepack_%j.err

# ====================== #
# Run GlycanDock Stage 0
# pre-packing once on a
# receptor-glycoligand
# complex, producing a
# prepacked structure for
# a whole docking run.
# ====================== #

# This is a single job, not an array: Stage 0 runs once per complex. It is a
# separate job from run_glycandock.sh because prepacking needs finer rotamer
# sampling (-ex1 -ex2 -ex3 -ex4 -ex1aro -ex2aro, in
# config/glycandock_stage_0.init) than refinement (-ex1 -ex2), and Rosetta
# takes its options once per process -- so the two stages cannot share one.
#
# Chain it in front of the sampling array with a dependency:
#
#   PREPACK=$(sbatch --parsable scripts/slurm/prepack_complex.sh \
#       complex.pdb -o complex_prepacked.pdb)
#   sbatch --dependency=afterok:$PREPACK --array=0-41 \
#       scripts/slurm/run_glycandock.sh complex_prepacked.pdb \
#       -n 1000 --grid grid.pdb --prepack-mode once
#
# NOTE: --prepack-mode once departs from how every existing glycographer
# ensemble was generated (in-loop prepacking under coarser flags). Validate
# before mixing the two in one analysis.

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
    echo "No arguments provided. See: prepack_complex.py --help"
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/prepack_complex.py "$@"

exit 0
