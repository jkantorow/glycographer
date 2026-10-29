#!/bin/bash

#SBATCH --job-name=gdock-merge
#SBATCH --partition=simbiosyslab,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=./gdock_merge_%j.out

# ============================================================ #
# Merge the per-block <run_id>_block<NNN>.tar.xz archives of a
# finished GlycanDock run into one <run_id>.tar.xz, then delete
# the block archives (and any --logs packed in with them).
#
# This wrapper is a thin pass-through: every argument is
# forwarded verbatim to merge_glycandock_blocks.py. See
# `merge_glycandock_blocks.py --help`.
#
# Chain it after the docking array so it starts once every
# task has succeeded:
#
#   jid=$(sbatch --parsable --array=0-41 run_glycandock.sh \
#       complex.pdb -n 1000 --grid grid.pdb --outdir runs/1k9i)
#   sbatch --dependency=afterok:$jid merge_glycandock_blocks.sh \
#       runs/1k9i --logs "gdock_${jid}_*.out"
#
# If any array task fails, afterok is never satisfied and this
# job stays pending (scancel it). Rerun the failed blocks with
# --resume, then submit the merge by hand.
#
# Resources: xz compresses on --cpus-per-task threads, using
# ~100 MB of memory per thread at the default preset. Reading
# the block archives back is fast; compression dominates.
#
# Under sbatch, BASH_SOURCE points at Slurm's spooled copy of this
# script, so export GLYCOGRAPHER_PATH=<repo root> before submitting
# if the default below does not resolve to the repo.
# ============================================================ #

CONDA_PATH="${CONDA_PATH:-}"
GLYCOGRAPHER_PATH="${GLYCOGRAPHER_PATH:-}"

module purge

## Resolve `GLYCOGRAPHER_PATH`, expand ~, and set defaults
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
    echo "No arguments provided. See: merge_glycandock_blocks.py --help"
    exit 1
fi

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/utils/merge_glycandock_blocks.py "$@"

exit 0
