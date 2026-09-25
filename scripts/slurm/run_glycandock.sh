#!/bin/bash

#SBATCH --job-name=glycandock
#SBATCH --partition=short
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=./gdock_%A_%a.out
#SBATCH --error=./gdock_%A_%a.err

# ====================== #
# Run the GlycanDock
# protocol on an input
# receptor-glycoligand
# complex, as a SLURM
# job array.
# ====================== #

# Each array task is one independent process that generates one block of
# decoys. GlycanDock samples are independent, so nothing is shared between
# tasks and no communication is needed -- the scheduler is the whole
# parallelization layer.
#
# One core per task is deliberate. PyRosetta runs this protocol
# single-threaded, and the `short` and `sharing` partitions cap a single job at
# MaxNodes=2 -- but an array is N separate jobs, so the array as a whole is not
# bound by that limit. Many small tasks also backfill into idle slots faster
# than a few large ones.
#
# --array is set to 0-0 here so a bare `sbatch` runs a single task. Override it
# at submit time, and pass a matching --n-blocks (or rely on
# $SLURM_ARRAY_TASK_COUNT, which run_glycandock.py reads automatically):
#
#   sbatch --array=0-41 scripts/slurm/run_glycandock.sh \
#       complex.pdb -n 1000 --grid grid.pdb
#
# Size blocks so each task runs 1-4 hours; check the plan first with
# `run_glycandock.py ... --dry-run`.
#
# Partition notes for this cluster:
#   simbiosyslab  lab-owned, PriorityTier=10, 30 day limit, 768 cores. Primary.
#   short         default, 2 day limit, ~9000 cores. Good general overflow.
#   sharing       ~22000 cores but a hard 1 hour limit -- size blocks to
#                 finish in ~45 minutes with margin.
# Throttle concurrent tasks on shared partitions with e.g. --array=0-41%20.
#
# Logs are written per task via %A_%a (job id, array index). Without that,
# every task in the array would clobber the same file.

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
    echo "No arguments provided. See: run_glycandock.py --help"
    exit 1
fi

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/run_glycandock.py "$@"

exit 0
