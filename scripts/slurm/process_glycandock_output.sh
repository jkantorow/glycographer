#!/bin/bash

#SBATCH --job-name=gdock-process
#SBATCH --partition=simbiosyslab,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=./gdock_process_%j.out
#SBATCH --error=./gdock_process_%j.err

# ============================================================ #
# Process a GlycanDock output pose directory: load the poses
# into a GlycanDockEnsemble, cluster the docked ligand poses,
# and dump scores, the cluster summary, the ensemble PDB, and
# per-residue energies.
#
# This wrapper is a thin pass-through: every argument is
# forwarded verbatim to process_glycandock_output.py, so the
# full Python CLI is available without maintaining a parallel
# copy of each flag here. See `process_glycandock_output.py --help`.
#
# Examples:
#   # Parse + cluster a full-glycoligand run:
#   sbatch process_glycandock_output.sh runs/1k9i_full \
#       --outdir processed --cluster-cutoff 2.0 --min-cluster-size 6
#
#   # Re-cluster processed output at another cutoff (no pose parsing):
#   sbatch process_glycandock_output.sh \
#       --ensemble-pdb processed/1k9i_full_5000p_ensemble.pdb \
#       --scores-csv processed/1k9i_full_scoredata.csv \
#       --outdir processed/c1p5 --cluster-cutoff 1.5
#
# Resources: clustering holds an n x n float32 RMSD matrix plus an
# n x n boolean neighbor matrix (~125 MB at 5000 poses, ~2 GB at
# 20000). --mem=8G covers that plus pose parsing with headroom;
# raise it for much larger ensembles. The RMSD matrix product is
# multithreaded by numpy's BLAS, which uses --cpus-per-task.
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

#source "$CONDA_PATH/etc/profile.d/conda.sh"
#conda activate glycographer

# Exit on any error
set -e

if [[ "$#" -eq 0 ]]; then
    echo "No arguments provided. See: process_glycandock_output.py --help"
    exit 1
fi

# Let numpy's BLAS use the allocated cores for the RMSD matrix product.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/utils/process_glycandock_output.py "$@"

exit 0
