#!/bin/bash

#SBATCH --job-name=glyco-grid
#SBATCH --partition=simbiosyslab,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=./glyco_grid_%j.out

# ====================== #
# Build a GlycanDock
# sampling grid over a
# receptor surface.
# ====================== #

# A single job, not an array: a global grid on a few thousand atoms takes
# well under a minute. The work is numpy/scipy only -- no Rosetta, no open3d,
# no surface mesh -- so this also runs fine on a workstation; the wrapper
# exists mainly to keep grid building inside a dependency chain:
#
#   sbatch scripts/slurm/build_sampling_grid.sh receptors/1pmh_rec.pdb -o grids/
#   sbatch scripts/slurm/build_sampling_grid.sh receptors/tetra.pdb -o grids/ \
#       -g crd --site 'chain A and resn CA' --site-radius 14
#
# --cpus-per-task feeds the KD-tree queries (scipy workers=-1); --mem covers
# the geometry lattice at ~50 bytes/voxel (the default cap is 40M voxels, so
# 2 GB; 8G leaves room for a finer --lattice-spacing).
#
# Quote PyMOL selections -- they contain spaces and parentheses. Check one
# first with `build_sampling_grid.py rec.pdb --site '...' --dry-run`, which
# prints the residues it matched and the lattice size without building.

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
    echo "No arguments provided. See: build_sampling_grid.py --help"
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/build_sampling_grid.py "$@"

exit 0
