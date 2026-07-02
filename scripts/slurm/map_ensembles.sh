#!/bin/bash

#SBATCH --job-name=gdock-map
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./gdock_map.out
#SBATCH --error=./gdock_map.err

# ============================================================ #
# Generate volumetric maps (and optional cross-probe consensus
# maps) for one or more GlycanDock ensembles docked against the
# same receptor.
#
# This wrapper is a thin pass-through: every argument after the
# script name is forwarded verbatim to map_ensembles.py, so the
# full Python CLI is available without maintaining a parallel
# copy of each flag here. See `map_ensembles.py --help`.
#
# Examples:
#   # Same boltzmann map (beta sweep) for three probes + consensus:
#   sbatch map_ensembles.sh runs/diSia runs/LacNAc runs/Man3 \
#       --grid-pdb 1o7v_receptor.pdb --beta 0.3 0.5 1.0 \
#       --outdir maps --consensus mean best_probe support
#
#   # Heterogeneous per-ensemble parameters via a job spec:
#   sbatch map_ensembles.sh --config jobs.json
# ============================================================ #

CONDA_PATH="${CONDA_PATH:-}"
GLYCOGRAPHER_PATH="${GLYCOGRAPHER_PATH:-}"

module purge

## Resolve `GLYCOGRAPHER_PATH`, expand ~, and set defaults
if [ -z "$GLYCOGRAPHER_PATH" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    GLYCOGRAPHER_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
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
    echo "No arguments provided. See: map_ensembles.py --help"
    exit 1
fi

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/utils/map_ensembles.py "$@"

exit 0
