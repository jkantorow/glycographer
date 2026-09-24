#!/bin/bash

#SBATCH --job-name=glycandock
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=./gdock.out
#SBATCH --error=./gdock.err

# ====================== #
# Run the GlycanDock
# protocol on an input
# receptor-glycoligand
# complex.
# ====================== #

CONDA_PATH="${CONDA_PATH:-}"
GLYCOGRAPHER_PATH="${GLYCOGRAPHER_PATH:-}"

module purge

## Resolve `GLYCOGRAPHER_PATH` and `CONDA_PATH`, expand ~, and set defaults
if [ -z "$GLYCOGRAPHER_PATH" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    GLYCOGRAPHER_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    case "$GLYCOGRAPHER_PATH" in
        ~*) GLYCOGRAPHER_PATH="${HOME}${GLYCOGRAPHER_PATH#\~}" ;;
    esac
fi

# Default options file (inside the repo) if not supplied
if [ -z "$options" ]; then
    options="$GLYCOGRAPHER_PATH/config/glycandock_defaults.init"
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
    echo "No arguments provided. See: map_ensembles.py --help"
    exit 1
fi

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/run_glycandock.py "$@"

exit 0