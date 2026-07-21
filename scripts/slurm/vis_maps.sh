#!/bin/bash

#SBATCH --job-name=gdock-vis
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=./gdock_vis.out
#SBATCH --error=./gdock_vis.err

# ============================================================ #
# Build a PyMOL visualization (.pse, optional PNG) for a set of
# glycan binding maps, driven by the map manifest (JSON or YAML)
# written by map_ensembles.py.
#
# Thin pass-through: every argument after the script name is
# forwarded verbatim to vis_maps.py, so the full Python CLI is
# available without maintaining a parallel copy of each flag.
# See `vis_maps.py --help`. The manifest format (JSON or YAML) is
# handled by vis_maps.py itself, so no change is needed here.
#
# Note: headless ray-tracing (--render) needs OSMesa/libGL on the
# compute node. Without it, the .pse is still saved for later
# interactive rendering; omit --render if the node lacks GL.
#
# Examples:
#   sbatch vis_maps.sh maps/map_manifest.json --receptor 1o7v_receptor.pdb \
#       --tag boltzmann_b0p5 --contour-mode components --hotspots -o 1o7v_vis
#
#   sbatch vis_maps.sh maps/map_manifest.yaml --receptor 1o7v_receptor.pdb
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
    echo "No arguments provided. See: vis_maps.py --help"
    exit 1
fi

$CONDA_PATH/envs/glycographer/bin/python \
    $GLYCOGRAPHER_PATH/scripts/vis/vis_maps.py "$@"

exit 0
