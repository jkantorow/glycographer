#!/bin/bash

#SBATCH --job-name=gdock-process
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./gdock_process.out
#SBATCH --error=./gdock_process.err

# ====================== #
# Process a GlycanDock
# output pose directory:
# load poses into a
# GlycanDockEnsemble and
# extract/cluster/dump the
# relevant analysis data.
# ====================== #

CONDA_PATH="${CONDA_PATH:-}"
GLYCOGRAPHER_PATH="${GLYCOGRAPHER_PATH:-}"

posedir=""
outprefix=""
outdir=""
rangestart=""
rangestop=""
clustercutoff=""
minclustersize=""
nocluster=false
nowritescoredata=false
noparseenergies=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -in|--posedir)
            if [[ $1 == *=* ]]; then posedir="${1#*=}"; shift; else posedir="$2"; shift 2; fi
            ;;
        -o|--outprefix)
            if [[ $1 == *=* ]]; then outprefix="${1#*=}"; shift; else outprefix="$2"; shift 2; fi
            ;;
        --outdir)
            if [[ $1 == *=* ]]; then outdir="${1#*=}"; shift; else outdir="$2"; shift 2; fi
            ;;
        -range|--poserange)
            # Takes two values: START STOP
            rangestart="$2"; rangestop="$3"; shift 3
            ;;
        --cluster-cutoff)
            if [[ $1 == *=* ]]; then clustercutoff="${1#*=}"; shift; else clustercutoff="$2"; shift 2; fi
            ;;
        --min-cluster-size)
            if [[ $1 == *=* ]]; then minclustersize="${1#*=}"; shift; else minclustersize="$2"; shift 2; fi
            ;;
        --no-cluster)
            nocluster=true; shift
            ;;
        --no-write-scoredata)
            nowritescoredata=true; shift
            ;;
        --no-parse-energies)
            noparseenergies=true; shift
            ;;
        *)
            echo "Unknown parameter passed: $1"; exit 1
            ;;
    esac
done

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

# Build python argument array to avoid empty flags/arguments being passed
python_args=()
if [[ -n "$posedir" ]]; then
    python_args+=("$posedir")
else
    echo "No pose directory provided (-in / --posedir)."; exit 1
fi

if [[ -n "$outprefix" ]]; then python_args+=("--outprefix" "$outprefix"); fi
if [[ -n "$outdir" ]]; then python_args+=("--outdir" "$outdir"); fi
if [[ -n "$rangestart" && -n "$rangestop" ]]; then python_args+=("--poserange" "$rangestart" "$rangestop"); fi
if [[ -n "$clustercutoff" ]]; then python_args+=("--cluster-cutoff" "$clustercutoff"); fi
if [[ -n "$minclustersize" ]]; then python_args+=("--min-cluster-size" "$minclustersize"); fi
if [[ $nocluster = true ]]; then python_args+=("--no-cluster"); fi
if [[ $nowritescoredata = true ]]; then python_args+=("--no-write-scoredata"); fi
if [[ $noparseenergies = true ]]; then python_args+=("--no-parse-energies"); fi

$CONDA_PATH/envs/glycographer/bin/python $GLYCOGRAPHER_PATH/scripts/utils/process_glycandock_output.py "${python_args[@]}"

exit 0
