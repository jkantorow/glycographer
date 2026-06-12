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

CONDA_PATH="/projects/SimBioSys/jkant/miniconda3-jkant/miniconda3-jkant"
GLYCOGRAPHER_PATH="~/glycographer"

nstruct=1
mccycles=1
outprefix=""
ctype=false
norandomstart=false
options=$GLYCOGRAPHER_PATH/config/glycandock_defaults.init
startcountfrom=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -in|--input-complex)
            if [[ $1 == *=* ]]; then complex="${1#*=}"; shift; else complex="$2"; shift 2; fi
            ;;
        -n|--nstruct)
            if [[ $1 == *=* ]]; then nstruct="${1#*=}"; shift; else nstruct="$2"; shift 2; fi
            ;;
        -grid|--meshgrid)
            if [[ $1 == *=* ]]; then meshgrid="${1#*=}"; shift; else meshgrid="$2"; shift 2; fi
            ;;
        -o|--outprefix)
            if [[ $1 == *=* ]]; then outprefix="${1#*=}"; shift; else outprefix="$2"; shift 2; fi
            ;;
        --start-count-from)
            if [[ $1 == *=* ]]; then startcountfrom="${1#*=}"; shift; else startcountfrom="$2"; shift 2; fi
            ;;
        --options)
            if [[ $1 == *=* ]]; then options="${1#*=}"; shift; else options="$2"; shift 2; fi
            ;;
        --mc-cycles)
            if [[ $1 == *=* ]]; then mccycles="${1#*=}"; shift; else mccycles="$2"; shift 2; fi
            ;;
        --no-random-start)
            norandomstart=true; shift
            ;;
        --c-type)
            ctype=true; shift
            ;;
        *)
            echo "Unknown parameter passed: $1"; exit 1
            ;;
    esac
done

if [ $ctype = true ]; then
    options=$GLYCOGRAPHER_PATH/config/glycandock_ctype.init
fi

# Source required python env:
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate glycographer

# Exit on any error
set -e

# Build python argument array to avoid empty flags/arguments being passed
python_args=()
if [[ -n "$complex" ]]; then
    python_args+=("$complex")
else
    echo "No input complex provided (-in / --input-complex)."; exit 1
fi

python_args+=("--nstruct" "$nstruct")
python_args+=("--mc-cycles" "$mccycles")
if [[ -n "$meshgrid" ]]; then python_args+=("--meshgrid" "$meshgrid"); fi
if [[ -n "$outprefix" ]]; then python_args+=("--outprefix" "$outprefix"); fi
if [[ -n "$options" ]]; then python_args+=("--options" "$options"); fi
if [[ -n "$startcountfrom" ]]; then python_args+=("--start-count-from" "$startcountfrom"); fi
if [[ $norandomstart = true]]; then python_args+=("--no-random-start"); fi

python $GLYCOGRAPHER_PATH/scripts/run_glycandock.py "${python_args[@]}"

exit 0
