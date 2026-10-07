#!/bin/sh
# One dataset by default. Pass experiment options after the script name.
cd "$(dirname "$0")/.." || exit 1
export PYTHONHASHSEED=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/experiment3-matplotlib}
exec "${PYTHON:-python3}" experiment3/experiment3_perturb.py "$@"
