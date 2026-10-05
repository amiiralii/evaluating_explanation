#!/bin/sh
# One dataset by default. Pass experiment options after the script name.
cd "$(dirname "$0")/.." || exit 1
export PYTHONHASHSEED=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/c7-matplotlib}
exec "${PYTHON:-python3}" experiment3/c7_perturb.py "$@"
