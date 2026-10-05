#!/bin/sh
# Full sweep, only when explicitly invoked: sh experiment3/sweep.sh [jobs] [data-dir]
cd "$(dirname "$0")/.." || exit 1
export PYTHONHASHSEED=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/c7-matplotlib}
PYTHON=${PYTHON:-python3}
export PYTHON
JOBS=${1:-10}
DATA=${2:-data/optimize}
OUT=experiment3/results
FULL=$OUT/c7_results_full.csv
REPORT=$OUT/c7_results.csv
mkdir -p "$OUT" || exit 1
LIST=$(mktemp) || exit 1
trap 'rm -f "$LIST"' EXIT HUP INT TERM
find "$DATA" -name '*.csv' | sort > "$LIST"
N=$(wc -l < "$LIST" | tr -d ' ')
[ "$N" -gt 0 ] || { echo "No datasets at $DATA" >&2; exit 1; }
echo "running $N datasets, $JOBS at a time"
: > "$OUT/sweep.err"
"$PYTHON" experiment3/c7_perturb.py --header > "$FULL" || exit 1
cat "$LIST" | xargs -P "$JOBS" -n 1 sh -c \
  '"$PYTHON" experiment3/c7_perturb.py -c 1 -f "$1" 2>>experiment3/results/sweep.err' _ \
  | sort >> "$FULL"
cut -d, -f1-6 "$FULL" > "$REPORT"
GOT=$(( $(wc -l < "$REPORT") - 1 ))
echo "$GOT of $N datasets -> $REPORT"
[ "$GOT" -eq "$N" ] || { echo "WARNING: missing datasets; see $OUT/sweep.err" >&2; exit 1; }
