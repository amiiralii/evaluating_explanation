#!/bin/sh
# Run the C7 experiment on every dataset and rewrite the report.
#
# One process per dataset. Each dataset's row, including which methods tie for
# best, is computed independently of the others, so sharding gives exactly what
# one long run would give. Single-process takes hours; sharding
# divides that by the number of jobs, minus the widest file.
#
# Writes two files:
#   c7/results/c7_results.csv       the report: dataset, one column per method, best
#   c7/results/c7_results_full.csv  the same plus found_pct, leaf_pct, lime_self, x_*
# Errors and warnings go to c7/results/sweep.err
#
# PYTHONHASHSEED is pinned because ezr's _symCuts iterates a set of strings when
# it picks a symbolic cut (tools/ezr.py:330). Python randomizes string hashing per
# process, so ties between equally good cuts break differently every run and the
# tree changes. Without this the ezr column moves by a few win units on datasets
# with symbolic columns.
#
# Usage:  sh c7/sweep.sh [jobs] [data-dir]        defaults: 10  data/optimize

cd "$(dirname "$0")/.." || exit 1                  # always run from the repo root
export PYTHONHASHSEED=0                            # see the note above
JOBS=${1:-10}
DATA=${2:-data/optimize}
OUT=c7/results
FULL=$OUT/c7_results_full.csv
REPORT=$OUT/c7_results.csv

mkdir -p "$OUT"
N=$(find "$DATA" -name '*.csv' | wc -l | tr -d ' ')
echo "running $N datasets, $JOBS at a time. watch progress with: wc -l $FULL"
: > "$OUT/sweep.err"

echo "dataset,ezr,lime,shap,rand,best,found_pct,leaf_pct,lime_self,x_ezr,x_lime,x_shap,x_rand" > "$FULL"
find "$DATA" -name '*.csv' | sort \
  | xargs -P "$JOBS" -n 1 sh -c \
      'python3 c7/c7_local_stability.py -c 1 -f "$1" 2>>'"$OUT"'/sweep.err' _ \
  | sort >> "$FULL"

cut -d, -f1-6 "$FULL" > "$REPORT"
GOT=$(( $(wc -l < "$REPORT") - 1 ))
echo "$GOT of $N datasets -> $REPORT"
[ "$GOT" -eq "$N" ] || echo "WARNING: $(( N - GOT )) missing, see $OUT/sweep.err"

awk -F, 'NR>1 { n=split($6,w," "); for(i=1;i<=n;i++) c[w[i]]++ }
         END { printf "best on:"; for(k in c) printf "  %s %d", k, c[k]; print "" }' "$FULL"
