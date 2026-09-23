#!/bin/sh
# Run the C1 experiment on every dataset and rewrite the report.
#
# One process per dataset. Each dataset's row, including which methods tie for
# best, is computed independently of the others, so sharding gives exactly what
# one long run would give.
#
# Writes two files:
#   c1/results/c1_results.csv       the report: dataset, then per learner (e_ ezr
#                                   tree, l_ LightGBM) one column per method and best
#   c1/results/c1_results_full.csv  the same plus k, n_x and each full model's score
# Errors and warnings go to c1/results/sweep.err
#
# PYTHONHASHSEED is pinned because ezr's _symCuts iterates a set of strings when
# it picks a symbolic cut (tools/ezr.py:330). Python randomizes string hashing per
# process, so ties between equally good cuts break differently every run and the
# tree changes. Without this the ezr column moves by a few win units on datasets
# with symbolic columns.
#
# Usage:  sh c1/sweep.sh [jobs] [data-dir]        defaults: 10  data/optimize

cd "$(dirname "$0")/.." || exit 1                  # always run from the repo root
export PYTHONHASHSEED=0                            # see the note above
JOBS=${1:-10}
DATA=${2:-data/optimize}
OUT=c1/results
FULL=$OUT/c1_results_full.csv
REPORT=$OUT/c1_results.csv

mkdir -p "$OUT"
N=$(find "$DATA" -name '*.csv' | wc -l | tr -d ' ')
echo "running $N datasets, $JOBS at a time. watch progress with: wc -l $FULL"
: > "$OUT/sweep.err"

echo "dataset,e_ezr,e_lime,e_shap,e_rand,e_best,l_ezr,l_lime,l_shap,l_rand,l_best,k,n_x,e_full,l_full" > "$FULL"
find "$DATA" -name '*.csv' | sort \
  | xargs -P "$JOBS" -n 1 sh -c \
      'python3 c1/c1_global_topk.py -c 1 -f "$1" 2>>'"$OUT"'/sweep.err' _ \
  | sort >> "$FULL"

cut -d, -f1-11 "$FULL" > "$REPORT"
GOT=$(( $(wc -l < "$REPORT") - 1 ))
echo "$GOT of $N datasets -> $REPORT"
[ "$GOT" -eq "$N" ] || echo "WARNING: $(( N - GOT )) missing, see $OUT/sweep.err"

for F in 6:ezr 11:lgbm; do
  awk -F, -v f="${F%%:*}" -v l="${F#*:}" 'NR>1 { n=split($f,w," "); for(i=1;i<=n;i++) c[w[i]]++ }
       END { printf "best with %s learner:", l; for(k in c) printf "  %s %d", k, c[k]; print "" }' "$FULL"
done
