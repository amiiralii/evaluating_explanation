#!/usr/bin/env python3
"""
c1_tally.py: which method is top on how many datasets, from the C1 sweep.

Reads c1/results/c1_results_full.csv (written by c1/sweep.sh) and prints a csv
to stdout: one row per group and learner, with how often each method is among
the statistically best ("_best", ties included) and how often it is the ONLY
best ("_sole"). Groups are the subsets below (kind=subset) and every dataset
family (kind=family). Save it with:

    python3 c1/c1_tally.py > c1/results/c1_tally.csv

Subsets:
    all       every dataset
    K<n       K is smaller than the number of x columns, so methods can differ
    sep       K<n and random is NOT among the best on that learner, so the
              score actually told the methods apart

Remember that ezr on the ezr learner is 0 by construction (see RESULTS.md),
so ezr_best on learner=ezr is not skill.

Options:

    -f file=c1/results/c1_results_full.csv    the sweep's full report
    -d data=data/optimize                     where the dataset families live
"""
from collections import Counter
from pathlib import Path
import re, sys, csv as csvlib

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import coerce, main, the
from c1_global_topk import XAI

the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})
LRN = dict(e="ezr", l="lgbm")

def path(p) -> Path:
  "A relative path also tries the repo root."
  return Path(p) if Path(p).exists() else HERE.parent / p

def families() -> dict:
  "Dataset name to the folder it lives in."
  return {q.stem: q.parent.name for q in path(the.data).rglob("*.csv")}

def tally(rows, L) -> tuple:
  "Per method: datasets where it is among the best, and where it is the only best."
  best, sole = Counter(), Counter()
  for r in rows:
    won = r[f"{L}_best"].split()
    best.update(won)
    if len(won) == 1: sole.update(won)
  return best, sole

def show(kind, name, rows, L):
  "One csv row: a group, one learner, best then sole counts for every method."
  best, sole = tally(rows, L)
  print(",".join([kind, name, LRN[L], str(len(rows))]
                 + [str(best[x]) for x in XAI] + [str(sole[x]) for x in XAI]))

def c1tally():
  "top-level call"
  main(the, globals())
  rows = list(csvlib.DictReader(open(path(the.file))))
  fam  = families()
  kin  = [r for r in rows if r["k"] != r["n_x"]]
  print(",".join(["kind", "group", "learner", "n"]
                 + [f"{x}_best" for x in XAI] + [f"{x}_sole" for x in XAI]))
  for L in LRN:
    show("subset", "all", rows, L)
    show("subset", "K<n", kin, L)
    show("subset", "sep", [r for r in kin if "rand" not in r[f"{L}_best"].split()], L)
  for f in sorted(set(fam.get(r["dataset"], "?") for r in rows)):
    for L in LRN: show("family", f, [r for r in rows if fam.get(r["dataset"], "?") == f], L)

if __name__ == "__main__": c1tally()
