#!/usr/bin/env python3
"""
c7_tally.py: which method is most stable on how many datasets, from the C7 sweep.

Reads c7/results/c7_results_full.csv (written by c7/sweep.sh) and prints a csv
to stdout: one row per group, with how often each method is among the
statistically most stable ("_best", ties included) and how often it is the
ONLY most stable ("_sole"). Groups are the subsets below (kind=subset) and
every dataset family (kind=family). Save it with:

    python3 c7/c7_tally.py > c7/results/c7_tally.csv

Subsets:
    all       every dataset
    x>Top     more x columns than Top, so the Top sets are not the whole
              feature set and random is not trivially identical

Remember that ezr is identical by construction whenever the neighbour falls in
the same leaf, which is most of the time (see RESULTS.md).

Options:

    -f file=c7/results/c7_results_full.csv    the sweep's full report
    -d data=data/optimize                     where the datasets live
    -T Top=3                                  features kept per explanation
"""
from collections import Counter
from pathlib import Path
import re, sys, csv as csvlib

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import Cols, coerce, main, the
from c7_local_stability import XAI

the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})

def path(p) -> Path:
  "A relative path also tries the repo root."
  return Path(p) if Path(p).exists() else HERE.parent / p

def datasets() -> dict:
  "Dataset name to (its family folder, its number of x columns)."
  out = {}
  for q in path(the.data).rglob("*.csv"):
    with open(q, encoding="utf-8") as f: names = [s.strip() for s in f.readline().split(",")]
    out[q.stem] = (q.parent.name, len(Cols(names).x))
  return out

def show(kind, name, rows):
  "One csv row: a group, then best and sole counts for every method."
  best, sole = Counter(), Counter()
  for r in rows:
    won = r["best"].split()
    best.update(won)
    if len(won) == 1: sole.update(won)
  print(",".join([kind, name, str(len(rows))]
                 + [str(best[x]) for x in XAI] + [str(sole[x]) for x in XAI]))

def c7tally():
  "top-level call"
  main(the, globals())
  rows = list(csvlib.DictReader(open(path(the.file))))
  ds   = datasets()
  print(",".join(["kind", "group", "n"]
                 + [f"{x}_best" for x in XAI] + [f"{x}_sole" for x in XAI]))
  show("subset", "all", rows)
  show("subset", "x>Top", [r for r in rows if ds[r["dataset"]][1] > the.Top])
  for f in sorted(set(ds[r["dataset"]][0] for r in rows)):
    show("family", f, [r for r in rows if ds[r["dataset"]][0] == f])

if __name__ == "__main__": c7tally()
