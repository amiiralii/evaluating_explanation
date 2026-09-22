#!/usr/bin/env python3
"""
tools/oracle_check.py: can an oracle score a row it has never seen?

Hide Hold random rows. Fit each oracle on the rows that are left, ask it for the
disty of each hidden row, then compare with the truth. Everything is in "win"
units, where 100 is the best row in the file and 0 is the average row.

Three oracles are compared:

    forest   random forest, the judge used by the C2 experiment
    knn      weighted average of the Near closest kept rows, 1/distance weights
    mean     predict the mean of the kept rows, every time. the do-nothing control

Reported per dataset: mae is the mean absolute error, r is the correlation with
the truth, and skill is 1 - mae/mae(mean), so 0 means no better than the control
and 1 means perfect. spread is the sd of the true win values, the scale that
makes an mae readable.

This is an upper bound on how well an oracle scores a MADE-UP row, because a
hidden row is still a real row: a real combination of feature values. A made-up
row can be a combination that never occurs, where any oracle extrapolates.

Options:

    -H Hold=100      rows to hide (capped at a third of the file)
    -N Near=5        neighbours for knn
    -s seed=0        which shuffle to hide
    -f file=data/optimize/     one csv, or a folder of csvs
"""
from statistics import median
from pathlib import Path
import re, sys, random, warnings

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]              # the repo root, for tools.ezr
from tools.ezr import Data, Sym, csv, adds, disty, mid, coerce, main, the

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestRegressor

the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})
ORACLES = ["forest", "knn", "mean"]

# encoding, kept here so this tool needs no experiment to run
def codes(data) -> dict:
  "Integer codes for the values of each symbolic x column."
  return {c.at: {v:i for i,v in enumerate(sorted(c.has, key=str))}
          for c in data.cols.x if c.it is Sym}

def encode(data, code, rows):
  "Rows as a float matrix over the x columns ('?' becomes the column's mid)."
  val = lambda c,v: code[c.at][v] if c.it is Sym else float(v)
  return np.array([[val(c, r[c.at] if r[c.at] != "?" else mid(c))
                    for c in data.cols.x] for r in rows])

def csvs(path) -> list[str]:
  "One csv, or every csv under a folder. A relative path also tries the repo root."
  p = Path(path)
  if not p.exists(): p = HERE.parent / path
  return [str(p)] if str(p).endswith(".csv") else sorted(
          str(q) for q in p.rglob("*.csv"))

def knn1(Z, ys, sym, q) -> float:
  "Weighted mean of the Near closest kept rows."
  d = np.abs(Z - q)
  d[:,sym] = Z[:,sym] != q[sym]
  d = np.mean(d**the.p, axis=1) ** (1/the.p)
  k = min(the.Near, len(d))
  j = np.argpartition(d, k-1)[:k]
  w = 1/(d[j] + 1e-32)                        # an identical kept row wins outright
  return float((w * ys[j]).sum() / w.sum())

def check(file) -> dict:
  "Hide some rows, score them with each oracle, return each oracle's error."
  data = Data(csv(file))
  code = codes(data)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo)/(b4.mu - b4.lo + 1e-32))
  X    = encode(data, code, data.rows)
  ys   = np.array([win(disty(data,r)) for r in data.rows])
  sym  = np.array([c.it is Sym for c in data.cols.x])
  n    = len(X)
  hold = min(the.Hold, n//3)
  random.seed(the.seed)
  idx  = random.sample(range(n), n)
  te, tr = np.array(idx[:hold]), np.array(idx[hold:])

  rf = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=0).fit(X[tr], ys[tr])
  Z  = X.copy()
  lo, hi = X[tr].min(0), X[tr].max(0)
  Z[:,~sym] = (X[:,~sym] - lo[~sym]) / (hi[~sym] - lo[~sym] + 1e-32)
  got = dict(forest = rf.predict(X[te]),
             knn    = np.array([knn1(Z[tr], ys[tr], sym, Z[i]) for i in te]),
             mean   = np.full(hold, ys[tr].mean()))
  true = ys[te]
  base = np.mean(np.abs(got["mean"] - true))
  out  = {}
  for k, p in got.items():
    mae = float(np.mean(np.abs(p - true)))
    r   = float(np.corrcoef(p, true)[0,1]) if np.std(p) > 1e-9 else 0.0
    out[k] = (mae, r, 1 - mae/(base + 1e-32))
  return o_row(file, n, hold, float(np.std(true)), out)

def o_row(file, n, hold, spread, out) -> dict:
  "One dataset's answer, and the line that shows it."
  print(f"{Path(file).stem[:22]:24}{n:7}{hold:6}{spread:8.1f}", end="")
  for k in ORACLES:
    mae, r, skill = out[k]
    print(f"{mae:8.1f}{r:6.2f}{skill:7.2f}", end="")
  print(flush=True)
  return dict(file=Path(file).stem, n=n, spread=spread, **out)

def summary(rows):
  "Who wins, and by how much."
  print("\n" + "="*96)
  print(f"{len(rows)} datasets")
  for k in ORACLES:
    mae = [r[k][0] for r in rows]; sk = [r[k][2] for r in rows]
    print(f"   {k:7} median mae {median(mae):6.1f}   median skill {median(sk):6.2f}"
          f"   skill > 0 on {sum(s>0 for s in sk):4} datasets")
  f_better = sum(r["forest"][0] < r["knn"][0] for r in rows)
  print(f"\n   forest beats knn on {f_better} of {len(rows)} datasets"
        f" (knn better on {len(rows)-f_better})")
  worse = [r for r in rows if r["forest"][2] <= 0]
  print(f"   forest no better than the mean control on {len(worse)}:",
        ", ".join(r["file"][:18] for r in worse[:8]) or "none")

def c2check():
  "top-level call"
  main(the, globals())
  print(f"{'Data':24}{'rows':>7}{'held':>6}{'spread':>8}"
        + "".join(f"{k:>8}{'r':>6}{'skill':>7}" for k in ORACLES))
  print(f"{'':45}" + "".join(f"{'mae':>8}{'':6}{'':7}" for k in ORACLES))
  rows = [check(f) for f in csvs(the.file)]
  summary(rows)

if __name__ == "__main__": c2check()
