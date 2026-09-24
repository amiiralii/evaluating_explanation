#!/usr/bin/env python3
"""
c7_local_stability.py: README experiment 3, "Local / C7".

Explain a random test point, then explain a point just like it, and ask whether
the two explanations name the same features. Only pairs whose predictions stay
close count: when the model's answer changes a lot, a different explanation is
correct, not unstable.

The neighbour is a REAL row, never a made-up one: the nearest row in the data
with different x values whose prediction is within Eps win units of the point's
under BOTH models, searched among the Near nearest rows with different x (exact
duplicates of the point are skipped, not counted). Real rows cannot be
impossible configurations, and x values are free, so any row may serve.

An explanation is reduced to the set of its Top features:

    ezr   the first Top features on the tree path, root first (tools/ezr summarize)
    lime  the Top largest |lime weights| of LightGBM
    shap  the Top largest |shap values| of LightGBM
    rand  control: Top features at random, drawn afresh for every explanation

Stability is the Jaccard overlap of the point's set and the neighbour's set,
from 0 (nothing shared) to 1 (identical). rand gives the overlap two unrelated
explanations of that size reach by chance. The report also carries three
diagnostic columns: found%, how often a close enough neighbour existed;
leaf%, how often the neighbour fell in the SAME ezr leaf, where ezr is stable
by construction; lime-self, the overlap of two lime runs on the SAME point,
which caps how stable lime can be. The x-* columns repeat the four overlaps on
only the pairs that fall in DIFFERENT ezr leaves, pooled over every repeat
because such pairs are rare. That is where ezr's stability is actually tested.

Options:

    -B Budget=50      labels for ezr's active learner
    -c csv=0          print one csv row per dataset: no header, no tally
    -E Eps=10         most a neighbour's prediction may move, in win units
    -L Lime=1000      lime samples per explanation
    -N Near=20        nearest rows to try when looking for a neighbour
    -P Points=20      random points per repeat
    -R Repeats=20     number of train/test splits
    -T Top=3          features kept from each explanation
    -f file=data/optimize/     one csv, or a folder of csvs
"""
from types import SimpleNamespace as o
from typing import Any
from statistics import mean, median
from pathlib import Path
import re, sys, random, warnings

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import (Data, Sym, csv, clone, adds, mid, disty, likely, Tree,
                       treeLeaf, trace, summarize, coerce, main, the)
from tools.stats import top

warnings.filterwarnings("ignore")
import numpy as np, lightgbm as lgb, shap
from lime.lime_tabular import LimeTabularExplainer

the.__dict__.update({k: coerce(v) for k, v in re.findall(r"(\w+)=(\S+)", __doc__)})
XAI = ["ezr", "lime", "shap", "rand"]

# ## Encoding ---------------------------------------------------------
def codes(data) -> dict:
  "Integer codes for the values of each symbolic x column."
  return {c.at: {v:i for i,v in enumerate(sorted(c.has, key=str))}
          for c in data.cols.x if c.it is Sym}

def num(code, col, v) -> float:
  "One x value as a float (symbols become codes, '?' becomes the column's mid)."
  v = mid(col) if v == "?" else v
  return code[col.at][v] if col.it is Sym else float(v)

def encode(data, code, rows) -> Any:
  "Rows as a float matrix over the x columns."
  return np.array([[num(code,c,r[c.at]) for c in data.cols.x] for r in rows])

def syms(data) -> list[int]:
  "Positions of the symbolic x columns."
  return [j for j,c in enumerate(data.cols.x) if c.it is Sym]

# ## Neighbours -------------------------------------------------------
def Near(data, X) -> callable:
  "Distance from one encoded row to every row: numbers scaled 0..1, symbols 0 or 1."
  sym = syms(data)
  lo  = np.array([0 if c.it is Sym else c.lo for c in data.cols.x], dtype=float)
  hi  = np.array([1 if c.it is Sym else c.hi for c in data.cols.x], dtype=float)
  Z   = (X - lo) / (hi - lo + 1e-32)
  def dists(j):
    d = np.abs(Z - Z[j])
    d[:, sym] = d[:, sym] > 0
    return np.sqrt((d**2).mean(axis=1))
  return dists

def nearest(d) -> list[int]:
  "The Near closest rows with different x (duplicates of the point do not count)."
  return [i for i in np.argsort(d, kind="stable") if d[i] > 0][:the.Near]

def neighbour(data, near, j, close) -> int:
  "Nearest row with different x whose predictions stay close. None if no such row."
  for i in nearest(near(j)):
    if close(data.rows[i]): return i

# ## Models -----------------------------------------------------------
def Ezr(data, labels) -> o:
  "Regression tree over the labelled rows; leaves hold disty on the full data's scale."
  tree = Tree(clone(data, labels), Y=lambda r: disty(data, r))
  return o(tree=tree, predict=lambda row: treeLeaf(tree,row).mu)

def Lgbm(data, code, labels) -> o:
  "LightGBM regressor on those same labels; target is disty."
  m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=8,
                        min_child_samples=the.leaf, random_state=the.seed,
                        verbose=-1, n_jobs=1)
  m.fit(encode(data, code, labels), [disty(data,r) for r in labels],
        categorical_feature=syms(data) or "auto")
  return o(m=m, predictX=m.predict)

# ## Explanations as feature sets ---------------------------------------
# Each xFun returns the names of an explanation's Top features.
def xEzr(data, ezr, row, x, aux) -> set:
  "The first Top features along the row's tree path, root first."
  return set(list(summarize(data, trace(data, ezr.tree, row)))[:the.Top])

def xLime(data, ezr, row, x, aux) -> set:
  "The Top features with the largest |lime weight|."
  exp = aux.lime.explain_instance(x, aux.lgbm.predictX, num_features=len(x),
                                  num_samples=the.Lime)
  js  = sorted(exp.local_exp[1], key=lambda jw: -abs(jw[1]))[:the.Top]
  return {data.cols.x[j].txt for j,_ in js}

def xShap(data, ezr, row, x, aux) -> set:
  "The Top features with the largest |shap value|."
  phi = aux.shap.shap_values(x[None,:])[0]
  return {data.cols.x[j].txt for j in np.argsort(-np.abs(phi), kind="stable")[:the.Top]}

def xRand(data, *_) -> set:
  "Control: Top features at random."
  return {c.txt for c in random.sample(data.cols.x, min(the.Top, len(data.cols.x)))}

def jaccard(a, b) -> float:
  "Shared over all; two empty explanations are identical."
  return len(a & b) / len(a | b) if a | b else 1.0

# ## Experiment -------------------------------------------------------
def run(file) -> tuple:
  "One dataset: per method the mean overlap of each repeat, plus the diagnostics."
  data = Data(csv(file))
  code = codes(data)
  X    = encode(data, code, data.rows)
  near = Near(data, X)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo) / (b4.mu - b4.lo + 1e-32))
  xFun = dict(ezr=xEzr, lime=xLime, shap=xShap, rand=xRand)
  out  = {k:[] for k in XAI}
  tried, found, selfs = 0, 0, []
  cross = {k:[] for k in XAI}                  # pairs in different ezr leaves
  for seed in range(the.Repeats):
    the.seed = seed; random.seed(seed)
    order  = random.sample(range(len(data.rows)), len(data.rows))
    half   = len(order)//2
    test   = order[:half]
    train  = clone(data, [data.rows[j] for j in order[half:]])
    labels = likely(train)                     # ezr buys Budget labels
    ezr    = Ezr(data, labels)
    lgbm   = Lgbm(data, code, labels)
    bg     = encode(data, code, random.sample(train.rows, min(train.n,512)))
    aux    = o(lgbm=lgbm, shap=shap.TreeExplainer(lgbm.m),
               lime=LimeTabularExplainer(bg, mode="regression", random_state=seed,
                  discretize_continuous=False, categorical_features=syms(data) or None))
    both   = lambda r: (win(ezr.predict(r)), win(lgbm.predictX(encode(data,code,[r]))[0]))
    now    = {k:[] for k in XAI}
    for j in random.sample(test, min(the.Points, len(test))):
      a, was = data.rows[j], both(data.rows[j])
      i = neighbour(data, near, j,
                    lambda r: all(abs(p - q) <= the.Eps for p,q in zip(both(r), was)))
      tried += 1
      if i is None: continue
      b = data.rows[i]
      found += 1
      same   = treeLeaf(ezr.tree, a) is treeLeaf(ezr.tree, b)
      for k in XAI:
        now[k] += [jaccard(xFun[k](data, ezr, a, X[j], aux), xFun[k](data, ezr, b, X[i], aux))]
        if not same: cross[k] += [now[k][-1]]
      selfs += [jaccard(xLime(data, ezr, a, X[j], aux), xLime(data, ezr, a, X[j], aux))]
    for k in XAI:
      if now[k]: out[k] += [mean(now[k])]      # a repeat with no close pair says nothing
  nan  = float("nan")
  diag = o(found=100*found/(tried or 1), self=mean(selfs) if selfs else nan,
           leaf=100*(1 - len(cross["ezr"])/(found or 1)),
           cross={k: mean(v) if v else nan for k,v in cross.items()})
  return out, diag

def report(file, out, diag) -> set:
  "One line per dataset: median overlap, with '+' marking the most stable methods."
  best = top(out, reverse=True) if all(out.values()) else set()
  name = Path(file).stem
  med  = lambda k: median(out[k]) if out[k] else float("nan")
  if the.csv:                            # machine readable: one row, nothing else
    print(",".join([name] + [f"{med(k):.2f}" for k in XAI]
                   + [" ".join(k for k in XAI if k in best),
                      f"{diag.found:.0f}", f"{diag.leaf:.0f}", f"{diag.self:.2f}"]
                   + [f"{diag.cross[k]:.2f}" for k in XAI]), flush=True)
    return best
  print(f"{name[:28]:30}", end="")
  for k in XAI: print(f"{med(k):8.2f}{'+' if k in best else ' '}", end="")
  print(f"{diag.found:8.0f}{diag.leaf:7.0f}{diag.self:10.2f}"
        + "".join(f"{diag.cross[k]:8.2f}" for k in XAI), flush=True)
  return best

def csvs(path) -> list[str]:
  "One csv, or every csv under a folder. A relative path also tries the repo root."
  p = Path(path)
  if not p.exists(): p = HERE.parent / path
  return [str(p)] if str(p).endswith(".csv") else sorted(
          str(q) for q in p.rglob("*.csv"))

def c7main():
  "top-level call"
  main(the, globals())
  if not the.csv:
    print(f"{'Data':30}" + "".join(f"{k:>9}" for k in XAI)
          + f"{'found%':>8}{'leaf%':>7}{'lime-self':>10}"
          + "".join(f"{'x-'+k:>8}" for k in XAI))
  wins = {k:0 for k in XAI}
  for f in csvs(the.file):
    for k in report(f, *run(f)): wins[k] += 1
  if not the.csv:
    print(f"\n{'#best of ' + str(len(csvs(the.file))):30}"
          + "".join(f"{wins[k]:9}" for k in XAI))

if __name__ == "__main__": c7main()
