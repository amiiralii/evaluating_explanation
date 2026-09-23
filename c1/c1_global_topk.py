#!/usr/bin/env python3
"""
c1_global_topk.py: README experiment "Global / C1, C4".

Explain a model globally, keep only the K features the explanation ranks
highest, retrain on just those, and ask whether the smaller model still finds
good rows. A method whose top K carry the signal loses little; a method whose
top K miss it loses a lot.

K is fixed per dataset BEFORE the experiment starts: grow Tries ezr trees, each
on a random half of the data with its own Budget labels, count the distinct
features each tree uses, and take the median. So K is how many features ezr
naturally needs here. Only the count is kept, not which features they were.

Each repeat then buys Budget labels with ezr's active learner and fits two full
models on them, an ezr tree and a LightGBM regressor. Each method ranks the
features globally:

    ezr   mean decrease in impurity over the full ezr tree's splits
    lime  mean |weight| of LightGBM's lime explanations at Global train rows
    shap  mean |shap value| of LightGBM over a sample of train rows
    rand  control: a random order

Every top K set retrains BOTH learners on the same labels, so each method is
judged on both its home model and the other one. Score: the model sorts the
test half, and we take the true win of the best of the first Check rows it
picks. Test rows are real rows, so the score is exact and needs no oracle.
Reported as the retrained score minus the full model's score on the same split:
0 means nothing was lost by dropping the other features.

Options:

    -B Budget=50      labels for ezr's active learner
    -C Check=5        test rows each model picks; score the best of them
    -c csv=0          print one csv row per dataset: no header, no tally
    -G Global=20      train rows lime explains, to average into a global rank
    -L Lime=1000      lime samples per explanation
    -R Repeats=20     number of train/test splits
    -T Tries=10       ezr trees grown to fix K
    -f file=data/optimize/     one csv, or a folder of csvs
"""
from types import SimpleNamespace as o
from typing import Any
from statistics import mean, median
from pathlib import Path
import re, sys, random, warnings

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import (Data, Sym, csv, clone, adds, mid, disty, likely,
                       Tree, treeLeaf, treeNodes, treeSelects, coerce, main, the)
from tools.stats import top

warnings.filterwarnings("ignore")
import numpy as np, lightgbm as lgb, shap
from lime.lime_tabular import LimeTabularExplainer

the.__dict__.update({k: coerce(v) for k, v in re.findall(r"(\w+)=(\S+)", __doc__)})
XAI = ["ezr", "lime", "shap", "rand"]
LRN = ["ezr", "lgbm"]

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

# ## Models -----------------------------------------------------------
def Ezr(data, labels, keep) -> o:
  "Regression tree over the labelled rows, splitting only on the columns in keep."
  d = clone(data, labels)
  d.cols.x = [c for c in d.cols.x if c.at in keep]
  tree = Tree(d)
  return o(data=d, tree=tree, rank=lambda rows: sorted(rows, key=lambda r: treeLeaf(tree,r).mu))

def Lgbm(data, code, labels, keep) -> o:
  "LightGBM regressor on those same labels and only the columns in keep; target is disty."
  js   = [j for j,c in enumerate(data.cols.x) if c.at in keep]
  cats = [i for i,j in enumerate(js) if data.cols.x[j].it is Sym]
  X    = lambda rows: encode(data, code, rows)[:, js]
  m    = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=8,
                           min_child_samples=the.leaf, random_state=the.seed,
                           verbose=-1, n_jobs=1)
  m.fit(X(labels), [disty(data,r) for r in labels], categorical_feature=cats or "auto")
  def rank(rows):
    p = m.predict(X(rows))
    return [rows[i] for i in np.argsort(p, kind="stable")]
  return o(m=m, rank=rank)

def used(tree) -> set:
  "Column positions the tree splits on."
  return {node.how[1] for _, node in treeNodes(tree) if node.how}

# ## Global explanations ----------------------------------------------
# Each xFun returns {column position: importance}; bigger is more important.
def xEzr(data, model, _) -> dict:
  "Mean decrease in impurity: per column, the drop in sum of squared errors its splits buy."
  d, out = model.data, {c.at:0 for c in data.cols.x}
  sse = lambda rows: adds(disty(d,r) for r in rows).m2
  for _, node in treeNodes(model.tree):
    if not node.kids: continue
    rest = [r for r in node.rows if not any(treeSelects(r,*k.how) for k in node.kids)]
    kids = [k.rows for k in node.kids] + [rest]   # rows no kid took stay at this node
    out[node.kids[0].how[1]] += max(0, sse(node.rows) - sum(sse(rs) for rs in kids))
  return out

def xLime(data, model, aux) -> dict:
  "Mean |lime weight| over a sample of train rows."
  out = {c.at:0 for c in data.cols.x}
  for x in aux.X[:the.Global]:
    exp = aux.lime.explain_instance(x, model.m.predict, num_features=len(x),
                                    num_samples=the.Lime)
    for j,w in exp.local_exp[1]: out[data.cols.x[j].at] += abs(w)
  return out

def xShap(data, model, aux) -> dict:
  "Mean |shap value| over a sample of train rows."
  phi = np.abs(shap.TreeExplainer(model.m).shap_values(aux.X)).mean(axis=0)
  return {c.at: phi[j] for j,c in enumerate(data.cols.x)}

def xRand(data, *_) -> dict:
  "Control: a random order."
  return {c.at: random.random() for c in data.cols.x}

def topk(data, imp, k) -> set:
  "The k most important columns; ties (including all the zeros) broken at random."
  return {c.at for c in sorted(data.cols.x, key=lambda c: (-imp[c.at], random.random()))[:k]}

# ## Experiment -------------------------------------------------------
def kOf(data) -> int:
  "Median number of distinct features in Tries ezr trees, each on a random half."
  ns = []
  for i in range(the.Tries):
    the.seed = 1000 + i; random.seed(the.seed)
    half = clone(data, random.sample(data.rows, len(data.rows)//2))
    ns  += [len(used(Tree(clone(data, likely(half)))))]
  return max(1, round(median(ns)))

def run(file) -> tuple:
  "One dataset: per learner and method, the retrained minus full score of each repeat."
  data = Data(csv(file))
  code = codes(data)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo) / (b4.mu - b4.lo + 1e-32))
  pick = lambda model, rows: max(win(disty(data,r)) for r in model.rank(rows)[:the.Check])
  k    = kOf(data)
  every = {c.at for c in data.cols.x}
  out  = {(l,x):[] for l in LRN for x in XAI}
  full = {l:[] for l in LRN}
  for seed in range(the.Repeats):
    the.seed = seed; random.seed(seed)
    rows   = random.sample(data.rows, len(data.rows))
    half   = len(rows)//2
    test   = rows[:half]
    train  = clone(data, rows[half:])
    labels = likely(train)                     # ezr buys Budget labels
    X      = encode(data, code, random.sample(train.rows, min(train.n,512)))
    aux    = o(X=X, lime=LimeTabularExplainer(X, mode="regression",
                  random_state=seed, discretize_continuous=False,
                  categorical_features=[j for j,c in enumerate(data.cols.x)
                                        if c.it is Sym] or None))
    learn  = dict(ezr=lambda keep: Ezr(data, labels, keep),
                  lgbm=lambda keep: Lgbm(data, code, labels, keep))
    was    = {l: learn[l](every) for l in LRN}
    base   = {l: pick(was[l], test) for l in LRN}
    hows   = dict(ezr=(was["ezr"],xEzr), lime=(was["lgbm"],xLime),
                  shap=(was["lgbm"],xShap), rand=(None,xRand))
    for x,(model,xFun) in hows.items():
      keep = topk(data, xFun(data, model, aux), k)
      for l in LRN: out[l,x] += [pick(learn[l](keep), test) - base[l]]
    for l in LRN: full[l] += [base[l]]
  return out, full, k, len(data.cols.x)

def report(file, out, full, k, n) -> dict:
  "One line per dataset: median loss per learner and method, '+' marking the best."
  best = {l: top({x: out[l,x] for x in XAI}, reverse=True) for l in LRN}
  name = Path(file).stem
  if the.csv:                            # machine readable: one row, nothing else
    cells = [name]
    for l in LRN:
      cells += [f"{median(out[l,x]):.1f}" for x in XAI]
      cells += [" ".join(x for x in XAI if x in best[l])]
    cells += [str(k), str(n)] + [f"{median(full[l]):.1f}" for l in LRN]
    print(",".join(cells), flush=True)
    return best
  print(f"{name[:28]:30}{k:>4}/{n:<5}", end="")
  for l in LRN:
    print(f"{median(full[l]):8.1f} |", end="")
    for x in XAI: print(f"{median(out[l,x]):7.1f}{'+' if x in best[l] else ' '}", end="")
  print(flush=True)
  return best

def csvs(path) -> list[str]:
  "One csv, or every csv under a folder. A relative path also tries the repo root."
  p = Path(path)
  if not p.exists(): p = HERE.parent / path
  return [str(p)] if str(p).endswith(".csv") else sorted(
          str(q) for q in p.rglob("*.csv"))

def c1main():
  "top-level call"
  main(the, globals())
  if not the.csv:
    head = lambda l: f"{l+':full':>8} |" + "".join(f"{x:>7} " for x in XAI)
    print(f"{'Data':30}{'K/n':>5}     " + "".join(head(l) for l in LRN))
  wins = {(l,x):0 for l in LRN for x in XAI}
  for f in csvs(the.file):
    for l,b in report(f, *run(f)).items():
      for x in b: wins[l,x] += 1
  if not the.csv:
    print(f"\n{'#best of ' + str(len(csvs(the.file))):40}"
          + "".join(f"{'':10}" + "".join(f"{wins[l,x]:7} " for x in XAI) for l in LRN))

if __name__ == "__main__": c1main()
