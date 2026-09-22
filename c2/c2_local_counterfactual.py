#!/usr/bin/env python3
"""
c2_local_counterfactual.py: README experiment "Local / C2".

Build a model, explain it, then pick some random points. From each point's
explanation, change ONE feature to make an artificial suggestion. Score that
suggestion against the TRUTH, not against the model that proposed it, in "win"
units where 100 is the best row in the data and 0 is the average row.

The truth of a made-up row: if that row is already in the data, use its real y.
Otherwise a random forest, fit on every row and every label, predicts it. That
forest reads all the labels, which is allowed because it judges, it never learns.

The report's last two columns keep the reader honest. exact% is how often the row
really was in the data, so no estimate was needed. orc-r is the forest's own
out-of-bag accuracy, the correlation between its held-out guesses and the real y.
A dataset with a low orc-r cannot be scored by anything, so ignore what the
methods score there. c2_oracle_check.py measures all of this properly.

Methods: ezr (path to the best leaf), lime, shap (both on LightGBM trained on
the same labels ezr bought), rand (control: any feature, any direction).

Options:

    -B Budget=50      labels for ezr's active learner
    -J Judge=truth    who scores a suggestion (truth|model)
    -L Lime=1000      lime samples per explanation
    -P Points=20      random points per repeat
    -R Repeats=20     number of train/test splits
    -S Step=1         lime/shap/rand step size, in sds
    -f file=data/optimize/     one csv, or a folder of csvs
"""
from types import SimpleNamespace as o
from typing import Any
from statistics import mean, median
from pathlib import Path
import re, sys, math, random, warnings

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import (Data, Num, Sym, csv, clone, adds, mid, disty, likely,
                       Tree, treeLeaf, treeSelects, coerce, main, the)
from tools.stats import top

warnings.filterwarnings("ignore")
import numpy as np, lightgbm as lgb, shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor

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

# ## Ground truth ------------------------------------------------------
def Truth(data, code) -> o:
  "How to score a made-up row: a table of the real ones, plus a forest for the rest."
  X    = encode(data, code, data.rows)
  ys   = np.array([disty(data,r) for r in data.rows])
  seen = {}
  for x,y in zip(X, ys): seen.setdefault(tuple(x), []).append(y)
  rf   = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=1,
                               random_state=0).fit(X, ys)
  ok   = ~np.isnan(rf.oob_prediction_)           # a row can miss every out-of-bag
  r    = float(np.corrcoef(rf.oob_prediction_[ok], ys[ok])[0,1]) if ok.sum() > 2 else 0
  return o(seen=seen, rf=rf, r=r, n=0, exact=0)

def real(t, data, code, row) -> float:
  "A made-up row's y: the real one if that row exists, else the forest's guess."
  q = encode(data, code, [row])[0]
  t.n += 1
  if (hit := t.seen.get(tuple(q))) is not None:  # this row is really in the data
    t.exact += 1; return float(np.mean(hit))
  return float(t.rf.predict(q[None,:])[0])

# ## Models -----------------------------------------------------------
def Ezr(data, labels) -> o:
  "Regression tree over the rows ezr chose to label."
  tree = Tree(clone(data, labels))
  return o(tree=tree, predict=lambda row: treeLeaf(tree,row).mu)

def Lgbm(data, code, labels) -> o:
  "LightGBM regressor on those same labels; target is disty."
  m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=8,
                        min_child_samples=the.leaf, random_state=the.seed,
                        verbose=-1, n_jobs=1)
  m.fit(encode(data, code, labels), [disty(data,r) for r in labels],
        categorical_feature=syms(data) or "auto")
  return o(m=m, predictX=m.predict,
           predict=lambda row: m.predict(encode(data,code,[row]))[0])

def syms(data) -> list[int]:
  "Positions of the symbolic x columns."
  return [j for j,c in enumerate(data.cols.x) if c.it is Sym]

# ## Explanations to suggestions ---------------------------------------
# Each xFun returns candidate (column, new value) pairs, best first.
def leaves(tree, path=()):
  "Yield (leaf, its path of predicates) for every leaf."
  if not tree.kids: yield tree, path
  for kid in tree.kids: yield from leaves(kid, path + (kid.how,))

def nudge(col, v, sign) -> Any:
  "Move one value one Step along sign (symbols: to the most common other value)."
  if col.it is Sym:
    return max([(n,k) for k,n in col.has.items() if k != v],
               key=lambda t:t[0], default=(0,v))[1]
  v   = mid(col) if v == "?" else v
  new = min(col.hi, max(col.lo, v + sign * the.Step * col.sd))
  if isinstance(v, int): new = math.ceil(new) if sign > 0 else math.floor(new)
  return new

def xEzr(data, model, row, _) -> list:
  "Contrast row's path with the path to the best leaf; fix the first predicate it fails."
  leaf, path = min(leaves(model.tree), key=lambda lp: lp[0].mu)
  for op, at, y in path:
    if treeSelects(row, op, at, y): continue
    col  = data.cols.all[at]
    vals = [r[at] for r in leaf.rows if r[at] != "?"]
    # rows in that leaf all pass this predicate, so their mid passes it too
    return [(col, mid(adds(vals, (Num if col.it is Num else Sym)())))] if vals else []
  return []                                    # row is already in the best region

def xLime(data, model, row, aux) -> list:
  "Rank features by |lime weight|, then step against the weight's sign."
  x   = encode(data, aux.code, [row])[0]
  exp = aux.lime.explain_instance(x, model.predictX, num_features=len(x),
                                  num_samples=the.Lime)
  out = []
  for j,w in sorted(exp.local_exp[1], key=lambda jw: -abs(jw[1])):
    col = data.cols.x[j]
    if col.it is Sym and w <= 0: continue      # this category already helps
    out += [(col, nudge(col, row[col.at], -1 if w > 0 else 1))]
  return out

def xShap(data, model, row, aux) -> list:
  "Rank features by shap value (most harmful first), then step to lower it."
  phi = aux.shap.shap_values(encode(data, aux.code, [row]))[0]
  out = []
  for j in sorted(range(len(phi)), key=lambda j: -phi[j]):
    col = data.cols.x[j]
    if col.it is Sym and phi[j] <= 0: continue
    out += [(col, nudge(col, row[col.at], -aux.dirs[j]))]
  return out

def xRand(data, model, row, _) -> list:
  "Control: any feature, any direction."
  return [(c, nudge(c, row[c.at], random.choice([-1,1])))
          for c in random.sample(data.cols.x, len(data.cols.x))]

def suggest(row, cands) -> Any:
  "Copy row with ONE feature changed as the candidates advise. None if none change it."
  for col, new in cands:
    if new != row[col.at]:
      out = row[:]; out[col.at] = new
      return out

# ## Experiment -------------------------------------------------------
def Aux(data, code, model, rows) -> o:
  "Per-split explainer state, plus the sign that lowers each shap value."
  X, ex = encode(data, code, rows), shap.TreeExplainer(model.m)
  phi   = ex.shap_values(X)
  sign  = lambda j: np.sign(np.corrcoef(X[:,j], phi[:,j])[0,1]) \
                    if X[:,j].std() and phi[:,j].std() else 0
  return o(code=code, shap=ex, dirs=[sign(j) for j in range(X.shape[1])],
           lime=LimeTabularExplainer(X, mode="regression", random_state=the.seed,
                  discretize_continuous=False, categorical_features=syms(data) or None,
                  feature_names=[c.txt for c in data.cols.x]))

def run(file) -> tuple:
  "One dataset: per method the mean improvement of each repeat, and the exact-row rate."
  data = Data(csv(file))
  code = codes(data)
  t    = Truth(data, code)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo) / (b4.mu - b4.lo + 1e-32))
  out  = {k:[] for k in XAI}
  for seed in range(the.Repeats):
    the.seed = seed; random.seed(seed)
    rows  = random.sample(data.rows, len(data.rows))
    half  = len(rows)//2
    test  = clone(data, rows[:half])
    train = clone(data, rows[half:])
    labels = likely(train)                     # ezr buys Budget labels
    ezr    = Ezr(data, labels)
    lgbm   = Lgbm(data, code, labels)
    aux    = Aux(data, code, lgbm, random.sample(train.rows, min(train.n,512)))
    hows   = dict(ezr=(ezr,xEzr), lime=(lgbm,xLime), shap=(lgbm,xShap), rand=(lgbm,xRand))
    now    = {k:[] for k in XAI}
    for row in random.sample(test.rows, min(the.Points, test.n)):
      was = win(disty(data, row))              # the point is real, so this is exact
      for k,(model,xFun) in hows.items():
        new = suggest(row, xFun(data, model, row, aux))
        if new is None: now[k] += [0]
        elif the.Judge == "model":             # the old self-graded score
          now[k] += [win(model.predict(new)) - win(model.predict(row))]
        else:
          now[k] += [win(real(t, data, code, new)) - was]
    for k in XAI: out[k] += [mean(now[k])]
  return out, 100 * t.exact / (t.n or 1), t.r

def report(file, out, exact, r) -> set:
  "One line per dataset: median improvement, with '+' marking the best methods."
  best = top(out, reverse=True)
  print(f"{Path(file).stem[:28]:30}", end="")
  for k in XAI: print(f"{median(out[k]):8.1f}{'+' if k in best else ' '}", end="")
  print(f"{exact:7.0f}{r:7.2f}", flush=True)
  return best

def csvs(path) -> list[str]:
  "One csv, or every csv under a folder. A relative path also tries the repo root."
  p = Path(path)
  if not p.exists(): p = HERE.parent / path
  return [str(p)] if str(p).endswith(".csv") else sorted(
          str(q) for q in p.rglob("*.csv"))

def c2main():
  "top-level call"
  main(the, globals())
  print(f"{'Data':30}" + "".join(f"{k:>9}" for k in XAI) + f"{'exact%':>7}{'orc-r':>7}")
  wins = {k:0 for k in XAI}
  for f in csvs(the.file):
    for k in report(f, *run(f)): wins[k] += 1
  print(f"\n{'#best of ' + str(len(csvs(the.file))):30}"
        + "".join(f"{wins[k]:9}" for k in XAI))

if __name__ == "__main__": c2main()
