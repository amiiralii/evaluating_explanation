#!/usr/bin/env python3
"""
c7_trace.py: a step-by-step walkthrough of the "Local / C7" experiment, for one
dataset and one repeat. Calls the same functions as c7_local_stability.py in the
same order, so its closing score is exactly that repeat's score. It prints every
step: the labels ezr bought, the tree, and for the first Show points the
neighbour search, what changed between point and neighbour, each method's top
features on both sides, and the overlap.

Options:

    -f file=data/optimize/misc/auto93.csv    the dataset
    -s seed=0        which repeat to show
    -S Show=4        how many of the Points to print in full
    -W Wide=12       how many x columns to print per row
"""
from types import SimpleNamespace as o
from statistics import mean
from pathlib import Path
import re, sys, random

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import (Data, csv, clone, adds, disty, likely, treeShow, treeLeaf,
                       trace, coerce, main, the)
from c7_local_stability import (XAI, codes, encode, syms, Near, nearest, Ezr, Lgbm, csvs,
                                xEzr, xLime, xShap, xRand, jaccard)
from lime.lime_tabular import LimeTabularExplainer
import numpy as np, shap

the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})

# ## Printing ---------------------------------------------------------
def head(n, s): print(f"\n{'='*78}\n{n}. {s}\n{'='*78}")

def cells(row, cols) -> str:
  "Some columns of one row, as name=value text."
  txt = "  ".join(f"{c.txt}={row[c.at]}" for c in cols[:the.Wide])
  return txt + (f"  ...+{len(cols)-the.Wide} more" if len(cols) > the.Wide else "")

def diff(data, a, b) -> str:
  "The x columns where two rows differ, as name: old -> new."
  ds = [f"{c.txt}: {a[c.at]} -> {b[c.at]}" for c in data.cols.x if a[c.at] != b[c.at]]
  return ", ".join(ds[:6]) + (f"  ...+{len(ds)-6} more" if len(ds) > 6 else "")

def pair(sa, sb) -> str:
  "Two feature sets side by side, shared features marked with '='."
  both = sorted(sa & sb)
  return (f"{{{', '.join(both)}}}=" if both else "") + \
         f"  point only {{{', '.join(sorted(sa-sb))}}}  neighbour only {{{', '.join(sorted(sb-sa))}}}"

def bar(v, width=20) -> str:
  "A 0..1 value as a bar."
  n = round(width * v)
  return "#" * n + "." * (width - n)

# ## The walkthrough ---------------------------------------------------
def walk(file):
  "One dataset, one repeat, every step printed."
  seed = the.seed
  data = Data(csv(file))
  code = codes(data)
  X    = encode(data, code, data.rows)
  near = Near(data, X)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo)/(b4.mu - b4.lo + 1e-32))
  xFun = dict(ezr=xEzr, lime=xLime, shap=xShap, rand=xRand)

  head(1, f"THE DATASET   {file}")
  print(f"rows       : {len(data.rows)}")
  print(f"x columns  : {len(data.cols.x)}   {[c.txt for c in data.cols.x][:the.Wide]}"
        + (" ..." if len(data.cols.x) > the.Wide else ""))
  print(f"y columns  : {len(data.cols.y)}   {[c.txt for c in data.cols.y]}")
  print( "win        : 100 is the best row in this file, 0 is the average row")
  print( "the question: explain a point, explain a point just like it. do the two")
  print(f"              explanations name the same top {the.Top} features?")

  head(2, f"THE SPLIT, THE LABELS, THE TWO MODELS   (seed {seed}, Budget={the.Budget})")
  the.seed = seed; random.seed(seed)
  order  = random.sample(range(len(data.rows)), len(data.rows))
  half   = len(order)//2
  test   = order[:half]
  train  = clone(data, [data.rows[j] for j in order[half:]])
  labels = likely(train)
  ws     = [win(disty(data,r)) for r in labels]
  print(f"train {train.n} rows, test {len(test)} rows. ezr bought {len(labels)} labels,"
        f" spread win {max(ws):.1f} down to {min(ws):.1f}.")
  ezr  = Ezr(data, labels)
  lgbm = Lgbm(data, code, labels)
  print("\nthe ezr tree (win per node). its explanation of a row is that row's path:\n")
  treeShow(data, ezr.tree, lambda v: int(win(v)))
  print("\nlightgbm is fit on the same labels; lime and shap explain it.")
  bg  = encode(data, code, random.sample(train.rows, min(train.n,512)))
  aux = o(lgbm=lgbm, shap=shap.TreeExplainer(lgbm.m),
          lime=LimeTabularExplainer(bg, mode="regression", random_state=seed,
             discretize_continuous=False, categorical_features=syms(data) or None))
  both = lambda r: (win(ezr.predict(r)), win(lgbm.predictX(encode(data,code,[r]))[0]))

  head(3, f"WALKING {the.Points} RANDOM TEST POINTS   (first {the.Show} in full)")
  print(f"neighbour = nearest REAL row with different x whose prediction moves at most")
  print(f"{the.Eps} win units under BOTH models, among its {the.Near} nearest rows with different x.")
  print(f"each explanation is cut to its top {the.Top} features; overlap = shared / all.")
  now, selfs, n = {k:[] for k in XAI}, [], 0
  for j in random.sample(test, min(the.Points, len(test))):
    n   += 1
    say  = print if n <= the.Show else (lambda *_, **__: None)
    a, was = data.rows[j], both(data.rows[j])
    say(f"\n{'-'*78}\nPOINT {n}   {cells(a, data.cols.x)}")
    say(f"   predicted win: ezr {was[0]:.1f}, lightgbm {was[1]:.1f}   (true win {win(disty(data,a)):.1f})")
    d, i = near(j), None
    for c in nearest(d):
      now2 = both(data.rows[c])
      ok   = all(abs(p - q) <= the.Eps for p,q in zip(now2, was))
      say(f"   try row at distance {d[c]:.4f}: predicted {now2[0]:.1f} / {now2[1]:.1f}"
          f"   {'ACCEPT' if ok else 'reject, prediction moved too far'}")
      if ok: i = c; break
    if i is None:
      say("   no close enough neighbour: this point is skipped"); continue
    b    = data.rows[i]
    same = treeLeaf(ezr.tree, a) is treeLeaf(ezr.tree, b)
    say(f"   changed    : {diff(data, a, b)}")
    say(f"   ezr leaf   : {'SAME leaf, so ezr gives the same path by construction' if same else 'DIFFERENT leaf, paths diverge'}")
    for k in XAI:
      sa = xFun[k](data, ezr, a, X[j], aux)
      sb = xFun[k](data, ezr, b, X[i], aux)
      now[k] += [jaccard(sa, sb)]
      say(f"   [{k:4}] {now[k][-1]:.2f} {bar(now[k][-1])}  {pair(sa, sb)}")
    s1, s2 = xLime(data, ezr, a, X[j], aux), xLime(data, ezr, a, X[j], aux)
    selfs += [jaccard(s1, s2)]
    say(f"   lime twice on the SAME point: {selfs[-1]:.2f}  {pair(s1, s2)}")

  head(4, "THIS REPEAT'S SCORE   (mean overlap over the points with a neighbour)")
  for k in XAI:
    v = mean(now[k]) if now[k] else float("nan")
    print(f"   {k:5} {v:5.2f} {bar(v if v == v else 0, 30)}   from {len(now[k])} pairs")
  if selfs: print(f"   lime on the same point twice: {mean(selfs):.2f}")
  print(f"\nrand is what two unrelated sets of {the.Top} reach by chance. the real experiment")
  print(f"does this {the.Repeats} times per dataset, keeps one mean per repeat, then hands")
  print("the four lists to tools/stats.top, which marks the most stable with '+'.")

def c7trace():
  "top-level call"
  main(the, globals()); walk(csvs(the.file)[0])

if __name__ == "__main__": c7trace()
