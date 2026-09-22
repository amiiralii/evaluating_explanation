#!/usr/bin/env python3
"""
c2_trace.py: a step-by-step walkthrough of the "Local / C2" experiment, for one
dataset and one repeat. Calls the same functions as c2_local_counterfactual.py,
printing every step: the labels ezr bought, the tree it grew, what each xai
method says about a point, the one feature it changes, and how the model scores
the changed row.

Options:

    -f file=data/optimize/misc/auto93.csv    the dataset
    -P Points=5      how many random test points to walk through
    -s seed=0        which repeat to show
    -W Wide=12       how many x columns to print per row
"""
from statistics import mean
from pathlib import Path
import re, sys, random

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import (Data, csv, clone, adds, disty, likely, treeShow, treeLeaf,
                       treeSelects, trace as treeTrace, coerce, main, the)
from c2_local_counterfactual import (XAI, codes, encode, Ezr, Lgbm, Aux, Truth,
                                     real, leaves, xEzr, xLime, xShap, xRand, suggest)
import numpy as np

the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})

# ## Printing ---------------------------------------------------------
def head(n, s): print(f"\n{'='*74}\n{n}. {s}\n{'='*74}")

def cells(row, cols) -> str:
  "Some columns of one row, as name=value text."
  txt = "  ".join(f"{c.txt}={row[c.at]}" for c in cols[:the.Wide])
  return txt + (f"  ...+{len(cols)-the.Wide} more" if len(cols) > the.Wide else "")

def predicate(data, op, at, y) -> str:
  "One tree predicate as text."
  return f"{data.cols.names[at]} {op} {y}"

def changed(data, row, new):
  "The one x column where two rows differ."
  return next(c for c in data.cols.x if row[c.at] != new[c.at])

# ## The explanation each method gives for one point --------------------
def why(k, data, model, row, aux, ezr):
  "Print the raw explanation, before it is turned into a suggestion."
  if k == "ezr":
    leaf, path = min(leaves(ezr.tree), key=lambda lp: lp[0].mu)
    trail = treeTrace(data, ezr.tree, row)[:-1]
    print("     its path   :", " and ".join(f"{f} {op} {y}" for f,op,y,_,_ in trail) or "(root)")
    print("     best path  :", " and ".join(predicate(data,*p) for p in path))
    bad = next((p for p in path if not treeSelects(row, *p)), None)
    print("     it fails   :", predicate(data,*bad) if bad else "nothing, already inside")
  elif k == "lime":
    aux.lime.random_state = np.random.RandomState(the.seed)   # match the xLime call below
    x   = encode(data, aux.code, [row])[0]
    exp = aux.lime.explain_instance(x, model.predictX, num_features=len(x),
                                    num_samples=the.Lime)
    ws  = sorted(exp.local_exp[1], key=lambda jw: -abs(jw[1]))[:6]
    print("     weights    :", "  ".join(f"{data.cols.x[j].txt}{w:+.3f}" for j,w in ws))
    print("                  a positive weight raises disty, which is worse, so step the other way")
    aux.lime.random_state = np.random.RandomState(the.seed)
  elif k == "shap":
    phi = aux.shap.shap_values(encode(data, aux.code, [row]))[0]
    js  = sorted(range(len(phi)), key=lambda j: -phi[j])[:6]
    print("     shap       :", "  ".join(f"{data.cols.x[j].txt}{phi[j]:+.3f}" for j in js))
    print("                  positive means this value pushes disty up, so it is the harmful one")
  else:
    print("     control    : shuffle every column, pick a random direction")

# ## The walkthrough ---------------------------------------------------
def walk(file):
  "One dataset, one repeat, every step printed."
  data = Data(csv(file))
  code = codes(data)
  t    = Truth(data, code)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo)/(b4.mu - b4.lo + 1e-32))
  Win  = lambda r: win(disty(data, r))

  head(1, f"THE DATASET   {file}")
  print(f"rows       : {len(data.rows)}")
  print(f"x columns  : {len(data.cols.x)}   {[c.txt for c in data.cols.x][:the.Wide]}")
  print(f"y columns  : {len(data.cols.y)}   {[c.txt for c in data.cols.y]}")
  print( "             a trailing + is maximize, a trailing - is minimize")
  print(f"disty      : distance to heaven, 0 is perfect. best row {b4.lo:.3f}, average {b4.mu:.3f}")
  print( "win        : 100 is the best row in this file, 0 is the average row")

  head(2, "THE SPLIT")
  random.seed(the.seed)
  rows  = random.sample(data.rows, len(data.rows))
  half  = len(rows)//2
  test  = clone(data, rows[:half])
  train = clone(data, rows[half:])
  print(f"seed {the.seed}: shuffle, then halve. train {train.n} rows, test {test.n} rows")
  print("a row's y values are treated as unknown until the row is labelled, and labels cost money")

  head(3, f"WHAT EZR CHOSE TO LABEL   (Budget={the.Budget})")
  labels = likely(train)
  print(f"ezr spent {len(labels)} labels out of {train.n} available train rows.")
  print("it labels a few at random, splits them into best and rest, then keeps labelling")
  print("whichever unlabelled row looks nearer to best than to rest.\n")
  print("the labels it ended up with, best first:")
  for r in labels[:5]:  print(f"   win {Win(r):7.1f}   {cells(r, data.cols.x)}")
  if len(labels) > 7: print(f"   ... {len(labels)-7} more ...")
  for r in labels[-2:]: print(f"   win {Win(r):7.1f}   {cells(r, data.cols.x)}")
  ws = [Win(r) for r in labels]
  print(f"\nlabel spread: win {max(ws):.1f} down to {min(ws):.1f}")
  if max(ws) - min(ws) < 1:
    print("that is flat, so every model below will be near constant and every method will")
    print("score zero. this repeat cannot tell the methods apart. try another -s seed.")

  head(4, "THE EZR TREE   (grown on those labels only, win per node)")
  ezr = Ezr(data, labels)
  treeShow(data, ezr.tree, lambda v: int(win(v)))
  leaf, path = min(leaves(ezr.tree), key=lambda lp: lp[0].mu)
  print(f"\nbest leaf  : {len(leaf.rows)} rows, win {win(leaf.mu):.1f}")
  print(f"reached by : {' and '.join(predicate(data,*p) for p in path)}")
  print("that path is the ezr explanation: it is where the tree thinks good rows live")

  head(5, "THE LIGHTGBM MODEL   (exact same labels, exact same target)")
  lgbm = Lgbm(data, code, labels)
  imp  = sorted(zip([c.txt for c in data.cols.x], lgbm.m.feature_importances_),
                key=lambda t: -t[1])[:6]
  print("top splits :", "  ".join(f"{k}:{v}" for k,v in imp))
  fit = mean(abs(win(p) - Win(r)) for p,r in
             zip(lgbm.m.predict(encode(data, code, labels)), labels))
  print(f"fit error  : {fit:.1f} win units, mean absolute, on its own training labels")
  print("lime and shap both explain THIS model, so they are compared on equal footing with ezr")

  head(6, "EXPLAINER SETUP")
  bg  = random.sample(train.rows, min(train.n, 512))
  aux = Aux(data, code, lgbm, bg)
  print(f"background : {len(bg)} train rows, x values only, no labels needed")
  print("lime       : perturbs a point, then fits a local linear model to lgbm's output")
  print("shap       : TreeSHAP, plus one direction per feature, learned once from the background")
  for c,d in list(zip(data.cols.x, aux.dirs))[:8]:
    tag = {1:"lower it", -1:"raise it", 0:"flat, no direction"}[int(d)]
    print(f"             {c.txt:>14}  value/attribution correlation {int(d):+d}  so to help, {tag}")

  head(7, f"WALKING {the.Points} RANDOM TEST POINTS")
  got  = {k:[] for k in XAI}
  hows = dict(ezr=(ezr,xEzr), lime=(lgbm,xLime), shap=(lgbm,xShap), rand=(lgbm,xRand))
  for i,row in enumerate(random.sample(test.rows, min(the.Points, test.n))):
    print(f"\n{'-'*74}")
    print(f"POINT {i+1}   {cells(row, data.cols.x)}")
    print(f"  its real y : {cells(row, data.cols.y)}    win {Win(row):.1f}")
    print( "  no model ever sees that y. each one only predicts, and each starts from")
    print(f"  its own guess: ezr tree says win {win(ezr.predict(row)):.1f}, "
          f"lightgbm says win {win(lgbm.predict(row)):.1f}")
    for k,(model,xFun) in hows.items():
      print(f"\n   [{k}]")
      why(k, data, model, row, aux, ezr)
      cands = xFun(data, model, row, aux)
      print("     ranked     :", "  ".join(f"{c.txt}->{v}" for c,v in cands[:4]) or "nothing to offer")
      new = suggest(row, cands)
      if new is None:
        print( "     applied    : no change made, so no improvement to score")
        got[k] += [0]; continue
      col = changed(data, row, new)
      print(f"     applied    : {col.txt}   {row[col.at]}  ->  {new[col.at]}")
      print(f"     new row    : {cells(new, data.cols.x)}")
      if k == "ezr":       # one fixed predicate need not reach the best leaf
        print(f"     lands in   : a leaf with win {win(treeLeaf(ezr.tree,new).mu):.1f},"
              f" while the best leaf is win {win(leaf.mu):.1f}")
      was, seen = Win(row), t.exact             # the point is real, so was is exact
      now  = win(real(t, data, code, new))
      how  = "that exact row is in the data, so this is its real y" if t.exact > seen else \
             "that row is not in the data, so the forest oracle guessed it"
      print(f"     model says : win {win(model.predict(row)):.1f}  ->  "
            f"{win(model.predict(new)):.1f}   but the model is not the judge")
      print(f"     truth says : win {was:.1f}  ->  {now:.1f}      improvement {now-was:+.1f}")
      print(f"                  {how}")
      got[k] += [now - was]

  head(8, "THIS REPEAT'S SCORE")
  print("mean TRUE improvement over those points, in win units:\n")
  for k in XAI:
    print(f"   {k:5} {mean(got[k]):+8.1f}    from {[round(float(v),1) for v in got[k]]}")
  print(f"\nthe real experiment does this {the.Repeats} times per dataset, keeps one mean per")
  print("repeat, then hands the four lists to tools/stats.top, which marks the winners with '+'.")

def c2trace():
  "top-level call"
  main(the, globals()); walk(the.file)

if __name__ == "__main__": c2trace()
