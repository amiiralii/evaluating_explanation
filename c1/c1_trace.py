#!/usr/bin/env python3
"""
c1_trace.py: a step-by-step walkthrough of the "Global / C1, C4" experiment, for
one dataset and one repeat. Calls the same functions as c1_global_topk.py,
printing every step: how K is fixed, the labels ezr bought, the two full models,
each method's global feature ranking as a bar chart, which K features each method
keeps, the rows each retrained model picks from the test half, and the score.

Options:

    -f file=data/optimize/misc/auto93.csv    the dataset
    -s seed=0        which repeat to show
    -W Wide=12       how many features to draw in the bar charts
"""
from types import SimpleNamespace as o
from pathlib import Path
import re, sys, random

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root
from tools.ezr import (Data, Sym, csv, clone, adds, disty, likely, Tree, treeShow,
                       coerce, main, the)
from c1_global_topk import (XAI, LRN, codes, encode, Ezr, Lgbm, used, csvs,
                            xEzr, xLime, xShap, xRand, topk)
from lime.lime_tabular import LimeTabularExplainer
from statistics import median

the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})

# ## Printing ---------------------------------------------------------
def head(n, s): print(f"\n{'='*78}\n{n}. {s}\n{'='*78}")

def bar(v, most, width=30) -> str:
  "A horizontal bar, v as a share of most."
  n = round(width * v / most) if most > 0 else 0
  return "#" * n + "." * (width - n)

def names(data, ats) -> str:
  "Column names for a set of column positions, in file order."
  return " ".join(data.cols.names[a] for a in sorted(ats)) or "(none)"

# ## The walkthrough ---------------------------------------------------
def walk(file):
  "One dataset, one repeat, every step printed."
  seed = the.seed
  data = Data(csv(file))
  code = codes(data)
  b4   = adds(disty(data,r) for r in data.rows)
  win  = lambda v: 100 * (1 - (v - b4.lo)/(b4.mu - b4.lo + 1e-32))
  Win  = lambda r: win(disty(data, r))
  every = {c.at for c in data.cols.x}

  head(1, f"THE DATASET   {file}")
  print(f"rows       : {len(data.rows)}")
  print(f"x columns  : {len(data.cols.x)}   {[c.txt for c in data.cols.x][:the.Wide]}"
        + (" ..." if len(data.cols.x) > the.Wide else ""))
  print(f"y columns  : {len(data.cols.y)}   {[c.txt for c in data.cols.y]}")
  print(f"disty      : distance to heaven, 0 is perfect. best row {b4.lo:.3f}, average {b4.mu:.3f}")
  print( "win        : 100 is the best row in this file, 0 is the average row")
  print( "the question: if each method may keep only K features, does a model retrained")
  print( "              on those K still find rows as good as the model that saw them all?")

  head(2, f"FIXING K, BEFORE ANY REPEAT   ({the.Tries} ezr trees, each on a random half)")
  ns = []
  for i in range(the.Tries):
    the.seed = 1000 + i; random.seed(the.seed)
    half = clone(data, random.sample(data.rows, len(data.rows)//2))
    fs   = used(Tree(clone(data, likely(half))))
    ns  += [len(fs)]
    print(f"   tree {i+1:2}  {len(fs):3} features  {'#'*len(fs):<{len(data.cols.x)}}  {names(data, fs)[:60]}")
  k = max(1, round(median(ns)))
  print(f"\nK = median of {ns} = {k}   (out of {len(data.cols.x)} x columns)")
  print("only the COUNT is kept. which features those trees used is thrown away, so K")
  print("says how many features a good explanation may name, not which ones.")

  head(3, f"THE SPLIT AND THE LABELS   (seed {seed}, Budget={the.Budget})")
  the.seed = seed; random.seed(seed)
  rows   = random.sample(data.rows, len(data.rows))
  half   = len(rows)//2
  test   = rows[:half]
  train  = clone(data, rows[half:])
  labels = likely(train)
  ws     = [Win(r) for r in labels]
  print(f"train {train.n} rows, test {len(test)} rows. ezr bought {len(labels)} labels from train.")
  print(f"label spread: win {max(ws):.1f} down to {min(ws):.1f}")
  if max(ws) - min(ws) < 1:
    print("that is flat: every model is constant and every method scores zero. try another -s seed.")
  print("every model below, full or reduced, trains on exactly these labels.")

  head(4, "THE TWO FULL MODELS   (all features, the predecessors)")
  X   = encode(data, code, random.sample(train.rows, min(train.n,512)))
  aux = o(X=X, lime=LimeTabularExplainer(X, mode="regression", random_state=seed,
             discretize_continuous=False, feature_names=[c.txt for c in data.cols.x],
             categorical_features=[j for j,c in enumerate(data.cols.x) if c.it is Sym] or None))
  learn = dict(ezr=lambda keep: Ezr(data, labels, keep),
               lgbm=lambda keep: Lgbm(data, code, labels, keep))
  was   = {l: learn[l](every) for l in LRN}
  treeShow(was["ezr"].data, was["ezr"].tree, lambda v: int(win(v)))
  own = used(was["ezr"].tree)
  print(f"\nthis tree splits on {len(own)} features; K is {k}.")
  if len(own) <= k:
    print("since that is not more than K, MDI's top K will contain every one of them, and")
    print("retraining ezr on them grows this SAME tree: ezr-on-ezr scores 0 by construction.")
  print("\nlightgbm, same labels, same target, all features: lime and shap explain this one.")

  head(5, "EACH METHOD RANKS THE FEATURES GLOBALLY")
  hows = dict(ezr=(was["ezr"],xEzr,  "mean decrease in impurity over the ezr tree's splits"),
              lime=(was["lgbm"],xLime,f"mean |lime weight| over {the.Global} train rows, on lightgbm"),
              shap=(was["lgbm"],xShap,f"mean |shap value| over {len(X)} train rows, on lightgbm"),
              rand=(None,xRand,       "control: a random number per feature"))
  keep = {}
  for x,(model,xFun,how) in hows.items():
    imp  = xFun(data, model, aux)
    keep[x] = topk(data, imp, k)
    most = max(imp.values())
    print(f"\n[{x}]  {how}")
    for c in sorted(data.cols.x, key=lambda c: -imp[c.at])[:the.Wide]:
      mark = "KEEP" if c.at in keep[x] else "    "
      print(f"   {c.txt[:18]:>18} {bar(imp[c.at], most)} {imp[c.at]:9.4f}  {mark}")
    if len(data.cols.x) > the.Wide: print(f"   {'...':>18} {len(data.cols.x)-the.Wide} more")
    zeros = sum(1 for c in data.cols.x if imp[c.at] == 0)
    if zeros and x == "ezr" and zeros > len(data.cols.x) - k:
      print(f"   {zeros} features score exactly 0, so some KEEPs above were picked at random among them")

  head(6, f"WHO KEPT WHAT   (K = {k}; # kept, . dropped)")
  union = sorted(set().union(*keep.values()))
  print(f"{'':20}" + "".join(f"{x:>6}" for x in XAI))
  for a in union:
    print(f"{data.cols.names[a][:18]:>18}  " + "".join(f"{'#' if a in keep[x] else '.':>6}" for x in XAI))
  print(f"features no method kept: {len(data.cols.x) - len(union)}")

  head(7, f"RETRAIN ON EACH TOP-K, THEN PICK FROM THE TEST HALF   (Check={the.Check})")
  print(f"each model sorts the {len(test)} test rows by its prediction and picks its top {the.Check}.")
  print("test rows are real, so their true win is looked up, never estimated.")
  print(f"a model's score is the best TRUE win among its {the.Check} picks.\n")
  best = max(Win(r) for r in test)
  print(f"for reference: the best row in the test half is win {best:.1f}\n")
  got = {}
  for l in LRN:
    print(f"--- learner: {l} {'-'*(60-len(l))}")
    for x in ["full"] + XAI:
      model = was[l] if x == "full" else learn[l](keep[x])
      picks = [Win(r) for r in model.rank(test)[:the.Check]]
      got[l,x] = max(picks)
      what = "all features" if x == "full" else f"{x}'s top {k}"
      print(f"   {what:>16}  picks {' '.join(f'{p:6.1f}' for p in picks)}"
            f"   best {got[l,x]:6.1f}  {bar(max(0,got[l,x]), 100, 20)}")
    print()

  head(8, "THIS REPEAT'S SCORE   (reduced minus full; 0 = nothing lost)")
  print(f"{'':10}" + "".join(f"{x:>9}" for x in XAI))
  for l in LRN:
    print(f"{l:>8}  " + "".join(f"{got[l,x] - got[l,'full']:+9.1f}" for x in XAI))
  print(f"\nthe real experiment does this {the.Repeats} times per dataset, keeps one number per")
  print("repeat and learner, then hands each learner's four lists to tools/stats.top,")
  print("which marks the winners with '+'.")

def c1trace():
  "top-level call"
  main(the, globals()); walk(csvs(the.file)[0])

if __name__ == "__main__": c1trace()
