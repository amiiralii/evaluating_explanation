#!/usr/bin/env python3
"""
trace.py: walkthrough of one actual C7 repeat, using c7_perturb.run callbacks.
Print data, bought labels, fitted models, direct changes, explanations and scores.
Other c7_perturb options (Budget, Lime, Step, Eps, Top) also work here.

Options:

    -f file=data/optimize/misc/auto93.csv    example dataset
    -P Points=5      random held-out points to walk through
    -s seed=0        repeat to trace
    -W Wide=12       feature values displayed per row
"""
import re, shlex
from c7_perturb import (XAI, run, report, csvs, agreement, percent, overlap,
                       main, the, coerce, Sym, treeLeaf, treeSelects)
from tools.ezr import treeNodes, disty

the.__dict__.update({k:coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})

def head(title):
  "Separate walkthrough steps."
  print(f"\n{'='*78}\n{title}\n{'='*78}")

def cells(data, row):
  "Display a bounded number of feature values."
  cols = data.cols.x
  text = "  ".join(f"{c.txt}={row[c.at]}" for c in cols[:the.Wide])
  return text + (f"  ... {len(cols)-the.Wide} more" if len(cols)>the.Wide else "")

def setup(data, seed, train, test, labels, model):
  "Print the actual split, labelled training rows, and models without drawing randomness."
  head("1. THE DATASET")
  print(f"rows: {len(data.rows)}; features: {len(data.cols.x)}; objectives: {len(data.cols.y)}")
  print("features:",*[c.txt for c in data.cols.x[:the.Wide]])
  print("objectives:",*[c.txt for c in data.cols.y])
  print("Both models predict distance-to-heaven on full-dataset objective bounds; lower is better.")
  head(f"2. THE SPLIT AND BOUGHT LABELS (seed {seed}, Budget {the.Budget})")
  print(f"train: {train.n}; held-out test: {len(test)}; labels bought: {len(labels)}")
  print("EZR's active learner chooses labels from train; both models fit exactly those labels.")
  print("Example labelled rows (first five in the learner's returned order):")
  for r in labels[:5]: print(f"  target={disty(data,r):.6f}  {cells(data,r)}")
  ys = [disty(data,r) for r in labels]
  print(f"label target range: {min(ys):.6f} to {max(ys):.6f}")
  head("3. THE FITTED MODELS")
  print("EZR tree (raw distance-to-heaven means):")
  for depth,node in treeNodes(model.tree):
    rule = "root" if node.how is None else f"{data.cols.names[node.how[1]]} {node.how[0]} {node.how[2]}"
    print(f"  {'|  '*depth}{rule}: n={len(node.rows)}, prediction={node.mu:.6f}")
  splits = sorted(zip(data.cols.x,model.m.feature_importances_),key=lambda cw:-cw[1])
  print("LightGBM split counts:","  ".join(f"{c.txt}:{n}" for c,n in splits[:the.Wide]))
  head("4. EXPLANATIONS AND PERTURBATION SETTINGS")
  print(f"Top={the.Top} (0 keeps all nonzero weights); Step={100*the.Step:g}% of numeric range")
  print(f"LIME samples={the.Lime}; background={min(train.n,512)} unlabelled train feature rows")
  print("EZR weights: child prediction minus parent prediction, summed by feature along the path.")
  print("LIME weights: standardized numeric coefficients / categorical membership coefficients.")
  print("SHAP weights: contributions relative to LightGBM's expected prediction.")
  print("Random: fresh signed uniform weights; each method keeps its largest absolute weights.")
  print("Compare each method with itself; weights from different methods have different meanings.")
  print("One feature changes; integer values stay integer, categories switch to another observed value.")
  print(f"Close = 200*|p-q|/(|p|+|q|) <= {the.Eps:g}% under BOTH predictors.")
  print("Both zero: 0%; one zero: 200%. Larger changes still contribute to the all-pairs score.")
  print("Same-point explanations are drawn again to reveal LIME/random sampling noise.")


def trail(data, tree, row):
  "Print each predicate's signed contribution and the full path sum."
  root = tree.mu
  while True:
    kid = next((k for k in tree.kids if treeSelects(row,*k.how)),None)
    if kid is None: break
    op,at,value = kid.how
    print(f"    {data.cols.names[at]} {op} {value}: {tree.mu:.6f} -> {kid.mu:.6f}; delta={kid.mu-tree.mu:+.6f}")
    tree = kid
  print(f"    full path sum={tree.mu-root:+.6f} = leaf {tree.mu:.6f} - root {root:.6f}")


def show(data, seed, row, new, ep, lp, a, b, ra, rb, again, model, point):
  "Print one real experiment pair, native path contributions, weights and scoring arithmetic."
  head(f"5. TEST POINT {point} AND ITS DIRECT PERTURBATION")
  print("before:",cells(data,row))
  print("after: ",cells(data,new))
  col = next(c for c in data.cols.x if row[c.at]!=new[c.at])
  print(f"changed only {col.txt}: {row[col.at]} -> {new[col.at]}")
  if col.it is Sym:
    print("categorical switch; observed categories:",sorted(col.has,key=str))
  else:
    step = 100*abs(new[col.at]-row[col.at])/(col.hi-col.lo)
    print(f"bounds [{col.lo}, {col.hi}]; actual change={step:.2f}% of range (after rounding/clipping)")
  print("Objectives stay untouched and are not used to score this artificial point's outcome.")
  delta = [percent(*ep),percent(*lp)]
  for name,p,d in zip(["EZR","LightGBM"],[ep,lp],delta):
    calc = f"200*{abs(p[0]-p[1]):.6f}/{abs(p[0])+abs(p[1]):.6f}" if abs(p[0])+abs(p[1]) else "both zero"
    print(f"{name}: {p[0]:.6f} -> {p[1]:.6f}; {calc} = {d:.3f}%")
  near = max(delta)<=the.Eps
  print(f"both within {the.Eps:g}%: {near}; included in {'close and all-pairs' if near else 'all-pairs ONLY'} scores")
  same = treeLeaf(model.tree,row) is treeLeaf(model.tree,new)
  print(f"same EZR leaf: {same}; native rule overlap: {overlap(ra,rb):.3f}")
  print("EZR path before:"); trail(data,model.tree,row)
  print("EZR path after:"); trail(data,model.tree,new)
  for k in XAI:
    print(f"\n[{k}] weights after top-feature selection (absent features have weight 0)")
    print(f"  {'feature':20} {'before':>11} {'after':>11} {'same input again':>17}  sign / membership")
    keys = sorted(set(a[k])|set(b[k])|set(again[k]))
    for j in keys:
      p,q = a[k].get(j,0),b[k].get(j,0)
      change = "same-input run only" if j not in a[k] and j not in b[k] else "removed" if j not in b[k] else "added" if j not in a[k] else "same sign" if p*q>0 else "SIGN FLIPPED"
      print(f"  {data.cols.names[j]:20} {p:+11.6f} {q:+11.6f} {again[k].get(j,0):+17.6f}  {change}")
    if not keys: print("  (empty explanations)")
    union = set(a[k])|set(b[k])
    mass = sum(abs(a[k].get(j,0))+abs(b[k].get(j,0)) for j in union)
    diff = sum(abs(a[k].get(j,0)-b[k].get(j,0)) for j in union)
    mag = sum(abs(abs(a[k].get(j,0))-abs(b[k].get(j,0))) for j in union)
    signed,features,signs,weights = agreement(a[k],b[k])
    print(f"  mass={mass:.6f}; signed difference={diff:.6f}; magnitude difference={mag:.6f}")
    print(f"  signed stability: 1 - signed difference/mass = {signed:.6f}")
    print(f"  weight stability: 1 - magnitude difference/mass = {weights:.6f}")
    print(f"  feature overlap={features:.6f}; shared-feature sign agreement={signs:.6f}")
    print(f"  same-input signed stability={agreement(a[k],again[k])[0]:.6f}")
    if not mass: print("  Both empty: stability defined as 1; this supplies no nonzero explanatory content.")


def walk(file):
  "Trace exactly one seed through the experiment and print its actual report."
  print("Reproduce from the repository root:")
  print(f"PYTHONHASHSEED=0 python3 experiment3/trace.py -f {shlex.quote(file)}"
        f" -s {the.seed} -P {the.Points} -L {the.Lime} -B {the.Budget}"
        f" -T {the.Top} -E {the.Eps} -S {the.Step} -W {the.Wide}")
  out,diag = run(file,show,setup,seeds=[the.seed])
  head("6. THIS REPEAT'S REPORT")
  print("Rows below are means over this repeat's measured pairs; NaN means no eligible measurement.")
  print("Primary scores use close pairs; all uses every pair; cross uses close pairs in different EZR leaves.")
  print("A one-repeat trace does not identify statistically best methods.")
  report(file,out,diag)
  print("The full experiment repeats this over 20 splits by default, then compares repeat means.")

if __name__ == "__main__":
  main(the,{})
  for file in csvs(the.file): walk(file)
