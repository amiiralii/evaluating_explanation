#!/usr/bin/env python3
"""
c7_perturb.py: local explanation stability under one-feature perturbations.

Compare ezr paths, LIME/SHAP over LightGBM, and a fresh random explanation.
All learners use the same bought labels. Report signed-weight stability both
for all pairs and for pairs within Eps percent under BOTH predictors.

Options:

    -B Budget=50      maximum labels bought per split
    -c csv=0          one csv row per dataset, no header
    -E Eps=5          close predictions: symmetric percentage difference
    -L Lime=1000      samples per LIME explanation
    -P Points=20      test points per repeat
    -R Repeats=20     seeded 50/50 train/test splits
    -S Step=0.05      numeric perturbation as a fraction of observed range
    -T Top=3          largest absolute weights kept; 0 keeps all
    -f file=data/optimize/misc/auto93.csv    one csv, or a folder
"""
from types import SimpleNamespace as o
from statistics import mean, median
from pathlib import Path
import math, random, re, sys, warnings

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from tools.ezr import (Data, Sym, csv, clone, mid, disty, likely, Tree,
                       treeLeaf, treeSelects, coerce, main, the)
from tools.stats import top
import numpy as np, lightgbm as lgb, shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore", message="X does not have valid feature names")

the.__dict__.update({k:coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})
XAI = ["ezr", "lime", "shap", "rand"]
METRICS = ["stable", "features", "signs", "weights", "all", "self", "cross"]
DIAG = ["pairs", "changed_pct", "close_pct", "close_pairs", "close_repeats",
        "ezr_pred_pct", "lgbm_pred_pct", "ezr_close_pct", "lgbm_close_pct",
        "numeric_step_pct", "categorical_pct", "same_leaf_pct", "rules",
        *[f"empty_{k}_pct" for k in XAI]]
HEADER = ["dataset"] + XAI + ["best"] + [f"{m}_{k}" for m in METRICS[1:] for k in XAI] + DIAG

# ## Data and models ---------------------------------------------------
def codes(data):
  "Stable integer codes for symbolic features."
  return {c.at:{v:i for i,v in enumerate(sorted(c.has, key=str))}
          for c in data.cols.x if c.it is Sym}

def encode(data, code, rows):
  "Features as floats; replace missing values with column mids."
  def value(c, v):
    "Encode one value."
    v = mid(c) if v == "?" else v
    return code[c.at][v] if c.it is Sym else float(v)
  return np.array([[value(c,r[c.at]) for c in data.cols.x] for r in rows])

def Models(data, code, labels, train):
  "Fit both predictors and construct fixed-background explainers."
  tree = Tree(clone(data, labels), Y=lambda r:disty(data,r))
  cats = [j for j,c in enumerate(data.cols.x) if c.it is Sym]
  m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=8,
                        min_child_samples=the.leaf, random_state=the.seed,
                        verbose=-1, n_jobs=1)
  m.fit(encode(data,code,labels), [disty(data,r) for r in labels],
        categorical_feature=cats or "auto")
  bg = encode(data,code,random.sample(train.rows,min(train.n,512)))
  return o(tree=tree, m=m, shap=shap.TreeExplainer(m),
           lime=LimeTabularExplainer(bg, mode="regression", random_state=the.seed,
                  discretize_continuous=False, categorical_features=cats or None),
           rng=np.random.default_rng(the.seed))

# ## Perturbations -----------------------------------------------------
def perturb(data, row, rng):
  "Change one nonmissing, nonconstant feature; return row and actual step size."
  for c in rng.sample(data.cols.x, len(data.cols.x)):
    v = row[c.at]
    if v == "?" or not c.n: continue
    if c.it is Sym:
      choices = [x for x in sorted(c.has,key=str) if x != v]
      if not choices: continue
      new, step = rng.choice(choices), None
    else:
      if c.hi == c.lo: continue
      step = the.Step * (c.hi-c.lo)
      if isinstance(v,int): step = max(1,round(step))
      sign = rng.choice([-1,1])
      new = min(c.hi,max(c.lo,v+sign*step))
      if new == v: new = min(c.hi,max(c.lo,v-sign*step))
      step = 100 * abs(new-v)/(c.hi-c.lo)
    if new != v:
      out = row[:]; out[c.at] = new
      return out, step
  return None, None

def percent(a, b):
  "Symmetric relative difference: both zero -> 0%, one zero -> 200%."
  scale = abs(a) + abs(b)
  return 200 * abs(a-b)/scale if scale else 0.0

# ## Signed explanations ----------------------------------------------
def path(tree, row):
  "Sum child-minus-parent prediction changes by feature; retain the raw rules."
  out, rules = {}, set()
  while True:
    kid = next((k for k in tree.kids if treeSelects(row,*k.how)), None)
    if kid is None: return out, rules
    at = kid.how[1]
    out[at] = out.get(at,0) + kid.mu-tree.mu
    rules.add(kid.how)
    tree = kid

def keep(weights):
  "Keep nonzero features in descending absolute-weight order, ties by position."
  order = sorted((j for j,w in weights.items() if abs(w)>1e-12),
                  key=lambda j:(-abs(weights[j]),j))
  return {j:float(weights[j]) for j in (order[:the.Top] if the.Top else order)}

def explain(data, model, row, x):
  "Return each method's own signed weights; compare weights only within a method."
  ezr, rules = path(model.tree,row)
  exp = model.lime.explain_instance(x,model.m.predict,num_features=len(x),
                                     num_samples=the.Lime)
  phi = np.asarray(model.shap.shap_values(x[None,:])).reshape(-1)
  rand = model.rng.uniform(-1,1,len(x))
  return dict(ezr=keep(ezr), lime=keep({data.cols.x[j].at:w for j,w in exp.local_exp[1]}),
              shap=keep({c.at:phi[j] for j,c in enumerate(data.cols.x)}),
              rand=keep({c.at:rand[j] for j,c in enumerate(data.cols.x)})), rules

def overlap(a, b):
  "Jaccard feature/rule overlap; two empty sets agree."
  a, b = set(a), set(b)
  return len(a & b)/len(a | b) if a | b else 1.0

def agreement(a, b):
  "Signed stability, feature overlap, shared-feature sign agreement, weight stability."
  keys = set(a) | set(b)
  mass = sum(abs(a.get(j,0))+abs(b.get(j,0)) for j in keys)
  signed = sum(abs(a.get(j,0)-b.get(j,0)) for j in keys)
  weight = sum(abs(abs(a.get(j,0))-abs(b.get(j,0))) for j in keys)
  shared = set(a) & set(b)
  signs = mean(a[j]*b[j]>0 for j in shared) if shared else float("nan")
  return (1-signed/mass if mass else 1.0, overlap(a,b), signs,
          1-weight/mass if mass else 1.0)

# ## Experiment --------------------------------------------------------
def avg(values):
  "Mean of finite values, or NaN when nothing was measured."
  values = [v for v in values if math.isfinite(v)]
  return mean(values) if values else float("nan")

def run(file, show=None, setup=None, seeds=None):
  "One dataset: retain repeat means and report coverage and prediction closeness."
  data = Data(csv(file))
  if not data.cols.x or not data.cols.y or len(data.rows)<8:
    raise ValueError(f"{file}: need features, objectives and at least eight rows")
  if any(c.it is Sym and not c.has for c in data.cols.x):
    raise ValueError(f"{file}: all-missing symbolic feature has no imputation value")
  code = codes(data)
  out = {m:{k:[] for k in XAI} for m in METRICS}
  pred, steps, rules = [[],[]], [], []
  tried = pairs = close = same = cats = 0
  empty = {k:0 for k in XAI}
  budget = the.Budget
  try:
    for seed in (range(the.Repeats) if seeds is None else seeds):
      the.seed = seed; random.seed(seed)
      rows = random.sample(data.rows,len(data.rows)); half = len(rows)//2
      train = clone(data,rows[half:])
      the.Budget = min(budget,train.n)
      labels = likely(train)
      model = Models(data,code,labels,train)
      if setup: setup(data,seed,train,rows[:half],labels,model)
      rng = random.Random(seed)
      now = {m:{k:[] for k in XAI} for m in METRICS}
      for point,row in enumerate(rng.sample(rows[:half],min(the.Points,half)),1):
        tried += 1
        new, step = perturb(data,row,rng)
        if new is None: continue
        pairs += 1
        if step is None: cats += 1
        else: steps.append(step)
        x = encode(data,code,[row,new])
        ep = [treeLeaf(model.tree,r).mu for r in [row,new]]
        lp = model.m.predict(x)
        changes = [percent(*ep),percent(*lp)]
        for i,v in enumerate(changes): pred[i].append(v)
        near = max(changes)<=the.Eps
        close += near
        leaf = treeLeaf(model.tree,row) is treeLeaf(model.tree,new)
        same += leaf
        a, ra = explain(data,model,row,x[0])
        b, rb = explain(data,model,new,x[1])
        again, _ = explain(data,model,row,x[0])
        if show: show(data,seed,row,new,ep,lp,a,b,ra,rb,again,model,point)
        for k in XAI: empty[k] += not a[k] and not b[k]
        if near: rules.append(overlap(ra,rb))
        for k in XAI:
          score = agreement(a[k],b[k])
          now["all"][k].append(score[0])
          now["self"][k].append(agreement(a[k],again[k])[0])
          if near:
            for m,v in zip(METRICS[:4],score): now[m][k].append(v)
            if not leaf: now["cross"][k].append(score[0])
      for m in METRICS:
        for k in XAI:
          if math.isfinite(v := avg(now[m][k])): out[m][k].append(v)
  finally:
    the.Budget = budget
  diag = dict(pairs=pairs, changed_pct=100*pairs/(tried or 1),
              close_pct=100*close/(pairs or 1), close_pairs=close,
              close_repeats=len(out["stable"]["ezr"]),
              ezr_pred_pct=avg(pred[0]), lgbm_pred_pct=avg(pred[1]),
              ezr_close_pct=100*sum(v<=the.Eps for v in pred[0])/(pairs or 1),
              lgbm_close_pct=100*sum(v<=the.Eps for v in pred[1])/(pairs or 1),
              numeric_step_pct=avg(steps), categorical_pct=100*cats/(pairs or 1),
              same_leaf_pct=100*same/(pairs or 1), rules=avg(rules))
  diag.update({f"empty_{k}_pct":100*empty[k]/(pairs or 1) for k in XAI})
  return out, diag

# ## Reporting ---------------------------------------------------------
def report(file, out, diag):
  "One dataset: medians of repeat means, plus pooled coverage diagnostics."
  best = top(out["stable"],reverse=True) if all(len(v)>1 for v in out["stable"].values()) else set()
  med = lambda m,k: median(out[m][k]) if out[m][k] else float("nan")
  name = Path(file).stem
  values = [f"{med('stable',k):.3f}" for k in XAI]
  rest = [f"{med(m,k):.3f}" for m in METRICS[1:] for k in XAI]
  rest += [f"{diag[k]:.3f}" for k in DIAG]
  if the.csv:
    print(",".join([name]+values+[" ".join(k for k in XAI if k in best)]+rest),flush=True)
  else:
    print(f"{name:30}"+"".join(f"{v:>8}{'+' if k in best else ' '}" for k,v in zip(XAI,values)),flush=True)
    for m in METRICS[1:]:
      print(f"  {m:28}"+"".join(f"{med(m,k):9.3f}" for k in XAI))
    print("  "+"  ".join(f"{k}={diag[k]:.2f}" for k in DIAG),flush=True)


def csvs(path):
  "One csv, or sorted csv files under a folder; relative paths try the repo root."
  p = Path(path)
  if not p.exists(): p = HERE.parent/path
  files = [str(p)] if p.is_file() else sorted(str(q) for q in p.rglob("*.csv"))
  if not files: raise ValueError(f"No datasets at {p}")
  return files

def eg__header():
  "Print the runner's csv header without fitting models."
  print(",".join(HEADER)); sys.exit(0)

def c7main():
  "Read options and run only the requested file or folder."
  main(the,globals())
  if not (0<the.Step<=1 and the.Eps>=0 and the.Top>=0 and the.Repeats>0
          and the.Points>0 and the.Budget>=the.Any):
    raise ValueError("Invalid step, tolerance, top, repeats, points or budget")
  if not the.csv: print(f"{'Data / metric':30}"+"".join(f"{k:>9}" for k in XAI))
  for f in csvs(the.file): report(f,*run(f))

if __name__ == "__main__": c7main()
