#!/usr/bin/env python3
"One-off checks for signed stability, percentage closeness, perturbations and feature mapping."
from types import SimpleNamespace as o
import math, random
import numpy as np
from experiment3_perturb import (Data, agreement, percent, perturb, path, explain, codes,
                       encode, the)


def checks():
  "Catch sign reversals, scale changes, invalid mutations and objective-index leakage."
  a = {0:1,2:-2}
  assert agreement(a,a) == (1,1,1,1)
  flipped = agreement(a,{0:-1,2:2})
  assert flipped == (0,1,0,1)              # same features/magnitudes, opposite advice
  assert abs(agreement({0:1},{0:2})[0]-2/3)<1e-12
  assert agreement({0:1},{1:1})[0:2] == (0,0)
  assert math.isnan(agreement({0:1},{1:1})[2])
  assert agreement({}, {})[0] == 1
  assert percent(0,0) == 0 and percent(0,1) == 200
  assert percent(1,1.05) < 5 and percent(1,1.1) > 5
  assert percent(1,1.05) == percent(1.05,1)
  assert percent(-1,1) == 200
  data = Data([["A","Goal-","b","C","constant"],
               [0,10,"off",0,"fixed"],[10,20,"on",100,"fixed"]])
  original = data.rows[0][:]
  for seed in range(40):
    new, step = perturb(data,original,random.Random(seed))
    changed = [j for j,(a,b) in enumerate(zip(original,new)) if a!=b]
    assert len(changed)==1 and changed[0] in [0,2,3]
    assert new[1]==original[1] and new[4]=="fixed" and original==data.rows[0]
    if changed[0]!=2:
      c = data.cols.all[changed[0]]
      assert c.lo<=new[c.at]<=c.hi and isinstance(new[c.at],int)
      assert step>0
  missing = ["?",10,"?","?","fixed"]
  assert perturb(data,missing,random.Random(0)) == (None,None)
  child = o(mu=.2,how=("<=",0,5),kids=[])
  tree = o(mu=.8,kids=[child])
  weights, rules = path(tree,original)
  assert abs(sum(weights.values())-(child.mu-tree.mu))<1e-12
  model = o(tree=tree, m=o(predict=None),rng=np.random.default_rng(0),
            lime=o(explain_instance=lambda *a,**k:o(local_exp={1:[(0,1),(1,-2),(2,3),(3,0)]})),
            shap=o(shap_values=lambda x:np.array([[1,-2,3,0]])))
  out, _ = explain(data,model,original,encode(data,codes(data),[original])[0])
  assert set(out['lime']) == set(out['shap']) == {0,2,3}  # column 1 is an objective
  print("passed: signed metrics, percentage closeness, one-feature changes, missing values, tree weights, feature mapping")

if __name__ == '__main__': checks()
