# Global / C1, C4: do an explainer's top K features carry the signal?

`sh c1/sweep.sh 12` over 128 MOOT datasets, 20 repeats, Budget 50 labels,
Check 5, LIME averaged over 20 train rows, `PYTHONHASHSEED=0`. About 56 minutes
wall clock. Final report in `c1_results.csv`, the same plus K, n and the full
models' scores in `c1_results_full.csv`, worked examples of one repeat in
`c1_trace_X264.txt` (informative) and `c1_trace_auto93.txt` (flat).

## What is measured

1. **K** is fixed per dataset before any repeat: grow 10 ezr trees, each on a
   random half with its own 50 labels, count the distinct features each uses,
   take the median. Median K/n is 0.67.
2. Each repeat buys 50 labels with `likely()` and fits a full ezr tree and a full
   LightGBM on them.
3. Each method ranks the features globally: **ezr** by mean decrease in impurity
   over the full tree, **lime** by mean |weight| of LightGBM's explanations at 20
   train rows, **shap** by mean |shap value| of LightGBM, **rand** at random.
4. Each method's top K retrains BOTH learners on the same 50 labels. The
   dropped columns stay in the rows; the learner just may not use them.
5. **Score**: the model sorts the test half and we take the TRUE win of the best
   of its first 5 picks, minus the full model's score on the same split. Test rows
   are real, so there is no oracle and nothing is estimated. 0 means nothing was
   lost by dropping the other features; negative is a loss.

Columns: `e_*` retrain the ezr tree, `l_*` retrain LightGBM; `*_best` lists the
methods `tools/stats.top` could not separate from the best.

## Results

"wins" counts datasets where a method is among the statistically best.

| group                               |   n | learner | ezr | lime | shap | rand |
|-------------------------------------|----:|---------|----:|-----:|-----:|-----:|
| all datasets, wins                  | 128 | ezr     |  98 |  119 |  114 |   76 |
|                                     |     | lgbm    |  92 |  124 |  122 |   71 |
| all datasets, mean change           |     | ezr     | 0.0*|  1.3 |  0.8 | -2.4 |
|                                     |     | lgbm    |-3.1 | -0.7 | -0.6 | -7.3 |
| K < n, rand separated on lgbm, wins |  57 | ezr     | 35* |   51 |   49 |   16 |
|                                     |     | lgbm    |  25 |   54 |   52 |    0 |
| K < n, rand separated on lgbm, mean |     | ezr     | 0.0*|  2.4 |  1.8 | -5.2 |
|                                     |     | lgbm    |-6.8 | -1.3 | -1.2 |-16.0 |

\* the ezr-on-ezr cell is 0 by construction; see below.

The "separated" rows use the 57 datasets where random is not among the best on
LightGBM, for both learners. `c1_tally.csv` has the same counts per learner,
with the subset defined per learner instead (52 datasets for the ezr learner),
and per dataset family: for each method, how often it is among the best
(`_best`) and how often it is the only best (`_sole`). `ezr_best` on
`learner=ezr` counts the constant-0 cell, not skill. Regenerate it with
`python3 c1/c1_tally.py > c1/results/c1_tally.csv`.

**Headline.** LIME and SHAP pick the most representative features, and they are
nearly indistinguishable from each other. On the 57 datasets where the random
control is separated from the best, the LIME and SHAP feature sets lose about 1
win unit when LightGBM is retrained on them; ezr's MDI features lose 6.8, and
random loses 16. Ezr's features are clearly better than random, but not as
good as LIME's or SHAP's.

LIME and SHAP's features also *help* the ezr tree: its score rises on 24 datasets
with LIME's top K and on 16 with SHAP's, falling on only 2 each. A greedy tree grown
on 50 labels benefits from being kept away from features that look useful in
those 50 rows but are not.

## The split that matters

| family        |  n | lgbm learner wins: ezr / lime / shap / rand | mean ezr / lime / shap / rand |
|---------------|---:|---------------------------------------------|-------------------------------|
| binary_config | 12 |  0 / 11 / 10 /  0                            | -22.8 / -4.7 / -4.6 / -48.6   |
| config        | 35 | 27 / 35 / 35 / 16                            |  -2.5 / -0.6 / -0.3 /  -3.9   |
| hpo           | 35 | 28 / 35 / 35 / 29                            |  -0.6 / -0.1 /  0.1 /  -0.2   |

Most of the gap comes from the wide feature models. On the 12 `binary_config`
files (88 to 1,044 columns, K of 8 to 10) ezr's features are never among
the best on LightGBM and cost it 23 win units on average; LIME and SHAP cost
under 5, random costs 49. A 50-label tree sees only a handful of splits, so MDI
has nothing to say about most of hundreds of columns, and its tail is noise.
The worst single cases are SS-P (-40.8 for ezr against 0.0 for LIME and SHAP)
and the FM/FFM feature models.

The `hpo` family says nothing: every method, random included, is within a
fraction of a win unit of the full model. Its 35 datasets inflate every win count.

## What not to trust

* **The ezr-on-ezr cell is not a result.** Its median is 0.0 on all 128 datasets.
  When a tree uses K or fewer features, MDI's top K contains all of them, and the
  retrained tree is the same tree. K comes from ezr's own trees, so that is the
  usual case. `e_ezr` is constant 0 and `top()` counts it as best whenever the
  others are at or below 0, which is where its 98 "wins" come from. Compare ezr
  only through `l_ezr`, and compare LIME and SHAP through `e_lime`, `e_shap`.
* **The score is coarse.** The best of 5 picks from half the data often lands
  on the same good row whatever the features, so 75 datasets have all four
  medians at exactly 0 on the ezr learner and 50 on LightGBM. `top()` then ties
  everything, and random is "best" on 71 to 76 datasets. Read the wins only on
  the 57-dataset subset where random is separated. A finer score (the mean of the
  top 5, or `-C 1`) would separate more datasets; it has not been run.
* **K = n on 22 datasets** (11 Health files, 8 SS files, 3 wc files). Every
  method keeps every feature and every change is 0. They are in the "all
  datasets" rows only.
* **Ceiling.** Full LightGBM already scores 100 on 21 datasets and the full tree
  on 17, so a reduced model can tie but never improve there.
* **K favours ezr.** K is how many features ezr's trees need, not how many
  LightGBM needs. ezr's features still lose; a K chosen by LightGBM might widen
  the gap or close it.
* **The labels were chosen with all features.** `likely()` picks rows by
  distance over every x column before any explanation exists, so the reduced
  models train on rows chosen with full-feature knowledge.
* **Unscorable datasets** from C2 (coc1000, pom3, PostgreSQL, Medical_Data,
  A2C_Acrobot, redis) need no oracle here because every score is a lookup, but
  their y is not a function of x, so their rows are noise: random is sole best on
  pom3c, and LightGBM on coc1000 GAINS 24.5 from ezr's features against a full
  score of only 24. Leave them out of any claim.
