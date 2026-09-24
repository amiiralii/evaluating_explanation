# Local / C7: is an explanation stable for similar inputs?

`sh c7/sweep.sh 12` over 128 MOOT datasets, 20 repeats, 20 random test points
per repeat, Budget 50 labels, Top 3 features, Eps 10 win units, Near 20,
`PYTHONHASHSEED=0`. About 7 minutes wall clock. Final report in
`c7_results.csv`, the same plus diagnostics in `c7_results_full.csv`, the
best-method counts per subset and family in `c7_tally.csv`, and worked examples
of one repeat in `c7_trace_SS-N.txt` (informative) and `c7_trace_auto93.txt`.

## What is measured

1. Each repeat buys 50 labels with `likely()` and fits an ezr tree and a
   LightGBM on them. The tree's leaves hold disty on the full data's scale, so
   both models' predictions are in the same win units.
2. For each random test point, the **neighbour** is the nearest REAL row with
   different x whose prediction moves at most 10 win units under BOTH models,
   searched among the 20 nearest rows with different x. A real row cannot be an
   impossible configuration, and the closeness condition is the README's
   "considering the predictions still being close": when the answer changes a
   lot, a different explanation is correct, not unstable. `found%` is how often
   such a neighbour existed.
3. Each explanation is cut to its top 3 features: **ezr** the first 3 on the
   tree path, root first; **lime** and **shap** the 3 largest |weights| of
   LightGBM; **rand** 3 at random, drawn afresh each time.
4. **Score**: Jaccard overlap of the point's set and the neighbour's set, 1 is
   identical, 0 is nothing shared. rand is the overlap two unrelated sets reach
   by chance.

Diagnostics in the full csv:

* `leaf_pct`: how often the neighbour fell in the SAME ezr leaf;
* `lime_self`: overlap of two lime runs on the SAME point, the README's "does
  the same input produce the same explanation";
* `x_*`: the four overlaps on only the pairs in DIFFERENT ezr leaves, pooled
  over all repeats. Not tested by `top()`, because such pairs are few.

## Results

The 118 datasets with more than 3 x columns; on the other 10 every top-3 set is
the whole feature set and all methods score 1.00.

| measure                                   | ezr  | lime | shap | rand |
|-------------------------------------------|-----:|-----:|-----:|-----:|
| among the most stable (datasets)          |  116 |   15 |    8 |    0 |
| the only most stable (datasets)           |   96 |    2 |    0 |    0 |
| median overlap, all pairs                 | 1.00 | 0.84 | 0.89 | 0.18 |
| median overlap, pairs in different leaves | 0.80 | 0.82 | 0.84 | 0.18 |
| median overlap, same point twice          | 1.00*| 0.84 | 1.00*|  n/a |

\* ezr and shap are deterministic, so the same point always gets the same set.

**Headline.** Ezr's explanations are the most stable on 116 of 118 datasets,
but mostly by construction. The neighbour lands in the same leaf in a median
90% of pairs, and there ezr gives the same path by definition. A tree's
explanation is piecewise constant, so it is stable wherever the prediction is
stable. That is a real property of the explainer, and the one C7 rewards, but
it is structural rather than earned.

Where ezr's paths actually diverge, it is slightly LESS stable than the other
two: 0.80 against 0.82 for lime and 0.84 for shap, and ezr matches or beats lime
on only 48 of 118 datasets and shap on 50.

**Lime's instability is its own noise.** Lime run twice on the same point
agrees 0.84, exactly as much as lime on a point and its neighbour. The two
differ by 0.05 or less on 94 of 118 datasets. So the neighbour costs lime almost
nothing; its sampling does. Lime fails the first half of C7 (same input, same
explanation) before the second half is even asked. Shap is deterministic and
is the more stable of the two on 65 datasets against 50.

All three are far above the random control (0.18), so every method's top
features carry information about the region, not noise.

## Split by family

Medians over the 118 datasets with more than 3 x columns.

| family         |  n | ezr  | lime | shap | rand | leaf% | lime self | x-ezr | x-lime | x-shap |
|----------------|---:|-----:|-----:|-----:|-----:|------:|----------:|------:|-------:|-------:|
| binary_config  | 12 | 0.99 | 1.00 | 0.89 | 0.01 |    88 |      0.94 |  0.80 |   0.94 |   0.89 |
| config         | 25 | 1.00 | 0.89 | 0.91 | 0.18 |    89 |      0.87 |  0.85 |   0.87 |   0.89 |
| hpo            | 35 | 1.00 | 0.80 | 0.88 | 0.46 |    94 |      0.80 |  0.78 |   0.77 |   0.83 |
| process        | 10 | 0.98 | 0.84 | 0.68 | 0.09 |    92 |      0.86 |  0.67 |   0.81 |   0.51 |
| systems        | 12 | 1.00 | 0.89 | 0.95 | 0.15 |    86 |      0.86 |  0.90 |   0.84 |   0.93 |
| sales_data     |  5 | 1.00 | 0.68 | 1.00 | 0.10 |    87 |      0.72 |  0.89 |   0.63 |   0.97 |

The ranking of lime and shap flips by family. On the wide `binary_config`
feature models lime is the steadier (1.00, and 0.94 across leaves) and is tied
with ezr on 8 of 12; on `process` shap is the weakest method at 0.68; on
`sales_data` shap is perfect and lime is worst. There is no single winner
between the two attribution methods.

## What not to trust

* **Ezr's lead is structural.** See the headline: in about 9 pairs in 10 its
  explanation cannot change. Compare it with lime and shap through the `x_*`
  columns, which are pooled and untested.
* **10 datasets are trivial.** SS-A to SS-G and the three `wc` files have 3 or
  fewer x columns, so with Top=3 every explanation is the whole feature set.
  Every method, random included, scores 1.00. They are in `c7_results.csv` but
  excluded above.
* **"Similar" is not controlled.** The neighbour is the nearest real row, so how
  far it is depends on how dense the table is. In the SS-N trace one neighbour
  differs in 12 features. Distance is not recorded; a fixed-size synthetic
  perturbation would control it but would create impossible rows.
* **The pairs are conditioned on stable predictions.** Only neighbours whose
  predictions stay within 10 win units count, which selects regions where the
  models are flat. On noisy data few such pairs exist: PostgreSQL found one for
  23% of points, player_statistics 71%, Medical_Data 76%. Eps and Near were not
  varied.
* **Only feature identity is compared.** Signs, magnitudes and ezr's ranges are
  ignored. `region_similarity` in `tools/ezr.py` compares ranges, but it has no
  lime or shap counterpart, so it is not used. "Top" also means different things:
  depth order for ezr, magnitude for lime and shap.
* **Unscorable datasets** from C2 (coc1000, pom3, PostgreSQL, Medical_Data,
  redis and friends) have y that x does not determine. C7 needs no oracle, but
  their models are fit to noise, so read their rows with care.
