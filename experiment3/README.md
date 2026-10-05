# Experiment 3 / C7: stability under direct perturbations

Explain a random test point, change one feature directly, explain it again,
and measure whether the features, signs and weights remain stable. Compare
**EZR + its tree path**, **LightGBM + LIME**, **LightGBM + SHAP**, and **Random**.
This measures explanation stability, not model accuracy or the benefit of advice.
The earlier `c7/` experiment is preserved.

## Run

From the repository root:

```sh
sh experiment3/run.sh                          # auto93 only, full default repeats
sh experiment3/run.sh -R 2 -P 3 -L 200          # short smoke run
sh experiment3/run.sh -f data/optimize/config/Apache_AllMeasurements.csv
PYTHONHASHSEED=0 python3 experiment3/trace.py -P 3 -L 200
python3 experiment3/check.py
```

A captured walkthrough is in `results/c7_trace_auto93.txt`: one repeat (seed 0),
five test points, and 1,000 LIME samples per explanation. It shows the split,
examples of the bought labels, fitted models, actual feature changes, both
predictions, full EZR path contributions, per-feature before/after weights,
same-input sampling noise, scoring arithmetic, and the repeat summary. Its first
lines contain the exact reproduction command. The trace uses print-only
callbacks inside the experiment's `run()`; it does not recreate the experiment
or draw extra explanations. Use `-s` to trace a different seed.

Requires the existing environment's `numpy`, `lightgbm`, `shap`, `lime`, and
`sympy` (imported by `tools/ezr.py`). Set `PYTHON=/path/to/python` for the runners
to use another environment. Runner paths work from any working directory.

The full sweep is prepared but **has not been run**:

```sh
sh experiment3/sweep.sh 10 data/optimize
```

It writes `experiment3/results/c7_results.csv` (primary scores and best methods),
`c7_results_full.csv` (all metrics and diagnostics), and `sweep.err`. Like the
existing runners, this overwrites that experiment's previous reports. It pins
`PYTHONHASHSEED=0`, sorts dataset output, and fails if a dataset is missing.
The experiment supports `-c 1` for one CSV row and `--header` for its header.

## Protocol

Each of 20 repeats uses its seed for a 50/50 train/test split. EZR's active
learner buys up to 50 labels from the training half. Both predictors fit those
same labels, targeting distance-to-heaven on the **same full-dataset scale**.
Background feature statistics may use unlabelled training rows; no test labels
train either model. As in the existing experiments, objective normalization uses
full-dataset bounds. Each repeat samples 20 held-out points.

A separate seeded random stream chooses perturbations so explanation sampling
does not affect which points/features are changed. For each point:

- Change one nonmissing feature with variation. Numeric values move by
  `Step=0.05` of their observed range, clipped to bounds. Integer values retain
  their type and move at least one unit. A boundary uses the other direction if
  the first direction does not change the value.
- Categorical values switch to a different observed category. Missing values,
  constants and objectives remain unchanged. A point with no changeable feature
  is skipped, with coverage reported.
- Predict both points and extract all four explanations. Explain the original
  point a second time to measure stochastic variation on identical input.
- Retain each explanation's `Top=3` largest absolute nonzero feature weights.
  `-T 0` retains all nonzero weights. Zero-weight features are omitted, rather
  than manufacturing explanations from arbitrary ties.

## Prediction closeness

For predictions `p` and `q`, report the symmetric percentage difference:

```text
prediction difference (%) = 200 * abs(p-q) / (abs(p)+abs(q))
```

Both zero gives 0%; one zero gives 200%. For example, 0.40 and 0.42 differ by
4.88%. Opposite-sign predictions differ by 200%. This uses raw model predictions,
not win units, and does not silently add a denominator offset near zero.

`ezr_pred_pct` and `lgbm_pred_pct` are mean percentage changes over all changed
pairs. `ezr_close_pct` and `lgbm_close_pct` report the percentage within `Eps=5`.
The primary stability scores use only pairs within this tolerance under **both**
predictors, giving all four methods the same comparison pairs. `close_pct`,
`close_pairs` and `close_repeats` show their coverage. `all_*` reports stability
without filtering. Use `-E` to change the tolerance; it is an experiment setting,
not an empirically established adequacy threshold.

## Explanation weights

- **EZR:** a split's signed contribution is `child.mu - parent.mu`. Contributions
  are summed when a feature occurs repeatedly on the path. Before truncation,
  their sum is `leaf.mu - root.mu`. Negative values lower distance-to-heaven.
  These are derived path contributions, not native EZR weights or SHAP values.
  The full native predicate set is retained for a separate rule-overlap metric.
- **LIME:** signed local regression coefficients from its undiscretized,
  standardized numeric features and categorical membership indicators. Their
  signs describe local surrogate coefficients.
- **SHAP:** signed TreeSHAP contributions to LightGBM's prediction relative to
  its expected prediction. Negative values lower distance-to-heaven.
- **Random:** fresh independent uniform signed weights over the feature space,
  truncated the same way. Its predictor for closeness is LightGBM.

Keep the same model and background for both points. LIME advances its sampling
state normally; the same-point diagnostic measures that noise instead of hiding
it by resetting the state. Random also draws afresh. Absolute weights are never
compared across methods: LIME's coefficients, SHAP's contributions, and EZR's
path changes have different meanings. We compare each method with itself.

## Metrics

Let `a` and `b` be a method's signed feature-weight dictionaries before and after
the perturbation. Use the union of their features; an absent feature has weight
zero. Let `mass = sum(abs(a[j]) + abs(b[j]))`.

| Column | Measurement, 0 to 1 (higher is more stable) |
|---|---|
| `ezr/lime/shap/rand` | Signed stability: `1 - sum(abs(a[j]-b[j])) / mass` on close pairs |
| `features_*` | Jaccard overlap of nonzero feature identities on close pairs |
| `signs_*` | Fraction of shared nonzero features retaining their sign; NaN if none shared |
| `weights_*` | Magnitude stability: `1 - sum(abs(abs(a[j])-abs(b[j]))) / mass` on close pairs |
| `all_*` | Signed stability over every perturbed pair |
| `self_*` | Signed stability for two explanations of the identical original point |
| `cross_*` | Signed stability for close pairs in different EZR leaves, for every method |

An identical explanation scores 1. Flipping every sign while keeping magnitudes
scores 0 for signed stability, 1 for magnitude stability and feature overlap,
and 0 for sign agreement. Doubling every weight scores 2/3 for signed and
magnitude stability. Thus magnitude changes are measured without normalizing
each vector separately and erasing changes in scale.

Two empty explanations have signed/magnitude stability and feature overlap 1,
but their sign agreement is NaN. `empty_*_pct` reports how often both explanations
are empty for each method, so a constant model's apparent stability is visible.

Each repeat is reduced to a mean per metric; the report shows the median of
those repeat means. Missing measurements stay NaN and never become zero.
`tools/stats.top()` marks statistically best primary scores with `+` (or lists
names in CSV). At least two contributing repeats per method are required before
calling it; a one-repeat smoke run receives no best-method designation.
Coverage diagnostics are pooled counts/means, not inputs to the statistical test.

Additional diagnostics: `pairs` counts actual changed pairs; `changed_pct` is
coverage over attempted points; `numeric_step_pct` is the mean actual numeric
change relative to that feature's range; `categorical_pct` is the fraction of
categorical changes; `same_leaf_pct` is the fraction of pairs sharing an EZR leaf;
`rules` is Jaccard overlap of full native EZR predicates on close pairs.

## What not to trust

- An integer step, binary flip or categorical switch may be a substantial change.
  Check actual step size and category share rather than assuming every pair is
  geometrically equally close. Only one feature changes, even on wide tables.
- Direct perturbations may produce infeasible or unseen configurations. This
  experiment measures the predictors' explanations at those inputs; it does not
  claim their real outcomes or feasibility. `tools/oracle_check.py` is needed
  when evaluating true/estimated outcome effectiveness, not for this model
  explanation stability measurement.
- EZR is piecewise constant: within a leaf both its weights and rules stay the
  same by construction. Use `same_leaf_pct` and `cross_*` to understand its lead.
  Few cross-leaf pairs mean little evidence about stability across boundaries.
- Stable, empty or uninformative explanations can score highly. Stability alone
  cannot establish faithfulness, usefulness or predictive accuracy.
- Filtering on close predictions selects flat regions. Read `all_*` and coverage
  with the primary score. Large percentages near zero predictions are deliberate.
- Top-three selection can change at near-tied weights. `-T 0` tests full weights;
  no sensitivity sweep has yet been run. LIME noise is exposed by `self_lime`.
- `rules` measures exact predicates, not interval overlap. Equal derived weights
  can conceal different tree constraints, so inspect `rules` or `trace.py` too.
- The inherited active learner can stop below Budget on small datasets. All-
  missing symbolic columns are rejected explicitly because the existing learner
  cannot compute their mode. Numeric objectives must be present, as in EZR.

## Validation status

Targeted checks cover reversed signs, changed magnitudes, disjoint features,
zero predictions, missing/constant features, integer bounds, preservation of
objectives, path contribution sums, and feature mapping when an objective occurs
between feature columns. Smoke runs cover continuous/missing (`auto93`),
categorical (`Apache_AllMeasurements`) and wide (`FFM-1000-200-0.50-SAT-1`) data.
They are execution checks, not experimental findings. Full results are pending.
