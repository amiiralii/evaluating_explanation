# Experiment 3 / C7: direct-perturbation stability

Completed October 5, 2026 on all 128 CSV datasets under `data/optimize/`.
The five-column report is `experiment3_results.csv`. Every dataset produced a row;
all scores are finite and between 0 and 1. `sweep.err` is empty. The `auto93`
row matches the previously saved single-dataset result.

## Reproduce

From the repository root:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    sh experiment3/sweep.sh 10 data/optimize
```

The runner sets `PYTHONHASHSEED=0`. Each dataset uses 20 seeded 50/50 splits,
20 sampled test points per split, up to 50 bought training labels, and 1,000
samples per LIME explanation. One feature changes directly: a numeric step is
nominally 5% of its observed range, rounded for integers and clipped to bounds;
a categorical value switches to another observed category. Explanations retain
the top three nonzero weights by magnitude.

The primary score compares signed explanation weights before and after the
perturbation, only on pairs whose predictions differ by at most 5% under BOTH
EZR and LightGBM. Prediction closeness uses symmetric percentage difference.
Each dataset score is the median of its repeat means. `best` compares EZR,
LIME and SHAP using `tools/stats.top()`. Scores are rounded to three decimals.

## Summary

| Method | Median dataset score | Among statistically best | Sole best |
|---|---:|---:|---:|
| EZR | 1.000 | 123 | 83 |
| LIME | 0.926 | 6 | 2 |
| SHAP | 0.981 | 42 | 2 |

Best-method counts include ties, so they need not sum to 128. The median dataset
score gives each dataset equal weight; it is not a pooled point-level score.
All 128 datasets are included in this summary, without selecting a subset.

## What not to trust

- These measure explanation stability under the chosen perturbations, not model
  accuracy, explanation faithfulness, or whether advice improves a real outcome.
- EZR's path and derived weights are constant within a tree leaf. Its high score
  can therefore reflect that structure. The compact report does not retain
  per-dataset same-leaf coverage or cross-leaf scores, so this report alone
  cannot establish stability across tree boundaries.
- Filtering for close predictions selects flat regions. The compact report
  does not retain close-pair counts or repeat coverage; every score is finite,
  but datasets need not contribute equally many eligible pairs.
- Direct perturbations can produce infeasible or unobserved configurations.
  Integer steps and categorical switches may exceed a small numeric change.
- Top-three selection, step size and the 5% prediction tolerance are fixed
  settings here. Their sensitivity has not been evaluated.
- Random and same-input diagnostics are computed internally but are absent
  from the requested compact report. It cannot establish performance relative
  to Random or separate LIME sampling noise from perturbation effects.
- Empty explanations can score 1, and a rounded median of 1.000 does not imply
  every explanation pair was identical. The detailed trace shows these metrics
  and scoring conventions for a worked example, not an aggregate validation.
