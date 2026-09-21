# Local / C2: do one-feature counterfactuals actually help?

`python3 c2_local_counterfactual.py -f data/optimize/` over 128 MOOT datasets,
20 repeats, 20 random test points per repeat, Budget 50 labels, step 1 sd.
Full table in `c2_results.txt`, a worked example of one repeat in `c2_trace_auto93.txt`.

## How a suggestion is scored

Each method changes ONE feature of a random test point, then the changed row is
scored against the truth, never against the model that proposed the change.

* the starting point is a real row, so its score is looked up exactly;
* if the changed row is also in the data, its score is looked up exactly;
* otherwise a random forest, fit on every row and every label, predicts it.

Two report columns keep the reader honest. `exact%` is how often no estimate was
needed. `orc-r` is the forest's out-of-bag correlation with the real y, measured
per dataset, so a weak oracle is visible rather than silent.

### Why a forest, and not knn

The first version of this experiment scored a suggestion with the model that
proposed it, which let every method grade its own homework. The second used
weighted knn over the 5 nearest rows. Both were wrong, and knn was wrong in a
subtle way: it shrinks toward the local mean, so a good starting point always
looks like it got worse. Measured by leave-one-out on real rows, knn's regression
slope runs from 0.14 to 0.99, understating good rows by up to 52 win units and
overstating bad ones by up to 57.

Estimating BOTH sides with knn does not fix this. Tested on pairs of real rows
that differ in exactly one feature, where the true change is known:

| dataset | knn mixed err / r | knn both sides err / r | forest err / r |
|---------|------------------:|-----------------------:|---------------:|
| LLVM    |     15.5 / 0.79   |          17.3 / 0.67   |   7.0 / 0.96   |
| Apache  |     24.8 / 0.88   |          28.1 / 0.83   |  10.7 / 0.96   |
| SS-H    |     18.0 / 0.82   |          11.4 / 0.81   |   0.2 / 1.00   |
| SS-M    |      2.6 / 0.70   |           4.6 / -0.06  |   2.0 / 0.77   |
| auto93  |     11.5 / 0.68   |          16.0 / 0.32   |  10.2 / 0.77   |

The baseline is exactly known, so replacing it with an estimate only adds noise
that does not cancel. The real problem was estimator quality: 5-nn scores r=-0.07
on SS-N where the forest scores 0.98. `-O knn` still runs the old way, and
`-J model` still runs the original self-graded score, which measures faithfulness
to one's own model (C4) rather than effectiveness (C2).

## Results

| group                              |  n  | ezr | lime | shap | rand |
|------------------------------------|----:|----:|-----:|-----:|-----:|
| all datasets, wins                 | 128 |  91 |   48 |   56 |    3 |
| all datasets, median               |     |11.9 | 16.1 | 17.2 | -0.2 |
| all datasets, mean                 |     |26.1 | 23.8 | 23.6 | -1.0 |
| trustworthy oracle (orc-r >= 0.9)  |  79 |  48 |   30 |   36 |    0 |
| trustworthy, median                |     |33.6 | 27.0 | 28.1 | -0.4 |
| pure lookup (exact% >= 80)         |  13 |   2 |    8 |   12 |    0 |
| pure lookup, median                |     |15.0 | 33.8 | 40.9 | -0.1 |

Every method beats the random control everywhere. Ezr is sole winner on 65
datasets against 9 for shap and 3 for lime, yet ezr has the best MEAN and the
worst MEDIAN of the three: it wins big on some datasets and slightly on the rest.

## The split that matters

The 13 datasets where suggestions land on real rows, so no estimate is involved
at all, reverse the headline. Shap wins 12 of them and ezr wins 2, with medians
of 40.9 against 15.0. These are the complete configuration tables: LLVM, deeparch,
Apache, SS-L, SS-P, SS-W, the FFM and FM feature models, billing10k. On the 68
datasets with a trustworthy oracle but no exact lookups, ezr wins 46 to 24 and 26.

So this is a dataset-family effect, not an artifact of the oracle: where the space
is a complete discrete configuration table, attribution over a boosted model gives
better one-feature advice than a shallow tree path built from 50 labels. That
result survived every change of judge in this experiment, so it is the most
robust thing here.

## What not to trust

24 datasets have orc-r below 0.7, meaning nothing can score them reliably. The
worst are coc1000 at 0.05, dress-up at 0.21, Medical_Data at 0.23, pom3d at 0.31,
PostgreSQL at 0.39 and nasa93dem at 0.41. Note PostgreSQL sits in the pure-lookup
group with a bad oracle, so its numbers come mostly from real lookups and are
still usable; the others are not. Their y values are not a function of the
recorded x columns, so drop them from any headline claim.

---

# Appendix: is the forest oracle trustworthy?

`python3 c2_oracle_check.py -f data/optimize/`  (full table in `oracle_check.txt`)

Hide 100 random rows, capped at a third of the file. Fit each oracle on what is
left, ask it for the disty of each hidden row, compare with the truth. A third
competitor always predicts the mean of the kept rows, so `skill` = 1 - mae/mae(mean)
reads 0 for "no better than guessing the average" and 1 for perfect.

| oracle | median mae | median r | median skill | skill > 0 |
|--------|-----------:|---------:|-------------:|----------:|
| forest |        5.5 |     0.97 |         0.84 |   124/128 |
| knn    |       12.1 |     0.87 |         0.58 |   122/128 |
| mean   |       31.7 |     0.00 |         0.00 |     0/128 |

The forest beats knn on 113 of 128 datasets. Its typical error is 14% of the
spread of the values it is predicting. On the 94 datasets where it has real skill,
median error is 3.9 win units against a median spread of 46.2.

Caveat on what this proves: a hidden row is still a REAL combination of feature
values. A made-up row can be a combination that occurs nowhere, where any oracle
extrapolates. So these numbers bound oracle accuracy from above.

## The orc-r column does its job

`orc-r`, the out-of-bag correlation printed in `c2_results.txt`, correlates 0.94 with the
held-out skill measured here. No dataset with orc-r >= 0.9 turned out to have poor
held-out skill. Of the 20 datasets orc-r flags below 0.7, 15 do indeed have skill
at or below 0.3. So the cheap self-report in the main table can be trusted to
identify which rows to discard.

## The 11 datasets no oracle can score

| dataset | spread | forest mae | forest r | forest skill | knn skill |
|---------|-------:|-----------:|---------:|-------------:|----------:|
| coc1000        |  35.8 |  29.3 | -0.01 | -0.02 | -0.08 |
| pom3d          |  49.1 |  43.2 |  0.19 | -0.02 | -0.07 |
| PostgreSQL     | 172.3 | 106.2 |  0.32 | -0.00 | -0.06 |
| pom3a          |  31.1 |  23.6 |  0.41 |  0.05 |  0.02 |
| pom3b          |  29.3 |  21.9 |  0.45 |  0.05 |  0.01 |
| Medical_Data   | 107.3 | 101.5 |  0.18 |  0.06 |  0.05 |
| pom3c          |  25.0 |  19.9 |  0.40 |  0.06 |  0.03 |
| Health-CPRs0006|  14.9 |   4.3 |  0.66 |  0.07 |  0.09 |
| A2C_Acrobot    |  38.0 |  27.2 |  0.42 |  0.16 |  0.18 |
| redis          |  20.9 |  12.5 |  0.45 |  0.20 |  0.26 |
| dress-up       |   0.9 |   0.6 |  0.86 | -0.70 |  0.38 |

knn is no better on these: both oracles fail together, so this is a property of
the data, not of the forest. In these files the recorded x columns do not
determine y. The pom3 family are stochastic project simulators, and coc1000 is
a sampled effort model, so repeated x really can give different y.

dress-up is a different case and not a real failure: its true values span only
0.9 win units, so the mean control is already almost exact and no method can
improve on it. An mae of 0.6 on a spread of 0.9 is harmless. Read skill as
meaningless whenever spread is near zero.
