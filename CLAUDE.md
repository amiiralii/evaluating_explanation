# Working guide: evaluating XAI methods for SBSE

Read this before writing code in this repo. It is written for a session that has
not seen the earlier work.

## The goal

`README.md` lists nine things an explanation should do for search-based software
engineering (C1 attribution ... C9 interactivity) and sketches one experiment per
criterion. The job is to turn each sketch into a real experiment that ranks XAI
methods against each other on the MOOT datasets. One script per experiment.

The baseline explainer is the decision path of an `ezr` regression tree. The
comparison methods are LIME and SHAP over a LightGBM regressor. Every experiment
also needs a random control; without one you cannot tell skill from noise.

## The data

140 csv files under `data/optimize/`, in 12 category folders. MOOT column
conventions, enforced by `tools/ezr.py`:

* an UPPERCASE first letter means numeric, lowercase means symbolic;
* a trailing `+` or `-` makes the column an objective to maximize or minimize;
* a trailing `X` means ignore the column;
* `?` is a missing value;
* `disty(data,row)` is distance to heaven, the objectives collapsed into one
  number where lower is better. This is the target for every model here.

Sizes run from 81 to 166,975 rows and from 3 to 1,044 x columns. Anything you
write must survive all-symbolic files, all-numeric files, missing values, and the
1,044-column file. Test on at least `misc/auto93.csv` (continuous, has `?`),
`config/Apache_AllMeasurements.csv` (all symbolic), and
`binary_config/FFM-1000-200-0.50-SAT-1.csv` (very wide).

## The tools

* `tools/ezr.py` (Tim Menzies, v0.5, locally patched): `Data`, `clone`, `adds`,
  `disty`, `likely` (active learner), `Tree`, `treeLeaf`, `treeShow`, `trace`,
  `summarize`, `region_similarity`, `main` (command-line parser), `the`
  (settings). `summarize` and `region_similarity` were added for the C7 stability
  experiment and are still unused; start there when you build C7.
* `tools/stats.py`: `top(dict_of_lists, reverse=True)` returns the keys that are
  statistically best, using Cliff's delta plus a Kolmogorov-Smirnov test. Use it.
  Do not hand-roll statistics.
* `example.py` is the original template. It does not run: it returns one value
  where the caller unpacks two, and it imports `sympy` for no reason. Read it for
  the intended shape, then write your own.

## House style: follow ezr

* two-space indent, one-line docstring on every function, terse;
* `SimpleNamespace as o` instead of classes;
* options live in the module docstring as `-X Name=default`, parsed with
  `the.__dict__.update({k: coerce(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})`
  and then `main(the, globals())`. No argparse;
* results print to stdout as one line per dataset. No csv writers, no plotting;
* minimal. If a feature is not needed to answer the question, leave it out.

## Experiment conventions

Established by `c2/c2_local_counterfactual.py` and worth keeping identical so
results are comparable across experiments:

* 20 repeats per dataset. Each repeat seeds both `the.seed` and `random.seed`,
  shuffles, and splits 50/50 into train and test;
* `likely(train)` buys 50 labels (`Budget`). Every model in the comparison trains
  on exactly those labels, so only the explanation differs;
* scores are reported in "win" units: `100*(1-(v-b4.lo)/(b4.mu-b4.lo))` where
  `b4` summarizes `disty` over all rows. 100 is the best row in the file, 0 is
  the average row. The scale is not clipped, so values above 100 are possible and
  legitimate;
* collapse each repeat to one number, then give `top()` the list of 20 repeat
  means. Repeats are the independent unit, not points;
* mark the statistically best methods with `+` in the output line.

## Scoring: the rules we learned the hard way

These cost several full rewrites. Do not relearn them.

1. **Never let a model grade its own output.** The first version of C2 scored a
   suggestion with the model that proposed it. That inflated ezr from 88 wins to
   98 and made it sole winner on 91 datasets instead of 60. If a method proposes a
   change, something independent must judge it.
2. **Prefer a lookup to an estimate.** The starting point is always a real row,
   so its score is known exactly. The changed row is often a real row too, and on
   complete configuration tables it nearly always is. Index rows by the tuple of
   their x values and look them up before estimating anything.
3. **When you must estimate, use a random forest fit on every row and every
   label.** It is the judge, never a learner, so it may see all the labels. It
   beat weighted 5-nn on 113 of 128 datasets: median error 5.5 win units versus
   12.1, median skill 0.84 versus 0.58 against a mean-predicting control.
4. **k-nn is a trap.** It shrinks toward the local mean, with regression slopes
   from 0.14 to 0.99, so a good starting point always looks like it got worse.
   Estimating both sides with k-nn to cancel the bias makes it worse still,
   because it replaces an exactly known baseline with noise. Both were measured.
5. **Report how much of the score was estimated.** Two columns: `exact%`, how
   often no estimate was needed, and `orc-r`, the forest's out-of-bag correlation
   with the truth. `orc-r` correlates 0.94 with a proper holdout test, so it is a
   trustworthy cheap warning.
6. **Eleven datasets cannot be scored by anything.** `coc1000`, the `pom3`
   family, `PostgreSQL`, `Medical_Data`, `A2C_Acrobot`, `redis` and friends have
   `orc-r` below 0.7 because their x columns do not determine y. Exclude them from
   headline claims. `tools/oracle_check.py` finds them.
7. **Do not invent impossible rows.** A one-standard-deviation numeric step
   produced an 8-cylinder car with 90cc displacement. No oracle can be right about
   a configuration that cannot exist. Prefer suggested values that actually occur
   in that column. This is a known open improvement, not yet implemented.
8. **Split results by dataset family before concluding.** In C2, ezr wins overall
   but shap wins 12 of the 13 datasets where every suggestion lands on a real row.
   Complete factorial configuration tables behave differently from sparse tables,
   and an aggregate hides it.

## Gotchas

* `the` flags are matched by first letter, so a new `Knn` option silently
  collides with ezr's existing `Ks`. Check `tools/ezr.py`'s docstring before
  naming one.
* a `LimeTabularExplainer` advances its own random state on every call, so two
  calls give different weights. Reset `explainer.random_state` if you need to
  print an explanation and then consume it.
* `treeSelects` returns True for a missing value, so `?` never triggers a
  suggestion on that feature.
* many datasets contain rows with identical x but different y. Average them; do
  not let one silently overwrite another.
* `Data(csv(f))` computes column bounds over every row, so `col.lo`, `col.hi` and
  `col.sd` are global statistics, not training statistics. That is deliberate:
  x values are free here, only labels cost.
* **ezr is not reproducible unless you pin `PYTHONHASHSEED`.** `_symCuts` in
  `tools/ezr.py:330` iterates `set(x for x, _ in xys)` to pick a symbolic cut, and
  Python randomizes string hashing per process, so ties between equally good cuts
  break differently on every run. On Telco the ezr column moved between 5.4, 3.3
  and 5.4 across three identical runs, and held at 5.3 with the seed pinned.
  `c2/sweep.sh` exports `PYTHONHASHSEED=0`. Do the same in any new experiment. The
  proper fix is `sorted(set(...))` in ezr, which nobody has applied yet because it
  changes tie-breaking and so shifts every published symbolic result.
* some repeats are degenerate. If the 50 bought labels all share one objective
  value, every model is constant and every method scores zero. Check the label
  spread before using a repeat as a talking example.

## Layout of an experiment

One directory per README criterion, named for it: `c1/`, `c3/`, `c5/`, and so on.
Shared machinery lives in `tools/`, never inside an experiment directory, and a
tool must never import an experiment. If two experiments need the same helper,
either give the tool its own copy or move the helper to `tools/`.

    c3/
      c3_<what_it_does>.py      the experiment                        required
      c3_trace.py               one dataset, one repeat, every step    optional
      sweep.sh                  run every dataset, rewrite the report  required
      results/
        c3_results.csv          the report: dataset, a column per method, best
        c3_results_full.csv     the same plus any diagnostic columns
        RESULTS.md              the write-up, including what not to trust
        c3_trace_<dataset>.txt  a captured walkthrough, if there is a trace
        oracle_check.txt        output of tools/oracle_check.py, if the
                                experiment estimates anything it cannot look up

The experiment script owes four things to the rest of the setup.

* **A `-c csv=1` flag** that prints ONE csv row per dataset and nothing else, no
  header and no tally. `sweep.sh` depends on it. Without it the sweep has to
  parse the aligned console output by character offset, which breaks silently the
  first time a number grows wider than its column.
* **A human readable default**, one aligned line per dataset with `+` marking the
  statistically best methods. Both formats come from the same `report()`.
* **This path preamble**, because the file now sits one level below the root:

      HERE = Path(__file__).resolve().parent
      sys.path[:0] = [str(HERE), str(HERE.parent)]   # own dir, then the repo root

* **A `csvs()` that falls back to the repo root** (`if not p.exists(): p =
  HERE.parent / path`), so a relative `-f data/optimize/...` works from any
  working directory, not only from the root.

For `sweep.sh`, copy `c2/sweep.sh` and change four things: `OUT`, the two result
filenames, and the script named inside the `xargs` line. It already handles what
is easy to forget: it `cd`s to the repo root, exports
`PYTHONHASHSEED=0`, sorts the interleaved output of the parallel shards so the
file is deterministic, splits the full csv into the report with `cut`, and warns
when a dataset produced no row instead of letting it vanish.

Sharding one process per dataset is only valid because each dataset's row is
self-contained: `top()` compares the methods within one dataset and never across
datasets. An experiment that needs cross-dataset statistics before printing
cannot be parallelized this way.

`RESULTS.md` carries the command that produced it, what the columns mean, the
headline table, and a section on what not to trust. A result with no caveats
section is not finished.

## What exists

* `c2/c2_local_counterfactual.py` — README experiment "Local / C2". Explain a random
  test point, change ONE feature on that explanation, score the changed row
  against the truth. `-J model` restores the old self-graded score, which measures
  faithfulness (C4) rather than effectiveness (C2).
* `c2/c2_trace.py` — the same experiment for one dataset and one repeat with every
  step printed: labels bought, tree grown, each explanation, the feature changed,
  the new row, the score. Use it for talks and for debugging a new experiment.
* `tools/oracle_check.py` — hides rows and asks each oracle to score them, so the
  scorer itself is validated rather than assumed. It lives in `tools/` because any
  experiment that estimates something needs it, and it is deliberately standalone:
  it carries its own copy of the encoding helpers rather than importing an
  experiment. Keep it that way.
* `c2/sweep.sh` — runs the experiment on every dataset and rewrites the report.
* `c2/results/` — `c2_results.csv` (the final report: dataset, one column per
  method, and which methods tied for best), `c2_results_full.csv` (the same
  plus exact% and orc-r), `RESULTS.md` (the write-up with its caveats),
  `oracle_check.txt` (oracle validation), and a captured walkthrough.

## Still to build

One script each, same conventions. The README has the sketches.

* **C1 + C4, global.** Take the top k features from each method, retrain on only
  those, compare accuracy.
* **C2 + C5, global and local.** Build an ideal point from the explanation alone,
  without looking at the best configuration, then score it. Tests whether the
  explanation conveys ranges, not just directions.
* **C3, local.** For multi-objective problems, does the explanation name which
  objective was most sacrificed?
* **C6, global.** Does the explanation exceed seven items?
* **C7, local.** Perturb a point slightly and check the explanation is stable
  while the prediction stays close. `summarize` and `region_similarity` in
  `tools/ezr.py` already exist for this.
* **C8, local.** Does the output signal its own confidence?
* **C9, local.** Does the output support what-if questions, and at what
  granularity: none, direction, or magnitude?

Note the asymmetry that keeps surfacing: a tree path carries a range, while LIME
and SHAP carry only a direction. C5 and C9 are where that difference should be
measured rather than worked around.

## Running

Run from the repo root. A relative `-f` path also resolves against the repo
root, so the scripts work from anywhere.

There is no parallel flag, deliberately. A full 128-dataset sweep takes about two
hours single-process. Shard it instead, one dataset per process, because each
dataset's line and its `+` markers are computed independently:

```sh
find data/optimize -name '*.csv' | sort \
  | xargs -P 10 -n 1 sh -c 'python3 c2/c2_local_counterfactual.py -f "$1" | sed -n 2p' _ \
  > lines.txt
```

That finishes in about ten minutes on 12 cores. Stitch the lines, add the header,
and tally the `+` marks for the summary row.

## How to work here

* measure before deciding. Every design choice above was settled by a small
  experiment, not by argument. When two approaches are plausible, test them on
  data where the answer is already known;
* validate the validator. If an experiment depends on an estimate, write the
  script that checks that estimate;
* report negative and awkward results. The most interesting finding in C2 is that
  the headline reverses on the subset where no estimation happens. Do not bury it;
* say what was not done. Scaling the work down is the researcher's call.
