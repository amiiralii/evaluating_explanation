import sys, random, os, itertools
from pathlib import Path
from timeit import repeat

from sympy.multipledispatch.conflict import consistent
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from tools.ezr import (Data, csv, clone, adds, disty,
                       likely, Tree, treeLeaf, trace,
                       region_similarity, treeShow,
                       the)
REPEATS = 20

def find_csvs(folder):
    """Yield every .csv under `folder`, recursively, sorted for stable output."""
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(root, f))
    out.sort()
    return out

def ezr_consistency_measure(file):
    all_data = Data(csv(file))
    half = int(len(all_data.rows) * 0.5)

    ## EZR
    ys  = [disty(all_data, r) for r in all_data.rows]
    b4  = adds(ys)
    win = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))

    cfg = dict(Budget=50, acq="near", leaf=3, Impurity="gini")
    for k, v in cfg.items(): setattr(the, k, v)
    all_best_paths = []
    errors = []
    for seed in range(REPEATS):
        the.seed = seed
        random.seed(seed)
        shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
        test  = clone(all_data, shuffled_rows[:half])
        train = clone(all_data, shuffled_rows[half:]) 
        labels = likely(train)
        tree = Tree(clone(train, labels))
        pick = min( [p for p in sorted(test.rows, key=lambda r: treeLeaf(tree, r).mu)[:the.Check]]
                , key = lambda r:disty(all_data,r))
        all_best_paths.append(trace(all_data, tree, pick))
        best = min( test.rows, key=lambda r: disty(all_data, r))
        errors.append( win(disty(all_data, best)) - win(disty(all_data,pick)))
    # print("EZR")
    # treeShow(all_data, tree)
    # print(errors)
    # print([round(v,2) for v in sorted(similarity)])
    # print(len(similarity), sorted(similarity)[10], round(sum(similarity)/len(similarity),2))
    return win(disty(all_data,pick))


def main():
    folder   = sys.argv[1] if len(sys.argv) > 2 else "data/optimize/"
    csv_paths = find_csvs(folder)
    print("Data, ezr-xplain, ezr-error, branch-explain, branch-error")
    for path in csv_paths:
        ezr_xplain, ezr_error = ezr_consistency_measure(path)


if __name__ == "__main__":
    main()