"""
rev_perm_retrain.py - Step 2(c): full-retraining trial-block permutation null for LR and RF (F0, F1).

Permutation b moves whole 3-zone label vectors between the 16 trials using
np.random.default_rng([0, b]) (identical permutations for every model/config -> comparable nulls),
then reruns the complete LOTO (16 folds x 5 seeds, unchanged models) and records the pooled AUC
of the seed-averaged out-of-fold scores.

    python rev_perm_retrain.py run --models logreg rf --configs F0 F1 --perms 500 --workers 10
    python rev_perm_retrain.py summarise

Rows are appended to revision_results/perm_retrain_draws.csv as they finish (resumable);
`summarise` writes the 2(c) rows into permutation_cluster.csv (retrained=True).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

import rev_common as C

DRAWS = C.REV / "perm_retrain_draws.csv"


def permuted_labels(graphs, b):
    rng = np.random.default_rng([C.STAT_SEED, b])
    perm = rng.permutation(len(graphs))
    return {g["trial_id"]: graphs[perm[i]]["y"].clone() for i, g in enumerate(graphs)}


def one(args):
    model, cfg, b, seeds = args
    base = [C.select(g, C.CONFIGS[cfg][0]) for g in C.load_graphs()]
    graphs = base if b < 0 else C.with_labels(base, permuted_labels(base, b))
    preds = []
    for s in seeds:
        preds += C.run_loto(model, graphs, s, cfg)
    z = C.zones(pd.DataFrame(preds))
    return dict(model=model, config=cfg, perm=b, n_seeds=len(seeds),
                auc=C.auc(z.y.to_numpy(int), z.probability.to_numpy()))


def run(a):
    done = set()
    if DRAWS.exists():
        d = pd.read_csv(DRAWS)
        done = set(zip(d.model, d.config, d.perm))
    seeds = {m: list(C.SEEDS) for m in a.models}
    if "logreg" in a.models and a.lr_one_seed:
        seeds["logreg"] = [C.SEEDS[0]]    # lbfgs LR has no randomness: verified seed-invariant below
    jobs = [(m, c, b, seeds[m]) for b in range(-1, a.perms) for m in a.models for c in a.configs
            if (m, c, b) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            pd.DataFrame([r]).to_csv(DRAWS, mode="a", header=not DRAWS.exists(), index=False)
            if n % 50 == 0:
                print(n, flush=True)


def summarise():
    d = pd.read_csv(DRAWS)
    rows = []
    for (m, c), g in d.groupby(["model", "config"]):
        obs = g[g.perm == -1].auc
        null = g[g.perm >= 0].auc.to_numpy()
        if obs.empty or len(null) == 0:
            continue
        a = float(obs.iloc[0]); n = len(null)
        rows.append(dict(model=m, config=c, score_source="this run (retrained under each permutation)",
                         null="trial_block_retrained", retrained=True, n_permutations=n, rng_seed="[0, b]",
                         auc=a, null_mean=null.mean(), null_q025=np.quantile(null, .025),
                         null_q975=np.quantile(null, .975),
                         p_two_sided=(1 + np.sum(np.abs(null - .5) >= abs(a - .5) - 1e-12)) / (n + 1),
                         p_one_sided=(1 + np.sum(null >= a - 1e-12)) / (n + 1),
                         n_seeds_per_permutation=int(g.n_seeds.iloc[0])))
    pc = C.REV / "permutation_cluster.csv"
    old = pd.read_csv(pc) if pc.exists() else pd.DataFrame()
    if len(old):
        old = old[old.retrained != True]  # noqa: E712
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(pc, index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarise", "check_lr"])
    ap.add_argument("--models", nargs="+", default=["logreg", "rf"])
    ap.add_argument("--configs", nargs="+", default=["F0", "F1"])
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--lr_one_seed", action="store_true")
    a = ap.parse_args()
    if a.action == "run":
        run(a)
    elif a.action == "check_lr":
        p = C.load_preds("logreg", "F0")
        w = p.pivot_table(index=["trial_id", "zone_id"], columns="seed", values="probability")
        print("LR max spread across seeds:", float((w.max(1) - w.min(1)).max()))
    else:
        summarise()
