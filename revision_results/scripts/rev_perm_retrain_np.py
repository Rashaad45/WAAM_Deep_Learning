"""
rev_perm_retrain_np.py - Step 2(c) without torch (memory-light): full LOTO retraining under trial-block
label permutations for LR and RF on the zone-mean feature matrix (revision_results/zone_mean_features.csv,
= src/run_experiments.zone_feature_matrix of every trial, written by rev_importance.py).

Model definitions are copied verbatim from src/run_experiments.classical_model. Permutation b uses
np.random.default_rng([0, b]) over the 16 trials (sorted trial order). perm = -1 is the observed labelling
and must reproduce the benchmark AUC (checked in summarise()).

    python rev_perm_retrain_np.py run --models logreg --configs F0 F1 --perms 500 --workers 1
    python rev_perm_retrain_np.py summarise      (-> permutation_cluster.csv rows with retrained=True)
"""
import argparse
import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REV = Path(__file__).resolve().parents[1]
DRAWS = REV / "perm_retrain_draws.csv"
SEEDS = (42, 43, 44, 45, 46)
STAT_SEED = 0


def classical_model(name, seed):        # verbatim from src/run_experiments.py
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    if name == "rf":
        return RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(name)


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def config_cols(cfg):
    sys.path.insert(0, str(Path(__file__).parent))
    # column lists duplicated from rev_common.CONFIGS (F0 = all 62, F1 = drop the 6 CI columns)
    allc = list(range(62))
    return {"F0": allc, "F1": [i for i in allc if i % 10 != 6 or i >= 60]}[cfg]


_D = None


def data():
    global _D
    if _D is None:
        d = pd.read_csv(REV / "zone_mean_features.csv")
        X = d.iloc[:, :62].to_numpy(np.float32)
        _D = (X, d.y.to_numpy(int), d.trial_id.to_numpy())
    return _D


def one(args):
    model, cfg, b, seeds = args
    X, y, tr = data()
    X = X[:, config_cols(cfg)]
    trials = list(dict.fromkeys(tr))
    rows = [np.flatnonzero(tr == t) for t in trials]
    if b >= 0:
        perm = np.random.default_rng([STAT_SEED, b]).permutation(len(trials))
        y = np.concatenate([y[rows[perm[i]]] for i in range(len(trials))])   # rows are contiguous, ordered
    probs = np.zeros((len(seeds), len(y)))
    for si, s in enumerate(seeds):
        for r in rows:
            m = np.setdiff1d(np.arange(len(y)), r)
            clf = classical_model(model, s).fit(X[m], y[m])
            probs[si, r] = clf.predict_proba(X[r])[:, 1]
            del clf                                  # free the fitted forest before the next fold
        gc.collect()
    return dict(model=model, config=cfg, perm=b, n_seeds=len(seeds), auc=auc(y, probs.mean(0)))


def run(a):
    done = set()
    if DRAWS.exists():
        d = pd.read_csv(DRAWS); done = set(zip(d.model, d.config, d.perm))
    seeds = {"logreg": [SEEDS[0]] if a.lr_one_seed else list(SEEDS), "rf": list(SEEDS)}
    jobs = [(m, c, b, seeds[m]) for b in range(-1, a.perms) for m in a.models for c in a.configs
            if (m, c, b) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    if a.workers <= 1:                               # single process, no worker pool; free memory per permutation
        for n, j in enumerate(jobs, 1):
            res = one(j)
            pd.DataFrame([res]).to_csv(DRAWS, mode="a", header=not DRAWS.exists(), index=False)
            del res
            gc.collect()
            if n % 10 == 0:
                print(n, flush=True)
        return
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            pd.DataFrame([f.result()]).to_csv(DRAWS, mode="a", header=not DRAWS.exists(), index=False)
            if n % 50 == 0:
                print(n, flush=True)


def summarise():
    d = pd.read_csv(DRAWS)
    expected = {("logreg", "F0"): 0.6921739130434783, ("rf", "F0"): 0.7426086956521739}
    rows = []
    for (m, c), g in d.groupby(["model", "config"]):
        obs = g[g.perm == -1].auc
        null = g[g.perm >= 0].auc.to_numpy()
        if obs.empty or not len(null):
            continue
        a = float(obs.iloc[0]); n = len(null)
        if (m, c) in expected:
            assert abs(a - expected[(m, c)]) < 1e-9, f"{m} {c}: observed {a} != benchmark {expected[(m, c)]}"
        rows.append(dict(model=m, config=c, score_source="this run (retrained under each permutation)",
                         null="trial_block_retrained", retrained=True, n_permutations=n, rng_seed="[0, b]",
                         auc=a, null_mean=null.mean(), null_q025=np.quantile(null, .025),
                         null_q975=np.quantile(null, .975),
                         p_two_sided=(1 + np.sum(np.abs(null - .5) >= abs(a - .5) - 1e-12)) / (n + 1),
                         p_one_sided=(1 + np.sum(null >= a - 1e-12)) / (n + 1),
                         n_seeds_per_permutation=int(g.n_seeds.iloc[0])))
    pc = REV / "permutation_cluster.csv"
    old = pd.read_csv(pc)
    old = old[old.retrained != True]  # noqa: E712
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(pc, index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarise"])
    ap.add_argument("--models", nargs="+", default=["logreg", "rf"])
    ap.add_argument("--configs", nargs="+", default=["F0", "F1"])
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lr_one_seed", action="store_true")
    a = ap.parse_args()
    run(a) if a.action == "run" else summarise()
