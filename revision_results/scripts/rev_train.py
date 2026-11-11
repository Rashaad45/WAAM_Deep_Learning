"""
rev_train.py - LOTO x 5 seeds for (model, feature configuration) pairs, same protocol as src/run_experiments.py.

    python revision_results/scripts/rev_train.py --models logreg rf svm --configs F0 F1 ... --workers 10

Writes revision_results/oof_raw/<model>_<config>.csv (one row per seed x zone),
revision_results/oof_raw/<model>_<config>_loss.csv (deep models) and appends run times to
revision_results/run_times.csv. Pairs that already have a CSV are skipped unless --force.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import rev_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pairs = [(m, c) for m in a.models for c in a.configs
             if a.force or not (C.RAW / f"{m}_{c}.csv").exists()]
    ck = C.RAW / "jobs"
    ck.mkdir(exist_ok=True)
    res = {p: ([], [], 0.0) for p in pairs}
    jobs = []
    for m, c in pairs:          # resume from per-job checkpoints (a crash loses only running jobs)
        for s in C.SEEDS:
            f = ck / f"{m}_{c}_{s}.pkl"
            if f.exists() and not a.force:
                p, l, sec = pd.read_pickle(f)
                res[(m, c)] = (res[(m, c)][0] + p, res[(m, c)][1] + l, res[(m, c)][2] + sec)
            else:
                jobs.append((m, c, s))
    order = {"gat": 0, "gcn": 1, "fusion_mlp": 2, "cnn1d": 3, "bilstm": 4, "small_gnn": 5}
    jobs.sort(key=lambda j: order.get(j[0], 9))
    print(f"{len(jobs)} jobs for {pairs}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {pool.submit(C.job, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            p, l, sec = f.result()
            pd.to_pickle((p, l, sec), ck / f"{j[0]}_{j[1]}_{j[2]}.pkl")
            key = (j[0], j[1])
            res[key][0].extend(p); res[key][1].extend(l)
            res[key] = (res[key][0], res[key][1], res[key][2] + sec)
            print(f"[{n}/{len(jobs)}] {j} {sec:.0f}s  (wall {(time.time() - t0) / 60:.1f} min)", flush=True)
            if all(len(res[k][0]) == 48 * len(C.SEEDS) for k in [key]):
                pd.DataFrame(res[key][0]).sort_values(["seed", "trial_id", "zone_id"]).to_csv(
                    C.RAW / f"{key[0]}_{key[1]}.csv", index=False)
                if res[key][1]:
                    pd.DataFrame(res[key][1]).to_csv(C.RAW / f"{key[0]}_{key[1]}_loss.csv", index=False)
                rt = C.REV / "run_times.csv"
                row = pd.DataFrame([dict(model=key[0], config=key[1], cpu_seconds_all_seeds=round(res[key][2], 1),
                                         finished=time.strftime("%Y-%m-%d %H:%M:%S"))])
                row.to_csv(rt, mode="a", header=not rt.exists(), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
