"""
run_experiments.py - STEP 3: leave-one-trial-out (LOTO) training of every model, 5 seeds.

Protocol (fixed before the rerun, DECISION.md):
  * outer loop: LOTO over the 16 usable trials (one trial = one test fold; no trial ever in train and test);
  * NO inner validation set, NO early stopping: every deep model trains for a fixed 40 epochs;
  * loss = class-weighted BCE only (pos_weight = negatives / positives of the training trials);
    weight decay 1e-4 is Adam's (decoupled from the loss); no Ra term;
  * inputs standardised with training-trial statistics only;
  * a fold whose training trials contain no positive zone predicts the training prevalence (0)
    and is flagged `no_positive_training`.

Writes results/predictions.csv, results/training_loss.csv, results/parameter_counts.csv.
Usage:  python src/run_experiments.py --workers 10
"""
import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CONFIG, RESULTS_DIR
from graph_data import TrialGraphBuilder, apply_variant, FORCE_COLS, ACCEL_COLS
from models import DEEP_MODELS, build_model, count_parameters

CLASSICAL = ["logreg", "rf", "svm"]
ALL_MODELS = CLASSICAL + DEEP_MODELS
ALIGNMENT_FLAGGED = ("trial_04", "trial_06")

# (config name, label threshold dB, excluded trials, [(model, variant), ...])
EXPERIMENTS = [
    ("main_15db", 15, (), [(m, "full") for m in ALL_MODELS]),
    ("main_20db", 20, (), [(m, "full") for m in CLASSICAL + ["bilstm", "small_gnn"]]),
    ("ablation_15db", 15, (), [("gat", "no_edges"), ("gcn", "no_edges"), ("small_gnn", "no_edges"),
                               ("gat", "no_force"), ("gat", "no_acceleration"),
                               ("logreg", "no_force"), ("logreg", "no_acceleration")]),
    ("alignment_15db", 15, ALIGNMENT_FLAGGED, [(m, "full") for m in ALL_MODELS]),
]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def loto_folds(graphs: list[dict]) -> list[tuple[list[int], int]]:
    """[(train indices, test index)]; asserts that no trial is in both and that only lab data is used."""
    ids = [g["trial_id"] for g in graphs]
    assert len(set(ids)) == len(ids), "duplicate trial id"
    assert all(g.get("source") == "WAAM_LAB" for g in graphs), "non-lab (external) sample in a WAAM split"
    folds = []
    for k in range(len(graphs)):
        train = [i for i in range(len(graphs)) if i != k]
        assert ids[k] not in {ids[i] for i in train}
        folds.append((train, k))
    return folds


class Standardiser:
    def __init__(self, train: list[dict]):
        x = torch.cat([g["x"].reshape(-1, g["x"].shape[-1]) for g in train])
        self.mean, self.std = x.mean(0), x.std(0).clamp_min(1e-6)
        self.fs = torch.cat([g["force_raw"].flatten()[::97] for g in train]).std().clamp_min(1e-6)
        self.as_ = torch.cat([g["acceleration_raw"].flatten()[::97] for g in train]).std().clamp_min(1e-6)

    def __call__(self, g: dict) -> dict:
        s = dict(g)
        s["x"] = (g["x"] - self.mean) / self.std
        s["force_raw"] = g["force_raw"] / self.fs
        s["acceleration_raw"] = g["acceleration_raw"] / self.as_
        return s


def bce_loss(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """The ONLY training loss: class-weighted binary cross-entropy (Eq. 9 of the paper)."""
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def zone_feature_matrix(g: dict) -> np.ndarray:
    """Classical models: window-mean of the hand-crafted features -> [N, F] (same aggregation rule)."""
    return g["x"].mean(dim=1).numpy()


def classical_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    if name == "rf":
        return RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=seed))
    raise ValueError(name)


def run_job(job: tuple) -> tuple[list[dict], list[dict]]:
    """One (experiment, model, variant, seed) over all LOTO folds."""
    exp, thr, exclude, model, variant, seed = job
    torch.set_num_threads(CONFIG.train.torch_threads)
    graphs = [apply_variant(g, variant) for g in TrialGraphBuilder(CONFIG, thr).load_all(exclude)]
    preds, losses = [], []
    for fold, (train_idx, k) in enumerate(loto_folds(graphs)):
        test = graphs[k]
        ytr = torch.cat([graphs[i]["y"] for i in train_idx])
        n_pos = int(ytr.sum())
        flag = n_pos == 0
        if flag:
            prob = np.full(len(test["y"]), float(ytr.mean()))
        elif model in CLASSICAL:
            xtr = np.concatenate([zone_feature_matrix(graphs[i]) for i in train_idx])
            prob = classical_model(model, seed).fit(xtr, ytr.numpy()).predict_proba(zone_feature_matrix(test))[:, 1]
        else:
            seed_everything(seed)
            scaler = Standardiser([graphs[i] for i in train_idx])
            train = [scaler(graphs[i]) for i in train_idx]
            net = build_model(model, train[0]["x"].shape[-1], CONFIG)
            opt = torch.optim.Adam(net.parameters(), lr=CONFIG.train.learning_rate,
                                   weight_decay=CONFIG.train.weight_decay)
            pw = torch.tensor(float((ytr == 0).sum()) / n_pos)
            for epoch in range(CONFIG.train.epochs):
                net.train()
                total = 0.0
                for i in np.random.permutation(len(train)):
                    opt.zero_grad()
                    loss = bce_loss(net(train[i]), train[i]["y"], pw)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    total += loss.item()
                losses.append(dict(experiment=exp, model=model, variant=variant, seed=seed, fold=fold,
                                   epoch=epoch + 1, train_loss=total / len(train)))
            net.eval()
            with torch.no_grad():
                prob = torch.sigmoid(net(scaler(test))).numpy()
        for z in range(len(test["y"])):
            preds.append(dict(experiment=exp, threshold_db=thr, model=model, variant=variant, seed=seed,
                              fold=fold, trial_id=test["trial_id"], zone_id=z, y=int(test["y"][z]),
                              probability=float(prob[z]), no_positive_training=flag))
    return preds, losses


def parameter_counts() -> pd.DataFrame:
    g = TrialGraphBuilder(CONFIG, 15).load_all()[0]
    f = g["x"].shape[-1]
    rows = [dict(model=m, parameters=count_parameters(build_model(m, f, CONFIG))) for m in DEEP_MODELS]
    rows += [dict(model="logreg", parameters=f + 1)]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--experiments", nargs="+", default=[e[0] for e in EXPERIMENTS])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    parameter_counts().to_csv(RESULTS_DIR / "parameter_counts.csv", index=False)
    jobs = [(e, thr, exc, m, v, s) for e, thr, exc, mv in EXPERIMENTS if e in args.experiments
            for m, v in mv if m in args.models for s in CONFIG.train.seeds]
    # Slowest first for better load balancing.
    order = {"gat": 0, "gcn": 1, "fusion_mlp": 2, "cnn1d": 3, "bilstm": 4, "small_gnn": 5}
    jobs.sort(key=lambda j: order.get(j[3], 9))
    t0, preds, losses = time.time(), [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            p, l = fut.result()
            preds += p; losses += l
            j = futures[fut]
            print(f"[{n}/{len(jobs)}] {j[0]} {j[3]} {j[4]} seed {j[5]}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    sfx = f"_{args.tag}" if args.tag else ""
    pd.DataFrame(preds).to_csv(RESULTS_DIR / f"predictions{sfx}.csv", index=False)
    pd.DataFrame(losses).to_csv(RESULTS_DIR / f"training_loss{sfx}.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
