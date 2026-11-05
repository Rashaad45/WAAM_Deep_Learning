"""Step 2 of the pipeline: leave-one-trial-out training and evaluation (Section 2.9).

What it produces (all in outputs/):
  metrics_main.csv        Tables 2 and 3: every model, pooled out-of-fold metrics + 95 % CI
  ablation.csv            Table 4: ST-GAT ablations
  attribution.csv         Table 5: integrated-gradient share per modality and material
  attention.csv           mean GAT attention on kinematic vs thermal edges per material
  stat_tests.csv          DeLong (AUC) and Wilcoxon (fold RMSE) tests, GAT vs fusion MLP
  fold_metrics.csv        per-fold metrics of every run
  predictions_oof.csv     every out-of-fold zone prediction
  latency.json            parameter count and inference time of the ST-GAT
  run_info.json           settings, software versions, dataset composition

Usage examples
  python src/train_evaluate.py                       # main comparison + ablations
  python src/train_evaluate.py --quick               # fast check (few epochs, 2 folds)
  python src/train_evaluate.py --seeds 42 43 44 45 46 # final paper run
  python src/train_evaluate.py --protocol loco        # leave-one-condition-out (Table S1)
"""
import argparse
import json
import platform
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from config import CONFIG, OUTPUT_DIR
from graph_data import (ACCEL_COLS, FORCE_COLS, KINEMATIC_COLS, apply_variant, load_graphs)
from models import build_model

warnings.filterwarnings("ignore", category=FutureWarning)      # sklearn SVC(probability=True) notice
warnings.filterwarnings("ignore", category=UserWarning)

DEEP_MODELS = ["cnn1d", "bilstm", "fusion_mlp", "gcn", "gat"]
ABLATIONS = ["without_force_features", "without_acceleration_features", "without_thermal_edges",
             "without_kinematic_edges", "no_spatial_edges", "without_temporal_module", "material_flag",
             "classification_only", "roughness_only"]


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_folds(graphs: list[dict], protocol: str) -> list[list[int]]:
    """Indices of the test graphs of every outer fold."""
    if protocol == "loto":
        return [[i] for i in range(len(graphs))]
    groups = {}
    for i, g in enumerate(graphs):
        groups.setdefault(g["condition"], []).append(i)
    return list(groups.values())


def pick_validation(train_idx: list[int], graphs: list[dict], rng: np.random.Generator) -> list[int]:
    """Inner trial-grouped validation split: one trial of each material when possible."""
    val = []
    for material in ("WAAM", "WROUGHT"):
        pool = [i for i in train_idx if graphs[i]["material_type"] == material]
        if len(pool) > 1:
            val.append(int(rng.choice(pool)))
    return val or [int(rng.choice(train_idx))]


def standardise(train: list[dict], others: list[dict]) -> tuple[list[dict], list[dict], float, float]:
    """Scale node features and Ra with statistics of the training trials only."""
    x = torch.cat([g["x"].reshape(-1, g["x"].shape[-1]) for g in train])
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    ra = torch.cat([g["y_reg"] for g in train])
    ra_mean, ra_std = float(ra.mean()), max(float(ra.std()), 1e-6)
    # Raw windows: one scale per modality so the encoders see O(1) inputs.
    f_scale = torch.cat([g["force_raw"].flatten()[::97] for g in train]).std().clamp_min(1e-6)
    a_scale = torch.cat([g["acceleration_raw"].flatten()[::97] for g in train]).std().clamp_min(1e-6)

    def scale(g):
        out = dict(g)
        out["x"] = (g["x"] - mean) / std
        out["force_raw"] = g["force_raw"] / f_scale
        out["acceleration_raw"] = g["acceleration_raw"] / a_scale
        out["y_reg_std"] = (g["y_reg"] - ra_mean) / ra_std
        return out
    return [scale(g) for g in train], [scale(g) for g in others], ra_mean, ra_std


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred) * 100, "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"rmse_um": float(np.sqrt(mean_squared_error(y, p))), "mae_um": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)) if np.var(y) > 0 else np.nan}


def pooled_metrics_with_ci(pred: pd.DataFrame, draws: int, seed: int) -> dict:
    """Pooled out-of-fold metrics and 95 % CIs from a bootstrap over TRIALS (not zones)."""
    result = {}
    trials = pred.trial_id.unique()
    rng = np.random.default_rng(seed)
    point = {}
    if pred.probability.notna().any():
        point.update(classification_metrics(pred.y_class.to_numpy(int), pred.probability.to_numpy()))
    if pred.ra_pred.notna().any():
        point.update(regression_metrics(pred.y_reg.to_numpy(), pred.ra_pred.to_numpy()))
    boot = {k: [] for k in point}
    by_trial = {t: pred[pred.trial_id == t] for t in trials}
    for _ in range(draws):
        sample = pd.concat([by_trial[t] for t in rng.choice(trials, len(trials), replace=True)])
        if "accuracy" in point:
            m = classification_metrics(sample.y_class.to_numpy(int), sample.probability.to_numpy())
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                boot[k].append(m[k])
        if "rmse_um" in point:
            m = regression_metrics(sample.y_reg.to_numpy(), sample.ra_pred.to_numpy())
            for k in ("rmse_um", "mae_um", "r2"):
                boot[k].append(m[k])
    for k, v in point.items():
        arr = np.asarray(boot[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        result[k] = v
        result[f"{k}_ci_low"] = float(np.quantile(arr, 0.025)) if arr.size else np.nan
        result[f"{k}_ci_high"] = float(np.quantile(arr, 0.975)) if arr.size else np.nan
    return result


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    """DeLong et al. (1988) test for two correlated ROC AUCs. Returns (auc1, auc2, p)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan

    def components(p):
        psi = (p[pos][:, None] > p[neg][None, :]).astype(float) + 0.5 * (p[pos][:, None] == p[neg][None, :])
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)
    a1, v10_1, v01_1 = components(p1)
    a2, v10_2, v01_2 = components(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 0:
        return a1, a2, 1.0
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------
# Classical baselines (zone features averaged over the windows)
# --------------------------------------------------------------------------
def run_classical(graphs: list[dict], folds: list[list[int]], seed: int) -> pd.DataFrame:
    clf = {"Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
           "Random forest": lambda: RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1),
           "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=seed))}
    reg = {"Ridge regression": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
           "Random forest regressor": lambda: RandomForestRegressor(300, random_state=seed, n_jobs=-1),
           "SVR (RBF)": lambda: make_pipeline(StandardScaler(), SVR())}
    X = [g["x"].mean(1).numpy() for g in graphs]
    rows = []
    for fold, test in enumerate(folds):
        train = [i for i in range(len(graphs)) if i not in test]
        xtr, xte = np.concatenate([X[i] for i in train]), np.concatenate([X[i] for i in test])
        yc = np.concatenate([graphs[i]["y_class"].numpy() for i in train])
        yr = np.concatenate([graphs[i]["y_reg"].numpy() for i in train])
        meta = [(graphs[i]["trial_id"], graphs[i]["material_type"], z, float(graphs[i]["y_class"][z]),
                 float(graphs[i]["y_reg"][z])) for i in test for z in range(len(graphs[i]["y_class"]))]
        for name, make in clf.items():
            prob = make().fit(xtr, yc).predict_proba(xte)[:, 1] if len(np.unique(yc)) > 1 else np.full(len(xte), yc.mean())
            rows += [dict(model=name, fold=fold, trial_id=t, material_type=mat, zone_id=z, y_class=c, y_reg=r,
                          probability=float(p), ra_pred=np.nan) for (t, mat, z, c, r), p in zip(meta, prob)]
        for name, make in reg.items():
            pred = make().fit(xtr, yr).predict(xte)
            rows += [dict(model=name, fold=fold, trial_id=t, material_type=mat, zone_id=z, y_class=c, y_reg=r,
                          probability=np.nan, ra_pred=float(p)) for (t, mat, z, c, r), p in zip(meta, pred)]
        rows += [dict(model="Predict training mean (reference)", fold=fold, trial_id=t, material_type=mat, zone_id=z,
                      y_class=c, y_reg=r, probability=np.nan, ra_pred=float(yr.mean())) for (t, mat, z, c, r) in meta]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Deep models
# --------------------------------------------------------------------------
def loss_fn(out, g, pos_weight, task: str, lam: float) -> torch.Tensor:
    """Eq. (14): class-weighted BCE + lambda * MSE on standardised Ra."""
    logits, ra, _ = out
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, g["y_class"], pos_weight=pos_weight)
    mse = torch.nn.functional.mse_loss(ra, g["y_reg_std"])
    if task == "classification_only":
        return bce
    if task == "roughness_only":
        return mse
    return bce + lam * mse


def train_one(kind: str, variant: str, train: list[dict], val: list[dict], seed: int, epochs: int, patience: int):
    """Train one model with early stopping on the inner-validation loss; return the best model."""
    cfg = CONFIG
    seed_everything(seed)
    model = build_model(kind, train[0]["x"].shape[-1], cfg, variant)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    y = torch.cat([g["y_class"] for g in train])
    pos_weight = torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))
    task = variant if variant in ("classification_only", "roughness_only") else "joint"
    best, best_state, stale = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for i in np.random.permutation(len(train)):
            opt.zero_grad()
            loss = loss_fn(model(train[i]), train[i], pos_weight, task, cfg.train.regression_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss_fn(model(g), g, pos_weight, task, cfg.train.regression_weight).item() for g in val]))
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1


def integrated_gradients(model, g: dict, steps: int = 16) -> dict:
    """Integrated gradients (Sundararajan et al., 2017) of the summed Ra output.

    Baseline = all-zero standardised input (the training mean). Attributions of
    hand-crafted features and raw windows are summed per modality.
    """
    keys = ["x", "force_raw", "acceleration_raw"]
    total = {k: torch.zeros_like(g[k]) for k in keys}
    for alpha in torch.linspace(1.0 / steps, 1.0, steps):
        scaled = dict(g)
        for k in keys:
            scaled[k] = (g[k] * alpha).requires_grad_(True)
        _, ra, _ = model(scaled)
        grads = torch.autograd.grad(ra.sum(), [scaled[k] for k in keys])
        for k, gr in zip(keys, grads):
            total[k] += gr / steps
    attr = {k: (total[k] * g[k]).abs() for k in keys}
    per_zone = lambda t: t.reshape(t.shape[0], -1).sum(1)
    return {"Force": per_zone(attr["x"][..., FORCE_COLS]) + per_zone(attr["force_raw"]),
            "Acceleration": per_zone(attr["x"][..., ACCEL_COLS]) + per_zone(attr["acceleration_raw"]),
            "Kinematic / geometric": per_zone(attr["x"][..., KINEMATIC_COLS])}


def run_deep(graphs, folds, kinds, variants, seeds, epochs, patience, protocol, log):
    """Outer CV loop for every (model, variant, seed). Returns predictions and extras."""
    rows, attributions, attention = [], [], []
    rng = np.random.default_rng(0)
    vals = []
    for test in folds:
        train_idx = [i for i in range(len(graphs)) if i not in test]
        vals.append(pick_validation(train_idx, graphs, rng))
    jobs = [(k, "full") for k in kinds] + [("gat", v) for v in variants]
    for kind, variant in jobs:
        started = time.time()
        for fold, test in enumerate(folds):
            val_idx = vals[fold]
            train_idx = [i for i in range(len(graphs)) if i not in test and i not in val_idx]
            view = [apply_variant(g, variant) for g in graphs]
            train, others, ra_mean, ra_std = standardise([view[i] for i in train_idx],
                                                         [view[i] for i in val_idx + test])
            val, tests = others[:len(val_idx)], others[len(val_idx):]
            for seed in seeds:
                model, n_epochs = train_one(kind, variant, train, val, seed, epochs, patience)
                for g in tests:
                    with torch.no_grad():
                        logits, ra, _ = model(g)
                    prob = torch.sigmoid(logits).numpy()
                    ra_um = ra.numpy() * ra_std + ra_mean
                    for z in range(len(prob)):
                        rows.append(dict(model=kind, variant=variant, seed=seed, fold=fold, trial_id=g["trial_id"],
                                         material_type=g["material_type"], zone_id=z,
                                         y_class=float(g["y_class"][z]), y_reg=float(g["y_reg"][z]),
                                         probability=float(prob[z]) if variant != "roughness_only" else np.nan,
                                         ra_pred=float(ra_um[z]) if variant != "classification_only" else np.nan,
                                         epochs_trained=n_epochs))
                    if kind == "gat" and variant == "full":
                        kin, th = model.attention_by_edge_type(g["adj"])
                        attention.append(dict(trial_id=g["trial_id"], material_type=g["material_type"], seed=seed,
                                              kinematic_attention=kin, thermal_attention=th))
                        for modality, values in integrated_gradients(model, g).items():
                            attributions.append(dict(trial_id=g["trial_id"], material_type=g["material_type"],
                                                     seed=seed, modality=modality, attribution=float(values.sum())))
        log(f"  {kind:<10} {variant:<32} done in {(time.time() - started) / 60:.1f} min")
    return pd.DataFrame(rows), pd.DataFrame(attributions), pd.DataFrame(attention)


def measure_latency(graphs: list[dict]) -> dict:
    """Parameter count and CPU inference time per graph of the ST-GAT."""
    model = build_model("gat", graphs[0]["x"].shape[-1], CONFIG).eval()
    train, tests, _, _ = standardise(graphs, graphs[:1])
    g = tests[0]
    n_windows = g["x"].shape[1]
    single = {k: (v[:, :1] if k in ("x", "force_raw", "acceleration_raw") else v) for k, v in g.items()}
    with torch.no_grad():
        for _ in range(3):
            model(single)
        t0 = time.perf_counter()
        for _ in range(20):
            model(single)
        per_window_ms = (time.perf_counter() - t0) / 20 * 1000
        t0 = time.perf_counter()
        for _ in range(3):
            model(g)
        per_trial_ms = (time.perf_counter() - t0) / 3 * 1000
    hop_ms = CONFIG.signal.window_ms * (1 - CONFIG.signal.overlap_fraction)
    return {"parameters": int(sum(p.numel() for p in model.parameters())),
            "ms_per_graph_one_window": round(per_window_ms, 2),
            "ms_per_graph_all_windows": round(per_trial_ms, 1), "windows_per_graph": int(n_windows),
            "budget_ms_one_hop": hop_ms, "within_budget": bool(per_window_ms < hop_ms),
            "hardware": platform.processor() or platform.machine(), "threads": torch.get_num_threads()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def summarise(pred: pd.DataFrame, key_cols: list[str], draws: int) -> pd.DataFrame:
    """Average seeds per zone, then pooled metrics with trial-bootstrap CI per group."""
    out = []
    for keys, frame in pred.groupby(key_cols, sort=False):
        zone = frame.groupby(["trial_id", "material_type", "zone_id"], as_index=False).agg(
            y_class=("y_class", "first"), y_reg=("y_reg", "first"),
            probability=("probability", "mean"), ra_pred=("ra_pred", "mean"))
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.append({**dict(zip(key_cols, keys)), **pooled_metrics_with_ci(zone, draws, 0)})
    return pd.DataFrame(out)


def fold_rmse(pred: pd.DataFrame) -> pd.DataFrame:
    """Ra RMSE of every (model, variant, fold), seeds averaged per zone first."""
    rows = []
    for (model, variant, fold), f in pred.groupby(["model", "variant", "fold"]):
        z = f.groupby(["trial_id", "zone_id"]).agg(y_reg=("y_reg", "first"), ra_pred=("ra_pred", "mean")).dropna()
        rows.append(dict(model=model, variant=variant, fold=fold, trial_id=f.trial_id.iloc[0],
                         rmse_um=np.sqrt(np.mean((z.y_reg - z.ra_pred) ** 2)) if len(z) else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["main", "ablations", "all"], default="main",
                        help="main = Tables 2-3, 5 (classical + 5 deep models); ablations = Table 4 (ST-GAT variants)")
    parser.add_argument("--models", nargs="+", default=DEEP_MODELS, choices=DEEP_MODELS)
    parser.add_argument("--ablations", nargs="+", default=ABLATIONS, choices=ABLATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CONFIG.train.seeds))
    parser.add_argument("--epochs", type=int, default=CONFIG.train.epochs)
    parser.add_argument("--protocol", choices=["loto", "loco"], default="loto")
    parser.add_argument("--quick", action="store_true", help="3 epochs, 2 folds: checks that everything runs")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    torch.set_num_threads(CONFIG.train.torch_threads)
    out = args.output if args.protocol == "loto" else args.output / "loco"
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / f"training_log_{args.stage}.txt").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    graphs = load_graphs(CONFIG)
    folds = make_folds(graphs, args.protocol)
    epochs, patience, draws = args.epochs, CONFIG.train.patience, CONFIG.train.bootstrap_draws
    if args.quick:
        folds, epochs, patience, draws = folds[:2], 3, 2, 50
    log(f"{len(graphs)} trial graphs, {len(folds)} {args.protocol.upper()} folds, seeds {args.seeds}, "
        f"up to {epochs} epochs, stage '{args.stage}'")
    t0 = time.time()
    run_main = args.stage in ("main", "all")
    run_ablation = args.stage in ("ablations", "all") and args.protocol == "loto"

    # ---- ablations (Table 4) ------------------------------------------------
    if run_ablation:
        abl, _, _ = run_deep(graphs, folds, [], args.ablations, args.seeds, epochs, patience, args.protocol, log)
        main_pred_path = out / "predictions_oof.csv"
        if main_pred_path.exists():                   # reuse the full ST-GAT from the main stage
            prev = pd.read_csv(main_pred_path)
            full = prev[(prev.model == "gat") & (prev.variant == "full")]
        else:
            full, _, _ = run_deep(graphs, folds, ["gat"], [], args.seeds, epochs, patience, args.protocol, log)
        abl = pd.concat([full, abl], ignore_index=True)
        abl.to_csv(out / "predictions_ablation.csv", index=False)
        summarise(abl, ["variant"], draws).to_csv(out / "ablation.csv", index=False)
        fold_rmse(abl).to_csv(out / "fold_metrics_ablation.csv", index=False)
        log(f"Ablations done in {(time.time() - t0) / 60:.1f} min -> {out / 'ablation.csv'}")
        if not run_main:
            return

    # ---- classical baselines -------------------------------------------------
    classical = run_classical(graphs, folds, args.seeds[0])
    log(f"Classical baselines done in {time.time() - t0:.0f} s")

    # ---- deep models -----------------------------------------------------------
    deep, attributions, attention = run_deep(graphs, folds, args.models, [], args.seeds, epochs, patience,
                                             args.protocol, log)

    # ---- tables ---------------------------------------------------------------
    classical["variant"], classical["seed"] = "full", args.seeds[0]
    all_pred = pd.concat([classical, deep], ignore_index=True)
    all_pred.to_csv(out / "predictions_oof.csv", index=False)

    main_table = summarise(all_pred, ["model"], draws)
    main_table.to_csv(out / "metrics_main.csv", index=False)
    summarise(all_pred, ["model", "material_type"], draws).to_csv(out / "metrics_by_material.csv", index=False)
    fold_metrics = fold_rmse(all_pred)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)

    # ---- statistical tests: ST-GAT vs fusion MLP ----------------------------
    tests = []
    if {"gat", "fusion_mlp"} <= set(args.models):
        z = (deep[deep.variant == "full"].groupby(["model", "trial_id", "zone_id"], as_index=False)
             .agg(y_class=("y_class", "first"), probability=("probability", "mean")))
        a = z[z.model == "gat"].sort_values(["trial_id", "zone_id"])
        b = z[z.model == "fusion_mlp"].sort_values(["trial_id", "zone_id"])
        auc1, auc2, p = delong_test(a.y_class.to_numpy(int), a.probability.to_numpy(), b.probability.to_numpy())
        tests.append(dict(test="DeLong AUC", model_a="gat", model_b="fusion_mlp", value_a=auc1, value_b=auc2, p_value=p))
        fa = fold_metrics[(fold_metrics.model == "gat") & (fold_metrics.variant == "full")].set_index("fold").rmse_um
        fb = fold_metrics[(fold_metrics.model == "fusion_mlp") & (fold_metrics.variant == "full")].set_index("fold").rmse_um
        diff = (fa - fb).dropna()
        if len(diff) >= 2 and np.any(diff != 0):
            w = stats.wilcoxon(diff)
            tests.append(dict(test="Wilcoxon fold RMSE", model_a="gat", model_b="fusion_mlp",
                              value_a=fa.mean(), value_b=fb.mean(), p_value=float(w.pvalue)))
    pd.DataFrame(tests).to_csv(out / "stat_tests.csv", index=False)

    # ---- attribution and attention ------------------------------------------
    if not attributions.empty:
        share = attributions.groupby(["material_type", "modality"]).attribution.sum()
        share = (share / share.groupby(level=0).transform("sum") * 100).rename("share_percent").reset_index()
        share.to_csv(out / "attribution.csv", index=False)
    if not attention.empty:
        attention.groupby("material_type")[["kinematic_attention", "thermal_attention"]].mean().to_csv(out / "attention.csv")

    # ---- latency and run info ------------------------------------------------
    (out / "latency.json").write_text(json.dumps(measure_latency(graphs), indent=2))
    composition = pd.DataFrame([dict(trial=g["trial_id"], material=g["material_type"],
                                     unstable=int(g["y_class"].sum()), zones=len(g["y_class"]),
                                     windows=int(g["x"].shape[1])) for g in graphs])
    info = {"protocol": args.protocol, "folds": len(folds), "seeds": args.seeds, "max_epochs": epochs,
            "patience": patience, "models": args.models, "ablations": args.ablations, "quick": args.quick,
            "ra_labels_imputed": any(g["ra_imputed"] for g in graphs),
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "config": {"window_ms": CONFIG.signal.window_ms, "overlap": CONFIG.signal.overlap_fraction,
                       "hidden_dim": CONFIG.train.hidden_dim, "heads": CONFIG.train.attention_heads,
                       "dropout": CONFIG.train.dropout, "lr": CONFIG.train.learning_rate,
                       "weight_decay": CONFIG.train.weight_decay, "temporal_kernel": CONFIG.train.temporal_kernel,
                       "regression_weight": CONFIG.train.regression_weight,
                       "thermal_diffusivity_m2_s": CONFIG.graph.thermal_diffusivity_m2_s},
            "dataset": composition.to_dict("records"),
            "total_minutes": round((time.time() - t0) / 60, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    log(f"\nMain results (pooled out-of-fold):\n{main_table.round(3).to_string(index=False)}")
    log(f"\nAll outputs written to {out}")


if __name__ == "__main__":
    main()
