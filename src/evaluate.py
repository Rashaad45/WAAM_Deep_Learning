"""
evaluate.py - turns results/predictions.csv into every table of the paper (results/*.csv).

* seeds are averaged per zone before any metric;
* pooled out-of-fold metrics; 95 % CI by resampling TRIALS (2000 draws);
* label-permutation null for AUC (2000 zone-label permutations; ignores within-trial clustering,
  so it is, if anything, narrower than a trial-level null);
* model comparisons only by TRIAL-LEVEL paired tests: paired trial bootstrap of the AUC difference
  and a paired permutation test that swaps the two models' scores trial by trial (no DeLong);
* label counts and label sensitivity (harmonic tolerance x threshold x measured / nominal spindle speed).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score)

import os
from pathlib import Path
from config import CONFIG, DATA_DIR, CACHE_DIR, RESULTS_DIR as _RD
RESULTS_DIR = Path(os.environ.get("WAAM_RESULTS_DIR", _RD))

DRAWS = CONFIG.train.bootstrap_draws
PERMS = CONFIG.train.permutation_draws
MODEL_ORDER = ["logreg", "rf", "svm", "cnn1d", "bilstm", "fusion_mlp", "small_gnn", "gcn", "gat"]


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction (identical to sklearn's roc_auc_score)."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc(y, p):
    return fast_auc(np.asarray(y), np.asarray(p))


def zones(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["trial_id", "zone_id"], as_index=False).agg(
        y=("y", "first"), probability=("probability", "mean"),
        no_positive_training=("no_positive_training", "max"))


def bootstrap_indices(trials: np.ndarray, rng) -> list[np.ndarray]:
    """DRAWS index arrays, each a resample of whole trials with replacement."""
    uniq = np.unique(trials)
    rows = [np.flatnonzero(trials == t) for t in uniq]
    return [np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(DRAWS)]


def summarise(z: pd.DataFrame, rng) -> dict:
    y, p = z.y.to_numpy(int), z.probability.to_numpy()
    yhat = (p >= 0.5).astype(int)
    a = auc(y, p)
    res = dict(n_zones=len(z), n_positive=int(y.sum()), n_trials=z.trial_id.nunique(),
               n_trials_with_positive=int(z.groupby("trial_id").y.max().sum()),
               folds_no_positive_training=int(z.groupby("trial_id").no_positive_training.max().sum()),
               auc=a, average_precision=average_precision_score(y, p) if y.sum() else np.nan,
               prevalence=y.mean(), balanced_accuracy=balanced_accuracy_score(y, yhat) if 0 < y.sum() < len(y) else np.nan,
               f1=f1_score(y, yhat, zero_division=0), precision=precision_score(y, yhat, zero_division=0),
               recall=recall_score(y, yhat, zero_division=0))
    within = [auc(g.y.to_numpy(int), g.probability.to_numpy()) for _, g in z.groupby("trial_id")]
    within = [w for w in within if np.isfinite(w)]
    res.update(n_trials_both_classes=len(within),
               mean_within_trial_auc=float(np.mean(within)) if within else np.nan)
    if np.isnan(a):
        return res
    idx = bootstrap_indices(z.trial_id.to_numpy(), rng)
    boot = np.array([auc(y[i], p[i]) for i in idx]); boot = boot[np.isfinite(boot)]
    bap = np.array([average_precision_score(y[i], p[i]) if y[i].sum() else np.nan for i in idx])
    bap = bap[np.isfinite(bap)]
    null = np.array([auc(rng.permutation(y), p) for _ in range(PERMS)])
    res.update(auc_ci_low=np.quantile(boot, .025), auc_ci_high=np.quantile(boot, .975),
               ap_ci_low=np.quantile(bap, .025), ap_ci_high=np.quantile(bap, .975),
               null_auc_q025=np.quantile(null, .025), null_auc_q975=np.quantile(null, .975),
               perm_p_auc=(1 + np.sum(np.abs(null - .5) >= abs(a - .5))) / (PERMS + 1))
    return res


def paired_test(za: pd.DataFrame, zb: pd.DataFrame, rng) -> dict:
    m = za.merge(zb, on=["trial_id", "zone_id", "y"], suffixes=("_a", "_b"))
    y = m.y.to_numpy(int)
    pa, pb = m.probability_a.to_numpy(), m.probability_b.to_numpy()
    d0 = auc(y, pa) - auc(y, pb)
    trials = m.trial_id.to_numpy()
    boot = np.array([auc(y[i], pa[i]) - auc(y[i], pb[i]) for i in bootstrap_indices(trials, rng)])
    boot = boot[np.isfinite(boot)]
    uniq = np.unique(trials)
    null = []
    for _ in range(PERMS):
        flip = np.isin(trials, uniq[rng.random(len(uniq)) < 0.5])
        null.append(auc(y, np.where(flip, pb, pa)) - auc(y, np.where(flip, pa, pb)))
    null = np.asarray(null)
    return dict(delta_auc=d0, delta_ci_low=np.quantile(boot, .025), delta_ci_high=np.quantile(boot, .975),
                perm_p=(1 + np.sum(np.abs(null) >= abs(d0) - 1e-12)) / (PERMS + 1), n_trials=len(uniq))

def label_tables() -> None:
    lab = pd.read_csv(DATA_DIR / "zone_labels.csv")
    summ = pd.read_csv(CACHE_DIR / "dataset_summary.csv").set_index("trial_id")
    rows = []
    for thr in CONFIG.labels.thresholds_db:
        c = lab[f"label_{int(thr)}db"]
        rows.append(dict(threshold_db=thr, n_zones=len(lab), n_positive=int(c.sum()),
                         n_trials=lab.trial_id.nunique(),
                         n_trials_with_positive=int(lab[c == 1].trial_id.nunique()),
                         zones_within_2db=int((np.abs(lab.prominence_db - thr) < 2).sum())))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "label_counts.csv", index=False)

    base = (lab.dist_to_spindle_harmonic_hz > CONFIG.signal.harmonic_tolerance_hz) & (lab.prominence_db >= 15)
    grid = []
    for source, dist in (("measured", lab.dist_to_spindle_harmonic_hz), ("nominal", lab.dist_to_nominal_harmonic_hz)):
        for tol in (0.5, 1.0, 2.0):
            for thr in CONFIG.labels.thresholds_db:
                lbl = (dist > tol) & (lab.prominence_db >= thr)
                grid.append(dict(spindle_frequency=source, tolerance_hz=tol, threshold_db=thr,
                                 n_positive=int(lbl.sum()), n_trials_with_positive=int(lab[lbl].trial_id.nunique()),
                                 n_changed_vs_main=int((lbl != base).sum())))
    pd.DataFrame(grid).to_csv(RESULTS_DIR / "label_sensitivity.csv", index=False)

    peaks = lab.merge(summ[["spindle_hz_measured"]], left_on="trial_id", right_index=True)
    peaks["on_harmonic"] = peaks.dist_to_spindle_harmonic_hz <= CONFIG.signal.harmonic_tolerance_hz
    peaks.to_csv(RESULTS_DIR / "zone_label_details.csv", index=False)


def constants() -> None:
    """Experimental-setup numbers quoted in the manuscript (so check_numbers.py can trace them)."""
    c, s = CONFIG, pd.read_csv(CACHE_DIR / "dataset_summary.csv")
    inc = s[s.included.astype(bool)]
    lab = pd.read_csv(DATA_DIR / "zone_labels.csv")
    rows = {
        "force_fs_khz": c.signal.force_fs_hz / 1000, "accel_fs_khz": c.signal.acceleration_fs_hz / 1000,
        "encoder_fs_khz": c.signal.force_fs_hz / c.signal.encoder_decimation / 1000,
        "feed_mm_min": c.process.feed_rate_mm_min, "tool_d_mm": c.process.tool_diameter_mm,
        "flutes": c.process.tooth_count, "wall_mm": c.process.wall_path_length_mm,
        "zones": c.process.zone_count, "exclusion_mm": c.process.end_exclusion_mm,
        "exclusion_s": c.process.end_exclusion_s, "window_ms": c.signal.window_ms,
        "windows_per_zone": c.signal.windows_per_zone, "context_s": c.signal.context_s,
        "context_bin_hz": 1 / c.signal.context_s, "label_nperseg": c.labels.label_nperseg,
        "label_bin_hz": c.signal.force_fs_hz / c.labels.label_nperseg,
        "band_low_khz": c.signal.chatter_band_hz[0] / 1000, "band_high_khz": c.signal.chatter_band_hz[1] / 1000,
        "tol_hz": c.signal.harmonic_tolerance_hz, "mode1_khz": c.signal.structural_modes_hz[0] / 1000,
        "mode2_khz": c.signal.structural_modes_hz[1] / 1000, "mode_hw_hz": c.signal.mode_half_width_hz,
        "welch_segment": c.signal.welch_segment, "wavelet_scales": c.signal.wavelet_scales,
        "features_per_channel": 10, "n_features": 62, "epochs": c.train.epochs, "lr": c.train.learning_rate,
        "weight_decay": c.train.weight_decay, "hidden": c.train.hidden_dim, "small_hidden": c.train.small_hidden_dim,
        "heads": c.train.attention_heads, "dropout": c.train.dropout, "n_seeds": len(c.train.seeds),
        "bootstrap": c.train.bootstrap_draws, "permutations": c.train.permutation_draws,
        "alignment_r_min": c.cut.alignment_ok_r, "n_trials_recorded": 21, "n_trials_force": len(s),
        "n_trials_used": len(inc), "n_zones": len(lab),
        "cut_s_min": inc.cut_duration_s.min(), "cut_s_max": inc.cut_duration_s.max(),
        "cut_mm_min": inc.cut_length_mm.min(), "cut_mm_max": inc.cut_length_mm.max(),
        "zone_s_min": lab.zone_duration_s.min(), "zone_s_max": lab.zone_duration_s.max(),
        "rpm_low_measured_mean": inc[inc.spindle_rpm_measured < 1250].spindle_rpm_measured.mean(),
        "rpm_high_measured_mean": inc[inc.spindle_rpm_measured >= 1250].spindle_rpm_measured.mean(),
        "rpm_low_nominal": 1000, "rpm_high_nominal": 1400,
        "comb_contrast_min_db": inc.spindle_comb_contrast_db.min(),
        "align_r_min": inc.accel_alignment_r.min(), "align_r_max": inc.accel_alignment_r.max(),
        "n_align_flagged": int((~inc.accel_ok.astype(bool)).sum()),
        "depth_min_mm": 0.5, "depth_max_mm": 1.5, "depth_step_mm": 0.25,
        "rf_trees": 300, "comb_low_hz": 12, "comb_high_hz": 30, "comb_harmonics": 60,
        "comb_grid_hz": 0.0005, "comb_refine_hz": 0.00002,
        "harmonic_number_min": int(lab.harmonic_number.min()), "harmonic_number_max": int(lab.harmonic_number.max()),
        "freq_precision_needed_hz": c.signal.harmonic_tolerance_hz / lab.harmonic_number.max(),
        "decision_threshold": 0.5, "ci_level_pct": 95, "alpha": 0.05, "small_gnn_param_limit": 5000,
        "min_confirmed_trials_per_material": 4, "n_confirmed_material_trials_with_force": 0,
        "speed_excess_low_pct": 100 * (inc[inc.spindle_rpm_measured < 1250].spindle_rpm_measured.mean() / 1000 - 1),
        "speed_excess_high_pct": 100 * (inc[inc.spindle_rpm_measured >= 1250].spindle_rpm_measured.mean() / 1400 - 1),
    }
    pd.DataFrame([dict(name=k, value=v) for k, v in rows.items()]).to_csv(RESULTS_DIR / "constants.csv", index=False)


def main() -> None:
    rng = np.random.default_rng(0)
    label_tables()
    constants()
    pred = pd.read_csv(RESULTS_DIR / "predictions.csv")

    metrics, folds, tests = [], [], []
    zone_tables = {}
    for (exp, model, variant), f in pred.groupby(["experiment", "model", "variant"], sort=False):
        z = zones(f)
        zone_tables[(exp, model, variant)] = z
        metrics.append(dict(experiment=exp, threshold_db=f.threshold_db.iloc[0], model=model, variant=variant,
                            **summarise(z, rng)))
        for t, g in z.groupby("trial_id"):
            folds.append(dict(experiment=exp, model=model, variant=variant, trial_id=t, n_positive=int(g.y.sum()),
                              mean_probability=g.probability.mean(), auc=auc(g.y.to_numpy(int), g.probability.to_numpy()),
                              no_positive_training=bool(g.no_positive_training.max())))
    m = pd.DataFrame(metrics)
    m["model_rank"] = m.model.map({k: i for i, k in enumerate(MODEL_ORDER)})
    m.sort_values(["experiment", "variant", "model_rank"]).drop(columns="model_rank").to_csv(
        RESULTS_DIR / "metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(RESULTS_DIR / "per_fold.csv", index=False)

    pairs = []
    for exp in ("main_15db", "alignment_15db"):
        for mdl in MODEL_ORDER[1:]:
            pairs.append((exp, (mdl, "full"), (exp, "logreg", "full")))
    for exp_ab in ("ablation_15db",):
        for mdl, var in (("gat", "no_edges"), ("gcn", "no_edges"), ("small_gnn", "no_edges"),
                         ("gat", "no_force"), ("gat", "no_acceleration"),
                         ("logreg", "no_force"), ("logreg", "no_acceleration")):
            pairs.append(("main_15db", (mdl, "full"), (exp_ab, mdl, var)))
    pairs += [("main_15db", ("gat", "full"), ("main_15db", "fusion_mlp", "full")),
              ("main_15db", ("gcn", "full"), ("main_15db", "fusion_mlp", "full")),
              ("main_15db", ("small_gnn", "full"), ("main_15db", "fusion_mlp", "full"))]
    for exp, (ma, va), (exp_b, mb, vb) in pairs:
        ka, kb = (exp, ma, va), (exp_b, mb, vb)
        if ka in zone_tables and kb in zone_tables:
            tests.append(dict(experiment_a=exp, model_a=ma, variant_a=va, experiment_b=exp_b, model_b=mb,
                              variant_b=vb, **paired_test(zone_tables[ka], zone_tables[kb], rng)))
    pd.DataFrame(tests).to_csv(RESULTS_DIR / "paired_tests.csv", index=False)

    loss = pd.read_csv(RESULTS_DIR / "training_loss.csv")
    last = loss[loss.epoch == loss.epoch.max()].groupby(["experiment", "model", "variant"]).train_loss.mean()
    prev = loss[loss.epoch == loss.epoch.max() - 5].groupby(["experiment", "model", "variant"]).train_loss.mean()
    first = loss[loss.epoch == 1].groupby(["experiment", "model", "variant"]).train_loss.mean()
    conv = pd.DataFrame({"loss_epoch1": first, "loss_last_minus5": prev, "loss_last": last}).reset_index()
    conv["ratio_first_to_last"] = conv.loss_epoch1 / conv.loss_last
    conv["relative_change_last5_pct"] = 100 * (conv.loss_last_minus5 - conv.loss_last) / conv.loss_last_minus5
    conv.to_csv(RESULTS_DIR / "training_convergence.csv", index=False)
    loss.groupby(["experiment", "model", "variant", "epoch"]).train_loss.mean().reset_index().to_csv(
        RESULTS_DIR / "training_loss_curves.csv", index=False)
    print(m[["experiment", "model", "variant", "n_positive", "auc", "auc_ci_low", "auc_ci_high",
             "null_auc_q025", "null_auc_q975", "perm_p_auc", "average_precision", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
