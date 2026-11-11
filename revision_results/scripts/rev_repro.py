"""
rev_repro.py - Step 7: reproducibility package (local files only; nothing is uploaded).

Writes revision_results/oof_predictions/*.csv, requirements.txt, CONFIG.md, SENSORS.md and copies
audit/trial_mapping_evidence.md into revision_results/audit/.
"""
import dataclasses
import platform
import shutil
from importlib.metadata import version

import pandas as pd

import rev_common as C

REV = C.REV


def oof():
    out = REV / "oof_predictions"
    out.mkdir(exist_ok=True)
    n = 0

    def wide(p, path, source):
        w = p.pivot_table(index=["trial_id", "zone_id", "y"], columns="seed", values="probability").reset_index()
        w.columns = ["trial_id", "zone_id", "label"] + [f"p_seed{s}" for s in w.columns[3:]]
        w["p_seed_mean"] = w[[c for c in w.columns if c.startswith("p_seed") and c != "p_seed_mean"]].mean(1)
        w["source"] = source
        w.to_csv(path, index=False)

    for f in sorted(C.RAW.glob("*_F[0-9].csv")):
        wide(pd.read_csv(f), out / f.name, "revision run (revision_results/scripts/rev_train.py)"); n += 1
    orig = pd.read_csv(C.ROOT / "results" / "predictions.csv")
    for m in C.MODEL_ORDER:
        if not (C.RAW / f"{m}_F0.csv").exists():
            p = orig[(orig.experiment == "main_15db") & (orig.model == m) & (orig.variant == "full")]
            wide(p, out / f"{m}_F0.csv", "manuscript run results/predictions.csv (main_15db, full)"); n += 1
    print("oof files:", n)


def requirements():
    pk = ["numpy", "pandas", "scipy", "scikit-learn", "PyWavelets", "matplotlib", "torch", "statsmodels", "shap",
          "python-docx"]
    lines = [f"# Python {platform.python_version()} ({platform.system()} {platform.release()})",
             "# Pinned versions actually imported by src/, make_figures.py and revision_results/ scripts.",
             "# python-docx is only needed to rebuild the manuscript (paper/build_manuscript.py)."]
    for p in pk:
        try:
            lines.append(f"{p}=={version(p)}")
        except Exception:
            lines.append(f"# {p}: not installed")
    (REV / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_md():
    c = C.CONFIG
    L = ["# CONFIG - every setting used for the revision analyses", "",
         "Generated from `src/config.py` (`CONFIG`) and the revision scripts. Nothing was changed relative to the "
         "manuscript run.", ""]
    for sec in dataclasses.fields(c):
        L += [f"## {sec.name}", "", "| setting | value |", "|---|---|"]
        obj = getattr(c, sec.name)
        for f in dataclasses.fields(obj):
            L.append(f"| {f.name} | {getattr(obj, f.name)} |")
        L.append("")
    L += ["## Training details (src/run_experiments.py, unchanged)", "",
          "| setting | value |", "|---|---|",
          "| outer CV | leave-one-trial-out, 16 folds, no inner validation, no early stopping |",
          f"| epochs (deep) | {c.train.epochs} (fixed) |",
          f"| optimiser | Adam, lr {c.train.learning_rate}, weight_decay {c.train.weight_decay} (Adam's L2 term) |",
          "| loss | BCE with logits, pos_weight = negatives/positives of the training trials |",
          "| batch | one trial graph (3 zones) per step, trial order shuffled each epoch (np.random) |",
          "| gradient clipping | global norm 5.0 |",
          "| deep input scaling | per-feature z-score from training trials; raw windows divided by the training std |",
          "| logistic regression | StandardScaler + LogisticRegression(class_weight='balanced', max_iter=5000), lbfgs |",
          "| random forest | 300 trees, class_weight='balanced', random_state=seed, n_jobs=1, no scaling |",
          "| SVM | StandardScaler + SVC(RBF, C=1, gamma='scale', class_weight='balanced', probability=True, random_state=seed) |",
          "| classical input | window-mean (16 windows) of the selected feature columns |",
          f"| model seeds | {list(c.train.seeds)} (torch, numpy, random; RF/SVM random_state) |",
          f"| torch threads | {c.train.torch_threads} per worker |",
          "| label | τ = 15 dB, ±2 Hz around measured spindle-rotation harmonics, 2.5–5.5 kHz, nperseg 32768 |",
          "| zones | 3 per trial, 5 mm removed at cut entry/exit; 16 windows of 100 ms per zone |", "",
          "## Statistics (revision)", "", "| setting | value |", "|---|---|",
          "| seed averaging | probabilities averaged over the 5 seeds per zone before any metric |",
          f"| trial bootstrap | {c.train.bootstrap_draws} resamples of whole trials, np.random.default_rng(0) per (analysis, model, config) |",
          f"| zone / trial-block / within-trial permutation (scores fixed) | {c.train.permutation_draws}, default_rng(0) |",
          "| paired test | trial bootstrap of ΔAUC + trial-wise score-swap permutation (2000), default_rng(0) |",
          "| retraining permutation (Step 2c) | trial-block, permutation b uses default_rng([0, b]); counts in permutation_cluster.csv |",
          "| permutation p | (1 + #{null at least as extreme}) / (N + 1); two-sided uses |AUC − 0.5| as in src/evaluate.py |",
          "| multiple comparisons | Holm–Bonferroni, statsmodels multipletests(method='holm') |",
          "| calibration | 5 equal-count bins; logistic recalibration via statsmodels Logit on logit(p) |",
          "| permutation importance | 20 repeats, permutations from default_rng(0), shared across seeds |"]
    (REV / "CONFIG.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def sensors_md():
    s = pd.read_csv(C.ROOT / "data" / "processed" / "dataset_summary.csv")
    fi = pd.read_csv(C.ROOT / "data" / "reference" / "force_recording_index.csv")
    L = ["# SENSORS", "",
         "Filled only from the code and data files in this repository. \"not recorded\" = not found in the archive.", "",
         "| channel | sensor | sampling rate | acquisition computer | units | calibration information | alignment method |",
         "|---|---|---|---|---|---|---|"]
    fs_force = ", ".join(sorted(set(fi.sampling_rate_hz.astype(int).astype(str))))
    for ch in ("Fx", "Fy", "Fz"):
        L.append(f"| {ch} | Kistler 9257B dynamometer (src/config.py, README.md; DynoWare .dwd/.csv export) | "
                 f"{fs_force} Hz (data/reference/force_recording_index.csv) | force acquisition PC (DynoWare); "
                 "identity not recorded | N (CSV units row) | not recorded (charge-amplifier settings not in the "
                 "archive) | reference clock |")
    for ch in ("Acc-X", "Acc-Y", "Acc-Z"):
        L.append(f"| {ch} | triaxial accelerometer; model not recorded | 25 000 Hz (from CSV time column, "
                 "checked to ±1 Hz in prepare_dataset.py) | separate acquisition computer, not synchronised with "
                 "the force PC; identity not recorded | g (CSV units row) | not recorded (sensitivity and mounting "
                 "not recorded) | log-RMS envelope vs force-cut step template, max correlation, lag ≤ 6 s, 10 ms "
                 "bins (prepare_dataset.align_acceleration) |")
    L += ["", "## Alignment per trial (data/processed/dataset_summary.csv)", "",
          "| trial | included | accel lag (s) | alignment r | r ≥ 0.7 | measured spindle rpm |", "|---|---|---|---|---|---|"]
    for r in s.itertuples():
        L.append(f"| {r.trial_id} | {r.included} | {r.accel_lag_s if pd.notna(r.accel_lag_s) else 'not computed'} | "
                 f"{r.accel_alignment_r if pd.notna(r.accel_alignment_r) else 'not computed'} | "
                 f"{r.accel_ok if pd.notna(r.accel_ok) else ''} | "
                 f"{r.spindle_rpm_measured if pd.notna(r.spindle_rpm_measured) else ''} |")
    (REV / "SENSORS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def audit_copy():
    src = C.ROOT / "audit" / "trial_mapping_evidence.md"
    if src.exists():
        (REV / "audit").mkdir(exist_ok=True)
        shutil.copy2(src, REV / "audit" / src.name)


if __name__ == "__main__":
    oof(); requirements(); config_md(); sensors_md(); audit_copy()
    print("repro package written")
