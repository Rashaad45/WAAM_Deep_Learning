# revision_results — faculty-revision analyses

Everything here was produced by the scripts in `revision_results/scripts/` from the unchanged pipeline in
`src/` (same labels, zones, windows, features, architectures, hyperparameters, 30 epochs, seeds 42–46, LOTO).
The manuscript outputs in `results/` were not modified. Start with `RESULTS_SUMMARY.md`.

## Setup

```
python -m pip install -r revision_results/requirements.txt
```

Processed inputs `data/processed/trial_XX.npz`, `data/processed/dataset_summary.csv` and `data/zone_labels.csv`
must exist. To rebuild them from the raw archive, run `python run_pipeline.py prepare`.

Run every command below from `revision_results/scripts/` unless stated otherwise.

## One command per table / figure

| Output | Command |
|---|---|
| Step 0 check (LR/RF reproduce 0.69/0.74, 25/48) | `python rev_train.py --models logreg rf --configs F0` then `python rev_analyse.py repro` → `reproduction_check.csv` |
| Out-of-fold predictions, classical models, F0–F8 | `python rev_train.py --models logreg rf svm --configs F0 F1 F2 F3 F4 F5 F6 F7 F8` |
| Out-of-fold predictions, deep models | `python rev_train.py --models gat fusion_mlp bilstm --configs F0 F1 F2 F3` and `python rev_train.py --models small_gnn gcn --configs F1` |
| `ablation_features.csv`, `ablation_features_paired.csv`, `single_feature_auc.csv` | `python rev_analyse.py 1` |
| `permutation_cluster.csv` (a, b) | `python rev_analyse.py 2` |
| `permutation_cluster.csv` (c, retraining; needs `zone_mean_features.csv` from `rev_importance.py`) | `python rev_perm_retrain_np.py run --models logreg --configs F0 F1 --perms 500 --lr_one_seed --workers 4` and `python rev_perm_retrain_np.py run --models rf --configs F0 F1 --perms 200 --workers 9` (resumable; this run stopped at 188), then `python rev_perm_retrain_np.py summarise` (draws in `perm_retrain_draws.csv`). `rev_perm_retrain.py` is the earlier torch-based runner that ran out of memory; it is kept but was not used for any result. |
| `holm_adjusted.csv` | `python rev_analyse.py 3` |
| `discrimination_levels.csv`, `within_trial_per_trial.csv` | `python rev_analyse.py 4` |
| `calibration.csv`, `calibration_bins.csv` | `python rev_analyse.py 5` |
| `importance.csv`, `shap_rf_oof.csv` | `python rev_importance.py` |
| All revision figures + re-export of the 5 manuscript figures | `python ../make_revision_figures.py` |
| `oof_predictions/`, `requirements.txt`, `CONFIG.md`, `SENSORS.md` | `python rev_repro.py` |
| `summary_tables.md`, `seed_spread.csv` (every table in RESULTS_SUMMARY.md) | `python rev_tables.py` |
| `RESULTS_SUMMARY.md` | PowerShell: `Get-Content summary_head.md, ..\summary_tables.md, summary_tail.md -Encoding UTF8 \| Out-File ..\RESULTS_SUMMARY.md -Encoding utf8` (or `cat` the three files in bash) |

`python rev_analyse.py` with no argument runs the reproduction check and Steps 1–5 in order.

## Layout

- `oof_raw/` — long-format out-of-fold predictions (one row per seed × zone) and deep-model training losses.
- `oof_predictions/` — one CSV per model × configuration: trial, zone, label, per-seed probabilities,
  seed-mean probability, source.
- `figures/` — `fig_*.png/pdf` (300 dpi); `figures/paper_figures/` holds the five manuscript figures re-exported.
- `CONFIG.md`, `SENSORS.md`, `PIPELINE_MAP.md`, `audit/trial_mapping_evidence.md`.
- `.zenodo.json`, `CITATION.cff` — metadata templates. Nothing has been uploaded.

## Determinism

The classical models and the deep models with the existing configuration reproduce `results/predictions.csv`
bit for bit on the machine used (see `reproduction_check.csv`). Every bootstrap and permutation uses
`numpy.random.default_rng(0)`, created fresh for each (analysis, model, configuration), so re-running one
analysis does not change the others. Step 2(c) permutation b uses `default_rng([0, b])`.
