
## Figures (`figures/`, 300 dpi PNG + PDF, all made by `make_revision_figures.py`)

| File | Caption |
|---|---|
| `fig_confusion` | Confusion matrices (counts, threshold p ≥ 0.5 on the seed-averaged OOF probability, τ = 15 dB labels) for logistic regression, random forest, BiLSTM and ST-GAT at F0. |
| `fig_roc_pr` | ROC (top) and precision–recall (bottom) curves of the same four models; shaded bands are trial-bootstrap 95% pointwise ranges (2000 resamples, rng 0); PR bands use interpolated precision; dotted lines mark chance (ROC) and prevalence 0.52 (PR). AUC/AP and bootstrap draws used in `fig_roc_pr_values.csv`. |
| `fig_importance` | Trial-aware permutation importance for LR and RF (F0): top 15 individual features (left) and feature family × modality groups (right). Markers: permuted across all held-out zones; bars: permuted within the held-out trial only. |
| `fig_shap` | RF TreeSHAP beeswarm, 15 features with the largest mean |SHAP|: each held-out zone explained by its own fold model, averaged over 5 seeds (48 zones × 62 features in `shap_rf_oof.csv`; shap 0.51.0). No CI feature appears among the 15. |
| `fig_ablation` | Pooled AUC with trial-bootstrap 95% CI for every feature configuration: classical models F0–F8 (left), deep models F0–F3 (right; their raw-window encoders are unchanged by the ablation). Grey band: mean 95% range of the zone-label permutation null. |
| `fig_calibration` | Reliability diagrams (5 equal-count bins) with trial-bootstrap 95% ranges of the observed frequency; panel titles give Brier, ECE and calibration slope (`calibration.csv`, `calibration_bins.csv`). |
| `fig_levels` | Pooled, within-trial (mean over the 8 mixed-label trials) and trial-level (≥ 1 unstable zone) AUC for all models at F0 and LR, RF, ST-GAT at F1 (`discrimination_levels.csv`). |
| `paper_figures/fig1_mapping` … `fig5_training_loss` | Re-export of the five manuscript figures by the unchanged `make_figures.py` (env `WAAM_FIGURES_DIR`), from `results/`. |

## Run log, versions, failures and skipped items

* **Software:** Python 3.11.4 (Windows), numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, PyWavelets 1.9.0,
  matplotlib 3.11.2, torch 2.6.0+cu124 (run on CPU), statsmodels 0.15.0 (installed during this run with pip; it was not
  present), shap 0.51.0 (`requirements.txt`). Seeds: training 42–46 (unchanged); all statistics `default_rng(0)`;
  Step 2(c) permutation b `default_rng([0, b])`.
* **Run times:** see the run-time table above (`run_times.csv`, CPU time). Wall clock: classical models finished
  14:44:54–14:46:39 (`run_times.csv`); deep ablations (60 jobs, 8 workers) 83.7 min (`deep_train_log.txt`);
  Step 2(c) LR and RF in parallel worker pools (`perm_lr_log.txt`, `perm_rf_log.txt`).
* **Failures (all recovered; none changed a result):** (1) the first deep-ablation launch crashed with
  "DefaultCPUAllocator: not enough memory" (7.3 GB RAM) and saved nothing; the runner was changed to checkpoint each
  (model, config, seed) job (`oof_raw/jobs/`, not in the zip) and relaunched with 8 workers. (2) The first Step 2(c)
  runner imported torch and failed with "BrokenProcessPool" / WinError 1455 (paging file too small) while deep training
  ran; it was replaced by the torch-free `rev_perm_retrain_np.py`, whose observed (unpermuted) AUCs equal the
  benchmark exactly (asserted). (3) The dataviz palette validator needs node, which is not installed; figure colours use
  the validated reference categorical order instead.
* **Reduced scope:** Step 2(c) RF reached 188 permutations per configuration, not 500: it ran with a
  200-permutation target to fit the deadline and was then stopped by Claude Code's low-memory guard at 17:26. Its p
  therefore has a resolution of 1/189 and should be read as approximate. LR Step 2(c) used one seed per permutation (LR is seed-invariant, `seed_spread.csv`).
* **Not run / skipped:** Step 8 (external turning dataset): `external/turning_chatter_mendeley/` does not exist, so it
  was skipped (`external_E1.md`); nothing was downloaded. 1D-CNN feature ablation: not applicable (raw windows only).
  Small GNN / ST-GCN at F2–F3 and their F0 retrain: not run (optional). Nothing was uploaded; `.zenodo.json` and
  `CITATION.cff` contain the placeholder title "TITLE" and co-author placeholders.
* **Reproducibility package:** `README.md` (one command per table and figure), `requirements.txt`, `CONFIG.md`,
  `SENSORS.md` (filled only from code/data; unknown items "not recorded"), `audit/trial_mapping_evidence.md`,
  `PIPELINE_MAP.md`, `oof_predictions/` (one CSV per model × configuration: per-seed probabilities and the seed mean),
  `oof_raw/`, `scripts/`.
