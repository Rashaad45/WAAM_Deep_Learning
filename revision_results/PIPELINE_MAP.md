# Pipeline map (Step 0)

All paths relative to the repository root. Confirmed by reading the code on 2026-11-10.

## Entry points

| Stage | File | Function(s) |
|---|---|---|
| Orchestration | `run_pipeline.py` | `main`, `STEPS` (prepare → mapping → train → evaluate → figures → paper) |
| Settings | `src/config.py` | `CONFIG` (`ProcessConfig`, `SignalConfig`, `CutDetectionConfig`, `LabelConfig`, `TrainConfig`) |
| Raw I/O, cut detection, alignment | `src/prepare_dataset.py` | `read_force`, `read_acceleration`, `detect_cut`, `align_acceleration` |
| Spindle frequency | `src/prepare_dataset.py` | `measure_spindle_hz` (harmonic comb on force spectrum) |
| Zones and windows | `src/prepare_dataset.py` | `zone_spans`, `window_starts`, `cut_windows`, `TrialProcessor.run` |
| **Labelling** | `src/signal_features.py` | `zone_spectrum_peak` (3-axis summed force Welch PSD, nperseg 32768, 2.5–5.5 kHz), `chatter_label` (peak not within ±2 Hz of a spindle harmonic AND prominence ≥ τ), `harmonic_mask`; called in `TrialProcessor.run` → `data/zone_labels.csv` |
| **Feature extraction** | `src/signal_features.py` | `WindowFeatureExtractor.channel_features`, `.window_features`, `.chatter_indicator`, `.wavelet_features`, `.band_centroid_khz`; cached to `data/processed/trial_XX.npz` (`features` [3, 16, 60]) |
| Graph/node assembly | `src/graph_data.py` | `TrialGraphBuilder.load_all` (adds spindle speed + zone position → 62), `apply_variant` (existing modality/edge ablations; zero-out) |
| **LOTO training loop** | `src/run_experiments.py` | `run_job`, `loto_folds`, `classical_model`, `Standardiser`, `bce_loss`, `zone_feature_matrix`; models in `src/models.py` (`build_model`) |
| **Statistics** | `src/evaluate.py` | `zones` (seed-average), `summarise` (pooled AUC/AP, trial bootstrap CI, zone-label permutation null), `paired_test` (paired trial bootstrap + trial-swap permutation), `bootstrap_indices`, `fast_auc` |
| **Figures** | `make_figures.py` | `fig1_mapping`, `fig2_prominence`, `fig3_benchmark`, `fig4_paired`, `fig5_loss` |
| Manuscript | `paper/build_manuscript.py` | Table 3 = `metrics.csv` (main_15db), Table 4 = 18 rows of `paired_tests.csv` (8 × "model − LR", 3 × "graph − fusion MLP", 7 × ablation) |

Revision code: `revision_results/scripts/` (`rev_common.py` reuses the functions above unchanged; it
re-implements only the per-seed body of `run_job` so that columns can be dropped, with identical code).

## Feature column order (62 inputs per window)

Source: `prepare_dataset.py` L203–205 (force features, then acceleration features),
`signal_features.py` L89 (per-channel order) and L94 (channel concatenation), `graph_data.py` L49–50
(kinematics appended). Force channels are the Kistler `Fx, Fy, Fz` columns (`read_force`); the
acceleration channels are columns 1–3 of the accelerometer CSV (`Acc-X, Acc-Y, Acc-Z`, in g).

Index = `channel × 10 + feature`:

| feature k | name | window |
|---|---|---|
| 0 | RMS | 100 ms |
| 1 | peak-to-peak | 100 ms |
| 2 | kurtosis | 100 ms |
| 3 | crest factor | 100 ms |
| 4 | log mode energy 3.5 kHz ± 150 Hz | 100 ms, Welch 1024 |
| 5 | log mode energy 4.3 kHz ± 150 Hz | 100 ms, Welch 1024 |
| 6 | chatter indicator (non-synchronous share of 2.5–5.5 kHz power, ±2 Hz spindle harmonics masked) | 0.5 s context |
| 7 | wavelet entropy | 100 ms |
| 8 | log wavelet energy | 100 ms |
| 9 | 2.5–5.5 kHz spectral centroid (kHz) | 100 ms |

| channel c | signal | indices |
|---|---|---|
| 0 | Fx | 0–9 |
| 1 | Fy | 10–19 |
| 2 | Fz | 20–29 |
| 3 | Acc-X | 30–39 |
| 4 | Acc-Y | 40–49 |
| 5 | Acc-Z | 50–59 |
| – | measured spindle rpm / 1000 | 60 |
| – | zone-centre position (fraction of detected cut) | 61 |

Chatter-indicator columns: 6, 16, 26 (force), 36, 46, 56 (acceleration).
Force chatter-band columns removed in F2: 4, 5, 6, 9, 14, 15, 16, 19, 24, 25, 26, 29.
The full name list is `FEATURE_NAMES` in `revision_results/scripts/rev_common.py`.

## Step 0.3 sanity check (τ = 15 dB)

Re-ran LR and RF with the current code (`rev_train.py --models logreg rf --configs F0`):
pooled AUC LR = 0.6922, RF = 0.7426, 25/48 unstable zones. The seed-level probabilities are
**bit-identical** to `results/predictions.csv` (max |Δp| = 0.0), as are the SVM ones. The check passes.
