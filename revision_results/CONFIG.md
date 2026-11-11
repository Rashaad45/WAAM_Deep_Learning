# CONFIG - every setting used for the revision analyses

Generated from `src/config.py` (`CONFIG`) and the revision scripts. Nothing was changed relative to the manuscript run.

## process

| setting | value |
|---|---|
| feed_rate_mm_min | 80.0 |
| tooth_count | 4 |
| tool_diameter_mm | 10.0 |
| wall_path_length_mm | 30.0 |
| zone_count | 3 |
| end_exclusion_mm | 5.0 |

## signal

| setting | value |
|---|---|
| force_fs_hz | 25000.0 |
| acceleration_fs_hz | 25000.0 |
| window_ms | 100.0 |
| windows_per_zone | 16 |
| context_s | 0.5 |
| encoder_decimation | 2 |
| structural_modes_hz | (3500.0, 4300.0) |
| mode_half_width_hz | 150.0 |
| chatter_band_hz | (2500.0, 5500.0) |
| harmonic_tolerance_hz | 2.0 |
| welch_segment | 1024 |
| wavelet | cmor1.5-1.0 |
| wavelet_scales | 24 |

## cut

| setting | value |
|---|---|
| envelope_bin_s | 0.01 |
| baseline_s | 1.0 |
| threshold_fraction | 0.3 |
| max_gap_s | 1.0 |
| min_cut_s | 15.0 |
| max_align_lag_s | 6.0 |
| alignment_ok_r | 0.7 |

## labels

| setting | value |
|---|---|
| thresholds_db | (15.0, 20.0, 25.0) |
| primary_threshold_db | 20.0 |
| label_nperseg | 32768 |

## train

| setting | value |
|---|---|
| epochs | 30 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| hidden_dim | 64 |
| small_hidden_dim | 16 |
| attention_heads | 4 |
| temporal_kernel | 3 |
| dropout | 0.25 |
| seeds | (42, 43, 44, 45, 46) |
| bootstrap_draws | 2000 |
| permutation_draws | 2000 |
| torch_threads | 1 |

## Training details (src/run_experiments.py, unchanged)

| setting | value |
|---|---|
| outer CV | leave-one-trial-out, 16 folds, no inner validation, no early stopping |
| epochs (deep) | 30 (fixed) |
| optimiser | Adam, lr 0.001, weight_decay 0.0001 (Adam's L2 term) |
| loss | BCE with logits, pos_weight = negatives/positives of the training trials |
| batch | one trial graph (3 zones) per step, trial order shuffled each epoch (np.random) |
| gradient clipping | global norm 5.0 |
| deep input scaling | per-feature z-score from training trials; raw windows divided by the training std |
| logistic regression | StandardScaler + LogisticRegression(class_weight='balanced', max_iter=5000), lbfgs |
| random forest | 300 trees, class_weight='balanced', random_state=seed, n_jobs=1, no scaling |
| SVM | StandardScaler + SVC(RBF, C=1, gamma='scale', class_weight='balanced', probability=True, random_state=seed) |
| classical input | window-mean (16 windows) of the selected feature columns |
| model seeds | [42, 43, 44, 45, 46] (torch, numpy, random; RF/SVM random_state) |
| torch threads | 1 per worker |
| label | τ = 15 dB, ±2 Hz around measured spindle-rotation harmonics, 2.5–5.5 kHz, nperseg 32768 |
| zones | 3 per trial, 5 mm removed at cut entry/exit; 16 windows of 100 ms per zone |

## Statistics (revision)

| setting | value |
|---|---|
| seed averaging | probabilities averaged over the 5 seeds per zone before any metric |
| trial bootstrap | 2000 resamples of whole trials, np.random.default_rng(0) per (analysis, model, config) |
| zone / trial-block / within-trial permutation (scores fixed) | 2000, default_rng(0) |
| paired test | trial bootstrap of ΔAUC + trial-wise score-swap permutation (2000), default_rng(0) |
| retraining permutation (Step 2c) | trial-block, permutation b uses default_rng([0, b]); counts in permutation_cluster.csv |
| permutation p | (1 + #{null at least as extreme}) / (N + 1); two-sided uses |AUC − 0.5| as in src/evaluate.py |
| multiple comparisons | Holm–Bonferroni, statsmodels multipletests(method='holm') |
| calibration | 5 equal-count bins; logistic recalibration via statsmodels Logit on logit(p) |
| permutation importance | 20 repeats, permutations from default_rng(0), shared across seeds |
