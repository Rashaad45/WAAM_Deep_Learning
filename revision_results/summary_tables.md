### Reproduction check (reproduction_check.csv)

| Model | Retrained here | max |Δp| vs results/predictions.csv | AUC this run | AUC manuscript |
|---|---|---|---|---|
| Logistic regression | True | 0 | 0.69 | 0.69 |
| Random forest | True | 0 | 0.74 | 0.74 |
| SVM (RBF) | True | 0 | 0.65 | 0.65 |
| 1D-CNN | False | – | – | 0.44 |
| BiLSTM | True | 0 | 0.73 | 0.73 |
| Fusion MLP | True | 0 | 0.71 | 0.71 |
| Small GNN | False | – | – | 0.72 |
| ST-GCN | False | – | – | 0.70 |
| ST-GAT | True | 0 | 0.69 | 0.69 |

### Seed dependence of the retrained probabilities (seed_spread.csv, from oof_raw/)

| Model | Config | seeds | max over zones of seed range of p |
|---|---|---|---|
| BiLSTM | F0 | 5 | 0.96 |
| BiLSTM | F1 | 5 | 0.89 |
| BiLSTM | F2 | 5 | 0.91 |
| BiLSTM | F3 | 5 | 0.87 |
| Fusion MLP | F0 | 5 | 0.86 |
| Fusion MLP | F1 | 5 | 0.78 |
| Fusion MLP | F2 | 5 | 0.7 |
| Fusion MLP | F3 | 5 | 0.93 |
| ST-GAT | F0 | 5 | 0.82 |
| ST-GAT | F1 | 5 | 0.99 |
| ST-GAT | F2 | 5 | 0.99 |
| ST-GAT | F3 | 5 | 0.85 |
| ST-GCN | F1 | 5 | 0.9 |
| Logistic regression | F0 | 5 | 0 |
| Logistic regression | F1 | 5 | 0 |
| Logistic regression | F2 | 5 | 0 |
| Logistic regression | F3 | 5 | 0 |
| Logistic regression | F4 | 5 | 0 |
| Logistic regression | F5 | 5 | 0 |
| Logistic regression | F6 | 5 | 0 |
| Logistic regression | F7 | 5 | 0 |
| Logistic regression | F8 | 5 | 0 |
| Random forest | F0 | 5 | 0.1 |
| Random forest | F1 | 5 | 0.14 |
| Random forest | F2 | 5 | 0.11 |
| Random forest | F3 | 5 | 0.11 |
| Random forest | F4 | 5 | 0.12 |
| Random forest | F5 | 5 | 0.11 |
| Random forest | F6 | 5 | 0.13 |
| Random forest | F7 | 5 | 0.11 |
| Random forest | F8 | 5 | 0.087 |
| Small GNN | F1 | 5 | 0.56 |
| SVM (RBF) | F0 | 5 | 0.088 |
| SVM (RBF) | F1 | 5 | 0.087 |
| SVM (RBF) | F2 | 5 | 0.089 |
| SVM (RBF) | F3 | 5 | 0.095 |
| SVM (RBF) | F4 | 5 | 0.081 |
| SVM (RBF) | F5 | 5 | 0.097 |
| SVM (RBF) | F6 | 5 | 0.12 |
| SVM (RBF) | F7 | 5 | 0.071 |
| SVM (RBF) | F8 | 5 | 0.1 |

### Step 1: feature ablation (ablation_features.csv, ablation_features_paired.csv, holm_adjusted.csv)

| Model | Config | n feat | AUC (95% CI) | AP (95% CI) | ΔAUC vs F0 (95% CI) | paired p | Holm p (per model) |
|---|---|---|---|---|---|---|---|
| Logistic regression | F0 | 62 | 0.69 (0.52 to 0.86) | 0.66 (0.48 to 0.91) | – | – | – |
| Logistic regression | F1 | 56 | 0.70 (0.53 to 0.86) | 0.74 (0.51 to 0.91) | +0.01 (-0.05 to 0.05) | 0.578 | 1.000 |
| Logistic regression | F2 | 50 | 0.67 (0.49 to 0.84) | 0.65 (0.49 to 0.89) | -0.02 (-0.09 to 0.03) | 0.350 | 1.000 |
| Logistic regression | F3 | 26 | 0.73 (0.56 to 0.88) | 0.75 (0.56 to 0.91) | +0.04 (-0.09 to 0.16) | 0.532 | 1.000 |
| Logistic regression | F4 | 36 | 0.70 (0.52 to 0.85) | 0.66 (0.46 to 0.90) | +0.01 (-0.08 to 0.09) | 0.857 | 1.000 |
| Logistic regression | F5 | 30 | 0.70 (0.49 to 0.87) | 0.67 (0.47 to 0.92) | +0.00 (-0.10 to 0.08) | 0.947 | 1.000 |
| Logistic regression | F6 | 32 | 0.77 (0.62 to 0.91) | 0.74 (0.55 to 0.94) | +0.08 (0.00 to 0.20) | 0.078 | 0.628 |
| Logistic regression | F7 | 32 | 0.71 (0.53 to 0.87) | 0.72 (0.51 to 0.92) | +0.02 (-0.06 to 0.08) | 0.582 | 1.000 |
| Logistic regression | F8 | 29 | 0.74 (0.58 to 0.90) | 0.81 (0.60 to 0.94) | +0.05 (-0.04 to 0.15) | 0.350 | 1.000 |
| Random forest | F0 | 62 | 0.74 (0.56 to 0.89) | 0.79 (0.59 to 0.92) | – | – | – |
| Random forest | F1 | 56 | 0.75 (0.57 to 0.90) | 0.80 (0.61 to 0.93) | +0.01 (-0.01 to 0.04) | 0.155 | 1.000 |
| Random forest | F2 | 50 | 0.72 (0.52 to 0.88) | 0.76 (0.54 to 0.91) | -0.02 (-0.08 to 0.04) | 0.412 | 1.000 |
| Random forest | F3 | 26 | 0.74 (0.55 to 0.89) | 0.77 (0.58 to 0.91) | +0.00 (-0.07 to 0.08) | 0.969 | 1.000 |
| Random forest | F4 | 36 | 0.75 (0.56 to 0.90) | 0.79 (0.59 to 0.93) | +0.00 (-0.05 to 0.06) | 0.863 | 1.000 |
| Random forest | F5 | 30 | 0.75 (0.57 to 0.90) | 0.80 (0.59 to 0.93) | +0.01 (-0.05 to 0.06) | 0.612 | 1.000 |
| Random forest | F6 | 32 | 0.76 (0.59 to 0.89) | 0.79 (0.62 to 0.93) | +0.01 (-0.07 to 0.11) | 0.811 | 1.000 |
| Random forest | F7 | 32 | 0.68 (0.47 to 0.86) | 0.76 (0.52 to 0.91) | -0.06 (-0.15 to -0.00) | 0.067 | 0.540 |
| Random forest | F8 | 29 | 0.71 (0.51 to 0.87) | 0.78 (0.55 to 0.92) | -0.04 (-0.11 to 0.02) | 0.218 | 1.000 |
| SVM (RBF) | F0 | 62 | 0.65 (0.47 to 0.83) | 0.68 (0.54 to 0.88) | – | – | – |
| SVM (RBF) | F1 | 56 | 0.63 (0.45 to 0.82) | 0.65 (0.50 to 0.87) | -0.02 (-0.07 to 0.02) | 0.433 | 1.000 |
| SVM (RBF) | F2 | 50 | 0.63 (0.45 to 0.81) | 0.66 (0.51 to 0.87) | -0.02 (-0.05 to 0.00) | 0.031 | 0.252 |
| SVM (RBF) | F3 | 26 | 0.66 (0.47 to 0.84) | 0.70 (0.55 to 0.87) | +0.01 (-0.06 to 0.08) | 0.753 | 1.000 |
| SVM (RBF) | F4 | 36 | 0.64 (0.47 to 0.82) | 0.66 (0.52 to 0.88) | -0.01 (-0.06 to 0.04) | 0.830 | 1.000 |
| SVM (RBF) | F5 | 30 | 0.61 (0.43 to 0.80) | 0.61 (0.45 to 0.87) | -0.03 (-0.13 to 0.03) | 0.394 | 1.000 |
| SVM (RBF) | F6 | 32 | 0.66 (0.48 to 0.84) | 0.69 (0.56 to 0.89) | +0.02 (-0.05 to 0.08) | 0.640 | 1.000 |
| SVM (RBF) | F7 | 32 | 0.62 (0.43 to 0.81) | 0.67 (0.53 to 0.87) | -0.03 (-0.09 to 0.02) | 0.373 | 1.000 |
| SVM (RBF) | F8 | 29 | 0.61 (0.42 to 0.80) | 0.64 (0.50 to 0.86) | -0.04 (-0.10 to 0.01) | 0.108 | 0.756 |
| BiLSTM | F0 | 62 | 0.73 (0.55 to 0.87) | 0.76 (0.54 to 0.92) | – | – | – |
| BiLSTM | F1 | 56 | 0.73 (0.55 to 0.89) | 0.77 (0.57 to 0.92) | +0.01 (-0.08 to 0.08) | 0.899 | 1.000 |
| BiLSTM | F2 | 50 | 0.69 (0.51 to 0.85) | 0.70 (0.50 to 0.90) | -0.04 (-0.11 to 0.02) | 0.240 | 0.721 |
| BiLSTM | F3 | 26 | 0.70 (0.50 to 0.85) | 0.75 (0.55 to 0.90) | -0.03 (-0.15 to 0.06) | 0.572 | 1.000 |
| Fusion MLP | F0 | 62 | 0.71 (0.51 to 0.88) | 0.68 (0.49 to 0.92) | – | – | – |
| Fusion MLP | F1 | 56 | 0.71 (0.52 to 0.87) | 0.67 (0.50 to 0.91) | +0.00 (-0.03 to 0.04) | 0.872 | 1.000 |
| Fusion MLP | F2 | 50 | 0.66 (0.47 to 0.83) | 0.64 (0.47 to 0.89) | -0.05 (-0.11 to -0.01) | 0.031 | 0.094 |
| Fusion MLP | F3 | 26 | 0.73 (0.55 to 0.87) | 0.73 (0.54 to 0.90) | +0.02 (-0.08 to 0.14) | 0.739 | 1.000 |
| Small GNN | F1 | 56 | 0.73 (0.54 to 0.88) | 0.71 (0.52 to 0.90) | – | – | – |
| ST-GCN | F1 | 56 | 0.68 (0.51 to 0.84) | 0.65 (0.47 to 0.90) | – | – | – |
| ST-GAT | F0 | 62 | 0.69 (0.50 to 0.84) | 0.66 (0.47 to 0.90) | – | – | – |
| ST-GAT | F1 | 56 | 0.69 (0.52 to 0.83) | 0.67 (0.48 to 0.90) | -0.00 (-0.04 to 0.03) | 0.908 | 1.000 |
| ST-GAT | F2 | 50 | 0.61 (0.43 to 0.79) | 0.64 (0.45 to 0.86) | -0.07 (-0.17 to 0.01) | 0.071 | 0.213 |
| ST-GAT | F3 | 26 | 0.69 (0.50 to 0.84) | 0.66 (0.49 to 0.88) | +0.00 (-0.10 to 0.10) | 1.000 | 1.000 |

### Single-feature AUC, requested features (single_feature_auc.csv)

| Feature | Sign | AUC (oriented, 95% CI) | zone-perm p |
|---|---|---|---|
| Fz_mode2_energy_4.3kHz | - | 0.77 (0.57 to 0.93) | 0.001 |
| Fx_mode2_energy_4.3kHz | - | 0.73 (0.51 to 0.90) | 0.005 |
| Fy_mode2_energy_4.3kHz | - | 0.73 (0.51 to 0.89) | 0.004 |
| Fx_mode1_energy_3.5kHz | + | 0.63 (0.40 to 0.80) | 0.127 |
| Fz_mode1_energy_3.5kHz | + | 0.63 (0.39 to 0.80) | 0.131 |
| Fy_chatter_indicator | - | 0.61 (0.38 to 0.84) | 0.213 |
| Fy_mode1_energy_3.5kHz | + | 0.59 (0.36 to 0.77) | 0.301 |
| AccX_chatter_indicator | - | 0.58 (0.34 to 0.80) | 0.361 |
| Fx_chatter_indicator | - | 0.58 (0.36 to 0.81) | 0.373 |
| AccY_chatter_indicator | - | 0.57 (0.32 to 0.81) | 0.412 |
| AccZ_chatter_indicator | - | 0.57 (0.33 to 0.80) | 0.407 |
| Fz_chatter_indicator | - | 0.56 (0.35 to 0.76) | 0.490 |

### Single-feature AUC, ten highest of the other features (single_feature_auc.csv)

| Feature | Sign | AUC (oriented, 95% CI) | zone-perm p |
|---|---|---|---|
| Fx_crest | - | 0.83 (0.68 to 0.94) | 0.000 |
| Fz_band_centroid | - | 0.82 (0.66 to 0.94) | 0.000 |
| AccX_kurtosis | + | 0.82 (0.68 to 0.93) | 0.000 |
| AccY_kurtosis | + | 0.81 (0.66 to 0.93) | 0.001 |
| AccZ_kurtosis | + | 0.81 (0.65 to 0.93) | 0.001 |
| AccY_crest | + | 0.81 (0.65 to 0.93) | 0.001 |
| AccZ_crest | + | 0.81 (0.65 to 0.93) | 0.001 |
| Fy_crest | + | 0.80 (0.64 to 0.93) | 0.000 |
| AccX_crest | + | 0.80 (0.64 to 0.92) | 0.001 |
| AccZ_band_centroid | - | 0.80 (0.63 to 0.95) | 0.000 |

### Single-feature AUC, kinematic inputs (single_feature_auc.csv)

| Feature | Sign | AUC (oriented, 95% CI) | zone-perm p |
|---|---|---|---|
| zone_position | + | 0.59 (0.45 to 0.73) | 0.309 |
| spindle_speed | - | 0.51 (0.27 to 0.74) | 0.945 |

### Step 2 (a, b): permutation nulls with scores fixed (permutation_cluster.csv, holm_adjusted.csv)

| Model | Config | AUC | zone-level null 95% | p zone | (a) trial-block null 95% | p block | p block Holm | (b) within-trial null 95% | within-trial null mean | p within (one-sided) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | F0 | 0.69 | 0.33 to 0.67 | 0.026 | 0.31 to 0.72 | 0.074 | 0.329 | 0.64 to 0.71 | 0.67 | 0.169 |
| Random forest | F0 | 0.74 | 0.33 to 0.67 | 0.004 | 0.29 to 0.71 | 0.023 | 0.193 | 0.72 to 0.79 | 0.75 | 0.742 |
| SVM (RBF) | F0 | 0.65 | 0.34 to 0.67 | 0.084 | 0.30 to 0.72 | 0.183 | 0.367 | 0.60 to 0.67 | 0.63 | 0.234 |
| 1D-CNN | F0 | 0.44 | 0.34 to 0.67 | 0.485 | 0.30 to 0.73 | 0.590 | 0.590 | 0.42 to 0.48 | 0.45 | 0.811 |
| BiLSTM | F0 | 0.73 | 0.33 to 0.67 | 0.010 | 0.32 to 0.71 | 0.021 | 0.193 | 0.65 to 0.78 | 0.72 | 0.349 |
| Fusion MLP | F0 | 0.71 | 0.33 to 0.67 | 0.017 | 0.32 to 0.72 | 0.047 | 0.329 | 0.64 to 0.75 | 0.69 | 0.302 |
| Small GNN | F0 | 0.72 | 0.33 to 0.67 | 0.011 | 0.29 to 0.71 | 0.050 | 0.329 | 0.71 to 0.73 | 0.72 | 0.610 |
| ST-GCN | F0 | 0.70 | 0.33 to 0.67 | 0.016 | 0.32 to 0.72 | 0.053 | 0.329 | 0.63 to 0.74 | 0.68 | 0.255 |
| ST-GAT | F0 | 0.69 | 0.33 to 0.67 | 0.029 | 0.32 to 0.72 | 0.072 | 0.329 | 0.63 to 0.74 | 0.68 | 0.410 |
| Logistic regression | F1 | 0.70 | 0.33 to 0.67 | 0.017 | 0.30 to 0.72 | 0.058 | – | 0.65 to 0.73 | 0.69 | 0.203 |
| Random forest | F1 | 0.75 | 0.33 to 0.67 | 0.002 | 0.29 to 0.71 | 0.015 | – | 0.73 to 0.80 | 0.76 | 0.626 |
| ST-GAT | F1 | 0.69 | 0.33 to 0.67 | 0.034 | 0.32 to 0.71 | 0.074 | – | 0.62 to 0.73 | 0.67 | 0.367 |

### Step 2 (c): retraining trial-block permutation (permutation_cluster.csv)

| Model | Config | AUC | permutations | seeds per permutation | null mean | null 95% | p two-sided | p one-sided |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | F0 | 0.69 | 500 | 1 | 0.48 | 0.24 to 0.73 | 0.146 | 0.052 |
| Logistic regression | F1 | 0.70 | 500 | 1 | 0.48 | 0.22 to 0.72 | 0.124 | 0.042 |
| Random forest | F0 | 0.74 | 188 | 5 | 0.47 | 0.21 to 0.76 | 0.106 | 0.037 |
| Random forest | F1 | 0.75 | 188 | 5 | 0.48 | 0.21 to 0.75 | 0.095 | 0.032 |

### Step 3: Holm–Bonferroni (holm_adjusted.csv)

**(i) Table 3, zone-level permutation (manuscript p)** (family size 9; raw p source: results/metrics.csv perm_p_auc (manuscript Table 3, zone-level null))

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| logreg | 0.027 | 0.110 | yes |  |
| rf | 0.002 | 0.018 | yes | yes |
| svm | 0.079 | 0.159 |  |  |
| cnn1d | 0.479 | 0.479 |  |  |
| bilstm | 0.007 | 0.056 | yes |  |
| fusion_mlp | 0.012 | 0.072 | yes |  |
| small_gnn | 0.008 | 0.056 | yes |  |
| gcn | 0.014 | 0.072 | yes |  |
| gat | 0.031 | 0.110 | yes |  |

**(i) Table 3, zone-level permutation (recomputed this run, rng 0)** (family size 9; raw p source: permutation_cluster.csv zone_level p_two_sided)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| logreg | 0.026 | 0.106 | yes |  |
| rf | 0.004 | 0.040 | yes | yes |
| svm | 0.084 | 0.169 |  |  |
| cnn1d | 0.485 | 0.485 |  |  |
| bilstm | 0.010 | 0.080 | yes |  |
| fusion_mlp | 0.017 | 0.099 | yes |  |
| small_gnn | 0.011 | 0.080 | yes |  |
| gcn | 0.016 | 0.099 | yes |  |
| gat | 0.029 | 0.106 | yes |  |

**(i) Table 3, trial-block permutation (this run)** (family size 9; raw p source: permutation_cluster.csv trial_block p_two_sided)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| logreg | 0.074 | 0.329 |  |  |
| rf | 0.023 | 0.193 | yes |  |
| svm | 0.183 | 0.367 |  |  |
| cnn1d | 0.590 | 0.590 |  |  |
| bilstm | 0.021 | 0.193 | yes |  |
| fusion_mlp | 0.047 | 0.329 | yes |  |
| small_gnn | 0.050 | 0.329 | yes |  |
| gcn | 0.053 | 0.329 |  |  |
| gat | 0.072 | 0.329 |  |  |

**(ii) Table 4, 18 paired comparisons** (family size 18; raw p source: results/paired_tests.csv perm_p (manuscript Table 4))

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| rf:full − logreg:full (main_15db) | 0.437 | 1.000 |  |  |
| svm:full − logreg:full (main_15db) | 0.596 | 1.000 |  |  |
| cnn1d:full − logreg:full (main_15db) | 0.061 | 1.000 |  |  |
| bilstm:full − logreg:full (main_15db) | 0.614 | 1.000 |  |  |
| fusion_mlp:full − logreg:full (main_15db) | 0.765 | 1.000 |  |  |
| small_gnn:full − logreg:full (main_15db) | 0.642 | 1.000 |  |  |
| gcn:full − logreg:full (main_15db) | 0.847 | 1.000 |  |  |
| gat:full − logreg:full (main_15db) | 0.950 | 1.000 |  |  |
| gat:full − gat:no_edges (ablation_15db) | 0.721 | 1.000 |  |  |
| gcn:full − gcn:no_edges (ablation_15db) | 0.642 | 1.000 |  |  |
| small_gnn:full − small_gnn:no_edges (ablation_15db) | 0.866 | 1.000 |  |  |
| gat:full − gat:no_force (ablation_15db) | 0.277 | 1.000 |  |  |
| gat:full − gat:no_acceleration (ablation_15db) | 0.213 | 1.000 |  |  |
| logreg:full − logreg:no_force (ablation_15db) | 0.570 | 1.000 |  |  |
| logreg:full − logreg:no_acceleration (ablation_15db) | 0.071 | 1.000 |  |  |
| gat:full − fusion_mlp:full (main_15db) | 0.519 | 1.000 |  |  |
| gcn:full − fusion_mlp:full (main_15db) | 0.884 | 1.000 |  |  |
| small_gnn:full − fusion_mlp:full (main_15db) | 0.912 | 1.000 |  |  |

**(iii) Step 1 ablation vs F0, logreg** (family size 8; raw p source: ablation_features_paired.csv paired_perm_p)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| logreg: F1 − F0 | 0.578 | 1.000 |  |  |
| logreg: F2 − F0 | 0.350 | 1.000 |  |  |
| logreg: F3 − F0 | 0.532 | 1.000 |  |  |
| logreg: F4 − F0 | 0.857 | 1.000 |  |  |
| logreg: F5 − F0 | 0.947 | 1.000 |  |  |
| logreg: F6 − F0 | 0.078 | 0.628 |  |  |
| logreg: F7 − F0 | 0.582 | 1.000 |  |  |
| logreg: F8 − F0 | 0.350 | 1.000 |  |  |

**(iii) Step 1 ablation vs F0, rf** (family size 8; raw p source: ablation_features_paired.csv paired_perm_p)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| rf: F1 − F0 | 0.155 | 1.000 |  |  |
| rf: F2 − F0 | 0.412 | 1.000 |  |  |
| rf: F3 − F0 | 0.969 | 1.000 |  |  |
| rf: F4 − F0 | 0.863 | 1.000 |  |  |
| rf: F5 − F0 | 0.612 | 1.000 |  |  |
| rf: F6 − F0 | 0.811 | 1.000 |  |  |
| rf: F7 − F0 | 0.067 | 0.540 |  |  |
| rf: F8 − F0 | 0.218 | 1.000 |  |  |

**(iii) Step 1 ablation vs F0, svm** (family size 8; raw p source: ablation_features_paired.csv paired_perm_p)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| svm: F1 − F0 | 0.433 | 1.000 |  |  |
| svm: F2 − F0 | 0.031 | 0.252 | yes |  |
| svm: F3 − F0 | 0.753 | 1.000 |  |  |
| svm: F4 − F0 | 0.830 | 1.000 |  |  |
| svm: F5 − F0 | 0.394 | 1.000 |  |  |
| svm: F6 − F0 | 0.640 | 1.000 |  |  |
| svm: F7 − F0 | 0.373 | 1.000 |  |  |
| svm: F8 − F0 | 0.108 | 0.756 |  |  |

**(iii) Step 1 ablation vs F0, bilstm** (family size 3; raw p source: ablation_features_paired.csv paired_perm_p)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| bilstm: F1 − F0 | 0.899 | 1.000 |  |  |
| bilstm: F2 − F0 | 0.240 | 0.721 |  |  |
| bilstm: F3 − F0 | 0.572 | 1.000 |  |  |

**(iii) Step 1 ablation vs F0, fusion_mlp** (family size 3; raw p source: ablation_features_paired.csv paired_perm_p)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| fusion_mlp: F1 − F0 | 0.872 | 1.000 |  |  |
| fusion_mlp: F2 − F0 | 0.031 | 0.094 | yes |  |
| fusion_mlp: F3 − F0 | 0.739 | 1.000 |  |  |

**(iii) Step 1 ablation vs F0, gat** (family size 3; raw p source: ablation_features_paired.csv paired_perm_p)

| Item | p raw | p Holm | sig. raw | sig. Holm |
|---|---|---|---|---|
| gat: F1 − F0 | 0.908 | 1.000 |  |  |
| gat: F2 − F0 | 0.071 | 0.213 |  |  |
| gat: F3 − F0 | 1.000 | 1.000 |  |  |

### Step 4: discrimination levels (discrimination_levels.csv)

| Model | Config | Pooled AUC | Within-trial mean AUC (n trials) | Trial-level AUC ≥1 unstable (95% CI) | Spearman ρ (p) | Pooled AUC w/o 04, 06 |
|---|---|---|---|---|---|---|
| Logistic regression | F0 | 0.69 | 0.81 (8) | 0.83 (0.57 to 1.00) | 0.49 (0.057) | 0.63 |
| Random forest | F0 | 0.74 | 0.50 (8) | 0.90 (0.69 to 1.00) | 0.67 (0.005) | 0.69 |
| SVM (RBF) | F0 | 0.65 | 0.62 (8) | 0.77 (0.50 to 1.00) | 0.31 (0.235) | 0.61 |
| 1D-CNN | F0 | 0.44 | 0.44 (8) | 0.46 (0.07 to 0.85) | -0.12 (0.645) | 0.52 |
| BiLSTM | F0 | 0.73 | 0.56 (8) | 0.88 (0.67 to 1.00) | 0.54 (0.031) | 0.67 |
| Fusion MLP | F0 | 0.71 | 0.56 (8) | 0.81 (0.56 to 1.00) | 0.41 (0.117) | 0.66 |
| Small GNN | F0 | 0.72 | 0.50 (8) | 0.85 (0.62 to 1.00) | 0.49 (0.053) | 0.63 |
| ST-GCN | F0 | 0.70 | 0.56 (8) | 0.81 (0.53 to 1.00) | 0.46 (0.071) | 0.64 |
| ST-GAT | F0 | 0.69 | 0.56 (8) | 0.85 (0.62 to 1.00) | 0.51 (0.045) | 0.63 |
| Logistic regression | F1 | 0.70 | 0.62 (8) | 0.83 (0.57 to 1.00) | 0.48 (0.058) | 0.65 |
| Random forest | F1 | 0.75 | 0.62 (8) | 0.88 (0.67 to 1.00) | 0.63 (0.009) | 0.70 |
| ST-GAT | F1 | 0.69 | 0.56 (8) | 0.83 (0.58 to 1.00) | 0.45 (0.084) | 0.63 |

### Step 4: within-trial AUC per trial (within_trial_per_trial.csv); trials used: trial_07, trial_10, trial_11, trial_13, trial_16, trial_17, trial_19, trial_20; column header = trial [zone labels 1-2-3]

| Model | config | 07 [001] | 10 [011] | 11 [110] | 13 [011] | 16 [011] | 17 [100] | 19 [101] | 20 [001] |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | F0 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| Logistic regression | F1 | 1.00 | 0.00 | 0.50 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |
| Random forest | F0 | 1.00 | 0.00 | 0.00 | 1.00 | 0.50 | 1.00 | 0.00 | 0.50 |
| Random forest | F1 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 |
| SVM (RBF) | F0 | 0.50 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| 1D-CNN | F0 | 0.00 | 1.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 1.00 |
| BiLSTM | F0 | 0.50 | 0.00 | 0.50 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |
| Fusion MLP | F0 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |
| Small GNN | F0 | 0.50 | 0.50 | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 0.50 |
| ST-GCN | F0 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |
| ST-GAT | F0 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |
| ST-GAT | F1 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 |

### Step 5: calibration (calibration.csv)

| Model | Config | Brier | Brier (constant 0.52) | Brier skill | ECE (5 bins) | Slope (95% CI) | Intercept | CITL | Probability |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | F0 | 0.263 | 0.250 | -0.05 | 0.193 | 0.18 (0.01 to 0.35) | 0.03 | 0.24 | LogisticRegression predict_proba (class_weight=balanced; not Platt-scaled) |
| Random forest | F0 | 0.214 | 0.250 | 0.14 | 0.088 | 0.64 (0.17 to 1.11) | -0.03 | -0.04 | RandomForest vote fraction (class_weight=balanced; not calibrated) |
| SVM (RBF) | F0 | 0.232 | 0.250 | 0.07 | 0.123 | 0.71 (-0.04 to 1.45) | -0.00 | -0.04 | SVC(probability=True): libsvm internal Platt scaling with 5-fold CV on the training trials |
| BiLSTM | F0 | 0.266 | 0.250 | -0.06 | 0.244 | 0.18 (0.04 to 0.32) | 0.16 | 1.03 | sigmoid of logit, trained with class-weighted BCE (not recalibrated) |
| Fusion MLP | F0 | 0.267 | 0.250 | -0.07 | 0.261 | 0.16 (0.02 to 0.30) | 0.09 | 0.48 | sigmoid of logit, trained with class-weighted BCE (not recalibrated) |
| ST-GAT | F0 | 0.293 | 0.250 | -0.18 | 0.284 | 0.16 (0.02 to 0.30) | 0.06 | 0.16 | sigmoid of logit, trained with class-weighted BCE (not recalibrated) |
| Logistic regression | F1 | 0.262 | 0.250 | -0.05 | 0.193 | 0.22 (0.02 to 0.41) | 0.04 | 0.26 | LogisticRegression predict_proba (class_weight=balanced; not Platt-scaled) |
| Random forest | F1 | 0.208 | 0.250 | 0.17 | 0.088 | 0.69 (0.21 to 1.18) | -0.04 | -0.04 | RandomForest vote fraction (class_weight=balanced; not calibrated) |

### Permutation importance, feature family × modality (importance.csv)

**Logistic regression** (AUC 0.69)

| Group | ΔAUC across | ΔAUC within-trial |
|---|---|---|
| [acc] statistical | 0.157 | 0.003 |
| [force] statistical | 0.099 | -0.003 |
| [force] mode_energy | 0.051 | 0.006 |
| [force] band_centroid | 0.005 | -0.006 |
| [force] chatter_indicator | -0.001 | 0.001 |
| [force] wavelet | -0.008 | -0.009 |
| [acc] wavelet | -0.015 | 0.000 |
| [acc] chatter_indicator | -0.021 | -0.009 |
| [acc] band_centroid | -0.031 | -0.007 |
| [acc] mode_energy | -0.058 | 0.002 |

**Random forest** (AUC 0.74)

| Group | ΔAUC across | ΔAUC within-trial |
|---|---|---|
| [acc] statistical | 0.043 | -0.007 |
| [force] statistical | 0.035 | 0.002 |
| [force] mode_energy | 0.017 | 0.003 |
| [acc] band_centroid | 0.011 | 0.000 |
| [acc] mode_energy | 0.007 | -0.003 |
| [force] band_centroid | 0.001 | -0.005 |
| [force] wavelet | 0.001 | -0.003 |
| [acc] wavelet | -0.004 | -0.010 |
| [force] chatter_indicator | -0.004 | -0.003 |
| [acc] chatter_indicator | -0.007 | -0.006 |

### Run times (run_times.csv; CPU seconds summed over seeds, jobs ran in parallel)

| Model | Configs trained | CPU time, all configs × 5 seeds (min) | Last finished |
|---|---|---|---|
| Logistic regression | F0 F1 F2 F3 F4 F5 F6 F7 F8 | 0.5 | 2026-11-10 14:45:39 |
| Random forest | F0 F1 F2 F3 F4 F5 F6 F7 F8 | 11.0 | 2026-11-10 14:46:39 |
| SVM (RBF) | F0 F1 F2 F3 F4 F5 F6 F7 F8 | 0.1 | 2026-11-10 14:46:39 |
| Small GNN | F1 | 5.9 | 2026-11-10 15:09:27 |
| ST-GCN | F1 | 48.6 | 2026-11-10 15:18:00 |
| ST-GAT | F0 F1 F2 F3 | 343.5 | 2026-11-10 16:11:07 |
| Fusion MLP | F0 F1 F2 F3 | 291.3 | 2026-11-10 16:40:12 |
| BiLSTM | F0 F1 F2 F3 | 29.3 | 2026-11-10 16:42:40 |
