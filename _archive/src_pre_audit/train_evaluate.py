"""
train_evaluate.py   -   STEP 2 of the pipeline
===============================================
Leave-one-trial-out training and evaluation of every model (Section 2.9).

Leave-one-trial-out (LOTO) cross-validation in one sentence:
    with 16 trials we train 16 times; each time ONE trial is hidden, the model
    learns from the others, and then predicts the hidden trial. Joining the 16
    sets of predictions gives an honest "out-of-fold" prediction for every zone.

Files written (all in outputs/):
  metrics_main.csv        Tables 2 and 3: every model, pooled out-of-fold metrics + 95 % CI
  metrics_by_material.csv the same, split by WAAM / wrought
  ablation.csv            Table 4: ST-GAT ablations
  attribution.csv         Table 5: integrated-gradient share per modality and material
  attention.csv           mean GAT attention on kinematic vs thermal edges per material
  attention_per_trial.csv the same per trial, plus the full 6 x 6 attention matrix (a_i_j columns)
  stat_tests.csv          DeLong (AUC) and Wilcoxon (fold RMSE) tests, ST-GAT vs fusion MLP
  fold_metrics.csv        per-fold metrics of every run
  predictions_oof.csv     every out-of-fold zone prediction
  latency.json            parameter count and inference time of the ST-GAT
  run_info.json           settings, software versions, dataset composition
  training_log_<stage>.txt  copy of what was printed on screen

How to run (from the project folder)
  python run_pipeline.py train          main comparison
  python run_pipeline.py ablations      Table 4
  python run_pipeline.py quick          5-minute check that everything works
or directly
  python src/train_evaluate.py                         main comparison
  python src/train_evaluate.py --stage ablations       Table 4
  python src/train_evaluate.py --quick --stage all     fast check (3 epochs, 2 folds)
  python src/train_evaluate.py --seeds 42 43 44 45 46  final paper run
  python src/train_evaluate.py --protocol loco         leave-one-condition-out (Table S1)
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
from graph_data import ACCEL_COLS, FORCE_COLS, KINEMATIC_COLS, AblationVariant, TrialGraphBuilder
from models import ModelFactory

# Hide two harmless library notices so the screen output stays readable.
warnings.filterwarnings("ignore", category=FutureWarning)      # sklearn SVC(probability=True) notice
warnings.filterwarnings("ignore", category=UserWarning)

# The deep models of Tables 2-3 and the ablations of Table 4.
DEEP_MODELS = ["cnn1d", "bilstm", "fusion_mlp", "gcn", "gat"]
ABLATIONS = ["without_force_features", "without_acceleration_features", "without_thermal_edges",
             "without_kinematic_edges", "no_spatial_edges", "without_temporal_module", "material_flag",
             "classification_only", "roughness_only"]


def seed_everything(seed: int) -> None:
    """Fix every random-number generator so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# 1. Small helper classes
# ===========================================================================
class TrainingLogger:
    """Prints a message AND writes it to a log file."""

    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()                     # write to disk now, in case the run is stopped

    def close(self) -> None:
        self.file.close()


class FoldSplitter:
    """Decides which trials are used for testing, validation and training."""

    @staticmethod
    def make_folds(graphs: list[dict], protocol: str) -> list[list[int]]:
        """List of folds; each fold is the list of TEST graph indices.

        loto = leave-one-trial-out:     one fold per trial.
        loco = leave-one-condition-out: one fold per (speed, depth) condition.
        """
        if protocol == "loto":
            return [[i] for i in range(len(graphs))]
        groups = {}
        for i, g in enumerate(graphs):
            groups.setdefault(g["condition"], []).append(i)   # collect trials with the same condition
        return list(groups.values())

    @staticmethod
    def pick_validation(train_idx: list[int], graphs: list[dict], rng: np.random.Generator) -> list[int]:
        """Inner validation set (for early stopping): one trial of each material when possible."""
        validation = []
        for material in ("WAAM", "WROUGHT"):
            candidates = [i for i in train_idx if graphs[i]["material_type"] == material]
            if len(candidates) > 1:
                validation.append(int(rng.choice(candidates)))
        if not validation:
            validation = [int(rng.choice(train_idx))]
        return validation


class Standardiser:
    """Scales inputs and Ra with statistics of the TRAINING trials only (no information leak)."""

    def __init__(self, train_graphs: list[dict]):
        # Mean and spread of every node feature over all zones and windows of the training trials.
        all_x = torch.cat([g["x"].reshape(-1, g["x"].shape[-1]) for g in train_graphs])
        self.x_mean = all_x.mean(0)
        self.x_std = all_x.std(0).clamp_min(1e-6)              # clamp avoids division by 0

        # Mean and spread of Ra.
        all_ra = torch.cat([g["y_reg"] for g in train_graphs])
        self.ra_mean = float(all_ra.mean())
        self.ra_std = max(float(all_ra.std()), 1e-6)

        # Raw windows: one scale per sensor, estimated from every 97th sample (fast).
        self.force_scale = torch.cat([g["force_raw"].flatten()[::97] for g in train_graphs]).std().clamp_min(1e-6)
        self.accel_scale = torch.cat([g["acceleration_raw"].flatten()[::97] for g in train_graphs]).std().clamp_min(1e-6)

    def transform(self, g: dict) -> dict:
        """Return a scaled copy of one graph (adds the key "y_reg_std")."""
        scaled = dict(g)
        scaled["x"] = (g["x"] - self.x_mean) / self.x_std
        scaled["force_raw"] = g["force_raw"] / self.force_scale
        scaled["acceleration_raw"] = g["acceleration_raw"] / self.accel_scale
        scaled["y_reg_std"] = (g["y_reg"] - self.ra_mean) / self.ra_std
        return scaled

    def transform_all(self, graphs: list[dict]) -> list[dict]:
        return [self.transform(g) for g in graphs]


# ===========================================================================
# 2. Metrics and statistical tests
# ===========================================================================
class Metrics:
    """All the numbers reported in the tables."""

    @staticmethod
    def classification(y: np.ndarray, p: np.ndarray) -> dict:
        """y = true labels (0/1), p = predicted probabilities of chatter."""
        predicted = (p >= 0.5).astype(int)                   # probability >= 0.5 -> "unstable"
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan   # AUC needs both classes
        return {"accuracy": accuracy_score(y, predicted) * 100,
                "precision": precision_score(y, predicted, zero_division=0),
                "recall": recall_score(y, predicted, zero_division=0),
                "f1": f1_score(y, predicted, zero_division=0),
                "roc_auc": auc}

    @staticmethod
    def regression(y: np.ndarray, p: np.ndarray) -> dict:
        """y = true Ra, p = predicted Ra (both in micrometre)."""
        return {"rmse_um": float(np.sqrt(mean_squared_error(y, p))),
                "mae_um": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p)) if np.var(y) > 0 else np.nan}

    @staticmethod
    def pooled_with_ci(pred: pd.DataFrame, draws: int, seed: int) -> dict:
        """Metrics of the pooled out-of-fold predictions + 95 % bootstrap confidence intervals.

        The bootstrap resamples whole TRIALS (not single zones), because zones of
        the same trial are not independent of each other.
        """
        trials = pred.trial_id.unique()
        rng = np.random.default_rng(seed)

        # --- point estimates on all predictions --------------------------------
        point = {}
        if pred.probability.notna().any():                      # model predicts chatter
            point.update(Metrics.classification(pred.y_class.to_numpy(int), pred.probability.to_numpy()))
        if pred.ra_pred.notna().any():                          # model predicts Ra
            point.update(Metrics.regression(pred.y_reg.to_numpy(), pred.ra_pred.to_numpy()))

        # --- bootstrap: draw trials with replacement many times ----------------
        boot = {name: [] for name in point}
        rows_of_trial = {t: pred[pred.trial_id == t] for t in trials}
        for _ in range(draws):
            chosen = rng.choice(trials, len(trials), replace=True)
            sample = pd.concat([rows_of_trial[t] for t in chosen])
            if "accuracy" in point:
                m = Metrics.classification(sample.y_class.to_numpy(int), sample.probability.to_numpy())
                for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    boot[name].append(m[name])
            if "rmse_um" in point:
                m = Metrics.regression(sample.y_reg.to_numpy(), sample.ra_pred.to_numpy())
                for name in ("rmse_um", "mae_um", "r2"):
                    boot[name].append(m[name])

        # --- 2.5 % and 97.5 % quantiles = 95 % confidence interval --------------
        result = {}
        for name, value in point.items():
            values = np.asarray(boot[name], dtype=float)
            values = values[np.isfinite(values)]                # drop NaN (e.g. AUC with one class)
            result[name] = value
            result[f"{name}_ci_low"] = float(np.quantile(values, 0.025)) if values.size else np.nan
            result[f"{name}_ci_high"] = float(np.quantile(values, 0.975)) if values.size else np.nan
        return result

    @staticmethod
    def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
        """DeLong et al. (1988): are two AUCs on the SAME zones different? Returns (auc1, auc2, p)."""
        positive, negative = y == 1, y == 0
        m, n = positive.sum(), negative.sum()
        if m < 2 or n < 2:
            return np.nan, np.nan, np.nan

        def components(p):
            # psi[i, j] = 1 if unstable zone i scores higher than stable zone j, 0.5 if tied.
            pos_scores = p[positive][:, None]
            neg_scores = p[negative][None, :]
            psi = (pos_scores > neg_scores).astype(float) + 0.5 * (pos_scores == neg_scores)
            return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)

        auc1, v10_1, v01_1 = components(p1)
        auc2, v10_2, v01_2 = components(p2)
        s10 = np.cov(np.vstack([v10_1, v10_2]))
        s01 = np.cov(np.vstack([v01_1, v01_2]))
        variance = ((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m
                    + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n)
        if variance <= 0:
            return auc1, auc2, 1.0
        z = (auc1 - auc2) / np.sqrt(variance)
        p_value = 2 * stats.norm.sf(abs(z))                      # two-sided p-value
        return float(auc1), float(auc2), float(p_value)


# ===========================================================================
# 3. Classical machine-learning baselines
# ===========================================================================
class ClassicalBaselines:
    """Logistic regression, random forest, SVM ... on zone features averaged over the windows."""

    def __init__(self, seed: int):
        self.seed = seed

    def classifiers(self) -> dict:
        """Name -> a function that returns a NEW untrained classifier."""
        return {
            "Logistic regression": lambda: make_pipeline(StandardScaler(),
                                                         LogisticRegression(class_weight="balanced", max_iter=5000)),
            "Random forest": lambda: RandomForestClassifier(300, class_weight="balanced",
                                                            random_state=self.seed, n_jobs=-1),
            "SVM (RBF)": lambda: make_pipeline(StandardScaler(),
                                               SVC(class_weight="balanced", probability=True, random_state=self.seed)),
        }

    def regressors(self) -> dict:
        return {
            "Ridge regression": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
            "Random forest regressor": lambda: RandomForestRegressor(300, random_state=self.seed, n_jobs=-1),
            "SVR (RBF)": lambda: make_pipeline(StandardScaler(), SVR()),
        }

    @staticmethod
    def prediction_row(model, fold, info, probability, ra_pred) -> dict:
        """One row of the predictions table. info = (trial, material, zone, y_class, y_reg)."""
        trial, material, zone, y_class, y_reg = info
        return dict(model=model, fold=fold, trial_id=trial, material_type=material, zone_id=zone,
                    y_class=y_class, y_reg=y_reg, probability=probability, ra_pred=ra_pred)

    def run(self, graphs: list[dict], folds: list[list[int]]) -> pd.DataFrame:
        # One feature vector per zone: average the window features -> [N, F] per trial.
        zone_features = [g["x"].mean(1).numpy() for g in graphs]
        rows = []
        for fold, test in enumerate(folds):
            train = [i for i in range(len(graphs)) if i not in test]
            x_train = np.concatenate([zone_features[i] for i in train])
            x_test = np.concatenate([zone_features[i] for i in test])
            y_class = np.concatenate([graphs[i]["y_class"].numpy() for i in train])
            y_ra = np.concatenate([graphs[i]["y_reg"].numpy() for i in train])

            # Information about every test zone (to write next to the predictions).
            test_zones = []
            for i in test:
                g = graphs[i]
                for z in range(len(g["y_class"])):
                    test_zones.append((g["trial_id"], g["material_type"], z,
                                       float(g["y_class"][z]), float(g["y_reg"][z])))

            # Classifiers -> probability of chatter.
            for name, make_model in self.classifiers().items():
                if len(np.unique(y_class)) > 1:
                    probabilities = make_model().fit(x_train, y_class).predict_proba(x_test)[:, 1]
                else:                                 # only one class in training: predict its rate
                    probabilities = np.full(len(x_test), y_class.mean())
                for info, p in zip(test_zones, probabilities):
                    rows.append(self.prediction_row(name, fold, info, float(p), np.nan))

            # Regressors -> Ra.
            for name, make_model in self.regressors().items():
                predictions = make_model().fit(x_train, y_ra).predict(x_test)
                for info, p in zip(test_zones, predictions):
                    rows.append(self.prediction_row(name, fold, info, np.nan, float(p)))

            # Reference: always predict the training mean of Ra.
            for info in test_zones:
                rows.append(self.prediction_row("Predict training mean (reference)", fold, info,
                                                np.nan, float(y_ra.mean())))
        return pd.DataFrame(rows)


# ===========================================================================
# 4. Deep models: training, cross-validation, attributions
# ===========================================================================
class DeepModelTrainer:
    """Trains the neural networks inside the cross-validation loop."""

    def __init__(self, config, epochs: int, patience: int, logger: TrainingLogger):
        self.config = config
        self.epochs = epochs
        self.patience = patience
        self.logger = logger

    # -----------------------------------------------------------------------
    # Loss and training of one model
    # -----------------------------------------------------------------------
    def loss(self, output, g: dict, pos_weight: torch.Tensor, task: str) -> torch.Tensor:
        """Eq. (14): class-weighted BCE (chatter) + lambda * MSE (standardised Ra)."""
        logits, ra, _ = output
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, g["y_class"], pos_weight=pos_weight)
        mse = torch.nn.functional.mse_loss(ra, g["y_reg_std"])
        if task == "classification_only":
            return bce
        if task == "roughness_only":
            return mse
        return bce + self.config.train.regression_weight * mse

    def train_one(self, kind: str, variant: str, train: list[dict], val: list[dict], seed: int):
        """Train one model with early stopping on the validation loss. Returns (best model, epochs run)."""
        cfg = self.config
        seed_everything(seed)
        model = ModelFactory.build(kind, train[0]["x"].shape[-1], cfg, variant)
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate,
                                     weight_decay=cfg.train.weight_decay)

        # Unstable zones are rare, so they get a larger weight in the loss:
        # pos_weight = (number of stable zones) / (number of unstable zones).
        y = torch.cat([g["y_class"] for g in train])
        pos_weight = torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))
        task = variant if variant in ("classification_only", "roughness_only") else "joint"

        best_loss, best_state, epochs_without_improvement = np.inf, None, 0
        for epoch in range(self.epochs):
            # --- training: one graph (trial) at a time, in random order --------
            model.train()                                       # switches dropout ON
            for i in np.random.permutation(len(train)):
                optimiser.zero_grad()                           # forget the previous gradients
                loss = self.loss(model(train[i]), train[i], pos_weight, task)
                loss.backward()                                 # compute new gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # avoid huge steps
                optimiser.step()                                # update the weights

            # --- validation loss ----------------------------------------------
            model.eval()                                        # switches dropout OFF
            with torch.no_grad():                               # no gradients needed here
                val_losses = [self.loss(model(g), g, pos_weight, task).item() for g in val]
                val_loss = float(np.mean(val_losses))

            # --- early stopping -----------------------------------------------
            if val_loss < best_loss - 1e-5:                     # clearly better: remember these weights
                best_loss, epochs_without_improvement = val_loss, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        model.load_state_dict(best_state)                       # go back to the best epoch
        model.eval()
        return model, epoch + 1

    # -----------------------------------------------------------------------
    # Integrated gradients (Table 5)
    # -----------------------------------------------------------------------
    @staticmethod
    def integrated_gradients(model, g: dict, steps: int = 16) -> dict:
        """How much each sensor contributes to the Ra prediction (Sundararajan et al., 2017).

        The input is scaled from 0 (= training mean after standardisation) to its
        real value in `steps` steps; the gradients along the way are averaged and
        multiplied by the input. Attributions are summed per sensor and per zone.
        """
        keys = ["x", "force_raw", "acceleration_raw"]
        average_gradient = {k: torch.zeros_like(g[k]) for k in keys}
        for alpha in torch.linspace(1.0 / steps, 1.0, steps):
            scaled = dict(g)
            for k in keys:
                scaled[k] = (g[k] * alpha).requires_grad_(True)   # ask PyTorch to track gradients
            _, ra, _ = model(scaled)
            gradients = torch.autograd.grad(ra.sum(), [scaled[k] for k in keys])
            for k, grad in zip(keys, gradients):
                average_gradient[k] += grad / steps
        attribution = {k: (average_gradient[k] * g[k]).abs() for k in keys}

        def per_zone(t: torch.Tensor) -> torch.Tensor:
            """Sum everything except the zone axis -> [N]."""
            return t.reshape(t.shape[0], -1).sum(1)

        return {"Force": per_zone(attribution["x"][..., FORCE_COLS]) + per_zone(attribution["force_raw"]),
                "Acceleration": per_zone(attribution["x"][..., ACCEL_COLS])
                                + per_zone(attribution["acceleration_raw"]),
                "Kinematic / geometric": per_zone(attribution["x"][..., KINEMATIC_COLS])}

    # -----------------------------------------------------------------------
    # The outer cross-validation loop
    # -----------------------------------------------------------------------
    def run(self, graphs: list[dict], folds: list[list[int]], kinds: list[str], variants: list[str],
            seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Train and test every (model, variant, seed) in every fold.

        Returns three tables: predictions, attributions and attention.
        """
        rows, attributions, attention = [], [], []

        # The same validation trials are used for every model (fixed random generator).
        rng = np.random.default_rng(0)
        validation_per_fold = []
        for test in folds:
            train_idx = [i for i in range(len(graphs)) if i not in test]
            validation_per_fold.append(FoldSplitter.pick_validation(train_idx, graphs, rng))

        # Jobs: every model with the full input, then the ST-GAT ablations.
        jobs = [(kind, "full") for kind in kinds] + [("gat", variant) for variant in variants]
        for kind, variant in jobs:
            started = time.time()
            modified_graphs = [AblationVariant.apply(g, variant) for g in graphs]

            for fold, test in enumerate(folds):
                val_idx = validation_per_fold[fold]
                train_idx = [i for i in range(len(graphs)) if i not in test and i not in val_idx]

                # Standardise with the training trials only.
                scaler = Standardiser([modified_graphs[i] for i in train_idx])
                train = scaler.transform_all([modified_graphs[i] for i in train_idx])
                val = scaler.transform_all([modified_graphs[i] for i in val_idx])
                tests = scaler.transform_all([modified_graphs[i] for i in test])

                for seed in seeds:
                    model, n_epochs = self.train_one(kind, variant, train, val, seed)
                    for g in tests:
                        rows += self.predict_trial(model, g, kind, variant, seed, fold, n_epochs, scaler)
                        # Extra analyses only for the full ST-GAT.
                        if kind == "gat" and variant == "full":
                            attention.append(self.attention_row(model, g, seed))
                            for modality, values in self.integrated_gradients(model, g).items():
                                attributions.append(dict(trial_id=g["trial_id"], material_type=g["material_type"],
                                                         seed=seed, modality=modality,
                                                         attribution=float(values.sum())))
            self.logger.log(f"  {kind:<10} {variant:<32} done in {(time.time() - started) / 60:.1f} min")
        return pd.DataFrame(rows), pd.DataFrame(attributions), pd.DataFrame(attention)

    @staticmethod
    def predict_trial(model, g, kind, variant, seed, fold, n_epochs, scaler: Standardiser) -> list[dict]:
        """Predict the 6 zones of one test trial -> 6 table rows."""
        with torch.no_grad():
            logits, ra, _ = model(g)
        probability = torch.sigmoid(logits).numpy()                     # logit -> probability 0..1
        ra_um = ra.numpy() * scaler.ra_std + scaler.ra_mean              # undo the standardisation
        rows = []
        for z in range(len(probability)):
            rows.append(dict(model=kind, variant=variant, seed=seed, fold=fold, trial_id=g["trial_id"],
                             material_type=g["material_type"], zone_id=z,
                             y_class=float(g["y_class"][z]), y_reg=float(g["y_reg"][z]),
                             probability=float(probability[z]) if variant != "roughness_only" else np.nan,
                             ra_pred=float(ra_um[z]) if variant != "classification_only" else np.nan,
                             epochs_trained=n_epochs))
        return rows

    @staticmethod
    def attention_row(model, g, seed) -> dict:
        kinematic, thermal = model.attention_by_edge_type(g["adj"])
        row = dict(trial_id=g["trial_id"], material_type=g["material_type"], seed=seed,
                   kinematic_attention=kinematic, thermal_attention=thermal)
        # The full matrix as columns a_0_0 ... a_5_5 (row i = receiving zone, column j = neighbour).
        alpha = model.attention_matrix().numpy()
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                row[f"a_{i}_{j}"] = float(alpha[i, j])
        return row


# ===========================================================================
# 5. Speed of the ST-GAT (Section 5 of the draft)
# ===========================================================================
class LatencyBenchmark:
    """Counts parameters and times how long the ST-GAT needs per prediction on the CPU."""

    def __init__(self, config):
        self.config = config

    def measure(self, graphs: list[dict]) -> dict:
        model = ModelFactory.build("gat", graphs[0]["x"].shape[-1], self.config).eval()
        scaler = Standardiser(graphs)
        g = scaler.transform(graphs[0])
        n_windows = g["x"].shape[1]

        # A graph with only ONE window per zone = what arrives online every hop.
        windowed_keys = ("x", "force_raw", "acceleration_raw")
        single = {k: (v[:, :1] if k in windowed_keys else v) for k, v in g.items()}

        with torch.no_grad():
            for _ in range(3):                             # warm-up runs (not timed)
                model(single)
            t0 = time.perf_counter()
            for _ in range(20):
                model(single)
            per_window_ms = (time.perf_counter() - t0) / 20 * 1000

            t0 = time.perf_counter()
            for _ in range(3):
                model(g)
            per_trial_ms = (time.perf_counter() - t0) / 3 * 1000

        # Budget: a prediction must be ready before the next window starts.
        hop_ms = self.config.signal.window_ms * (1 - self.config.signal.overlap_fraction)
        return {"parameters": int(sum(p.numel() for p in model.parameters())),
                "ms_per_graph_one_window": round(per_window_ms, 2),
                "ms_per_graph_all_windows": round(per_trial_ms, 1), "windows_per_graph": int(n_windows),
                "budget_ms_one_hop": hop_ms, "within_budget": bool(per_window_ms < hop_ms),
                "hardware": platform.processor() or platform.machine(), "threads": torch.get_num_threads()}


# ===========================================================================
# 6. Turning predictions into result tables
# ===========================================================================
class ResultSummariser:
    """Averages seeds and computes the metrics of every model / variant / material."""

    def __init__(self, draws: int):
        self.draws = draws                                 # number of bootstrap resamples

    @staticmethod
    def average_seeds(frame: pd.DataFrame) -> pd.DataFrame:
        """One row per zone: probability and Ra averaged over the seeds."""
        return frame.groupby(["trial_id", "material_type", "zone_id"], as_index=False).agg(
            y_class=("y_class", "first"), y_reg=("y_reg", "first"),
            probability=("probability", "mean"), ra_pred=("ra_pred", "mean"))

    def summarise(self, pred: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
        """Pooled metrics with trial-bootstrap CI for every group (e.g. every model)."""
        out = []
        for keys, frame in pred.groupby(key_cols, sort=False):
            zones = self.average_seeds(frame)
            if not isinstance(keys, tuple):
                keys = (keys,)
            group = dict(zip(key_cols, keys))                  # e.g. {"model": "gat"}
            out.append({**group, **Metrics.pooled_with_ci(zones, self.draws, 0)})
        return pd.DataFrame(out)

    @staticmethod
    def fold_rmse(pred: pd.DataFrame) -> pd.DataFrame:
        """Ra RMSE of every (model, variant, fold), seeds averaged per zone first."""
        rows = []
        for (model, variant, fold), f in pred.groupby(["model", "variant", "fold"]):
            zones = f.groupby(["trial_id", "zone_id"]).agg(y_reg=("y_reg", "first"),
                                                           ra_pred=("ra_pred", "mean")).dropna()
            rmse = np.sqrt(np.mean((zones.y_reg - zones.ra_pred) ** 2)) if len(zones) else np.nan
            rows.append(dict(model=model, variant=variant, fold=fold, trial_id=f.trial_id.iloc[0], rmse_um=rmse))
        return pd.DataFrame(rows)


# ===========================================================================
# 7. The whole of Step 2
# ===========================================================================
class ExperimentRunner:
    """Reads the command-line options and runs the requested stage(s)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = CONFIG
        torch.set_num_threads(self.config.train.torch_threads)

        # LOCO results go into a sub-folder so they never overwrite the LOTO results.
        self.out = args.output if args.protocol == "loto" else args.output / "loco"
        self.out.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(self.out / f"training_log_{args.stage}.txt")

        # Data and folds.
        self.graphs = TrialGraphBuilder(self.config).load_all()
        self.folds = FoldSplitter.make_folds(self.graphs, args.protocol)
        self.epochs = args.epochs
        self.patience = self.config.train.patience
        self.draws = self.config.train.bootstrap_draws
        if args.quick:                                   # tiny run, only to check that the code works
            self.folds, self.epochs, self.patience, self.draws = self.folds[:2], 3, 2, 50

        self.trainer = DeepModelTrainer(self.config, self.epochs, self.patience, self.logger)
        self.summariser = ResultSummariser(self.draws)
        self.start_time = time.time()

    def run(self) -> None:
        a = self.args
        self.logger.log(f"{len(self.graphs)} trial graphs, {len(self.folds)} {a.protocol.upper()} folds, "
                        f"seeds {a.seeds}, up to {self.epochs} epochs, stage '{a.stage}'")
        if a.stage in ("ablations", "all") and a.protocol == "loto":
            self.run_ablations()
        if a.stage in ("main", "all"):
            self.run_main()
        self.logger.close()

    # -----------------------------------------------------------------------
    # Table 4
    # -----------------------------------------------------------------------
    def run_ablations(self) -> None:
        a = self.args
        ablations, _, _ = self.trainer.run(self.graphs, self.folds, [], a.ablations, a.seeds)

        # The "full" row: reuse the ST-GAT of the main stage if it has been run already.
        main_predictions = self.out / "predictions_oof.csv"
        if main_predictions.exists():
            previous = pd.read_csv(main_predictions)
            full = previous[(previous.model == "gat") & (previous.variant == "full")]
        else:
            full, _, _ = self.trainer.run(self.graphs, self.folds, ["gat"], [], a.seeds)

        ablations = pd.concat([full, ablations], ignore_index=True)
        ablations.to_csv(self.out / "predictions_ablation.csv", index=False)
        self.summariser.summarise(ablations, ["variant"]).to_csv(self.out / "ablation.csv", index=False)
        self.summariser.fold_rmse(ablations).to_csv(self.out / "fold_metrics_ablation.csv", index=False)
        self.logger.log(f"Ablations done in {(time.time() - self.start_time) / 60:.1f} min "
                        f"-> {self.out / 'ablation.csv'}")

    # -----------------------------------------------------------------------
    # Tables 2, 3 and 5
    # -----------------------------------------------------------------------
    def run_main(self) -> None:
        a = self.args

        # Classical baselines.
        classical = ClassicalBaselines(a.seeds[0]).run(self.graphs, self.folds)
        self.logger.log(f"Classical baselines done in {time.time() - self.start_time:.0f} s")

        # Deep models.
        deep, attributions, attention = self.trainer.run(self.graphs, self.folds, a.models, [], a.seeds)

        # All predictions in one table.
        classical["variant"], classical["seed"] = "full", a.seeds[0]
        all_predictions = pd.concat([classical, deep], ignore_index=True)
        all_predictions.to_csv(self.out / "predictions_oof.csv", index=False)

        # Tables 2 and 3.
        main_table = self.summariser.summarise(all_predictions, ["model"])
        main_table.to_csv(self.out / "metrics_main.csv", index=False)
        self.summariser.summarise(all_predictions, ["model", "material_type"]).to_csv(
            self.out / "metrics_by_material.csv", index=False)
        fold_metrics = self.summariser.fold_rmse(all_predictions)
        fold_metrics.to_csv(self.out / "fold_metrics.csv", index=False)

        self.save_stat_tests(deep, fold_metrics)
        self.save_attribution_and_attention(attributions, attention)

        # Latency and run information.
        latency = LatencyBenchmark(self.config).measure(self.graphs)
        (self.out / "latency.json").write_text(json.dumps(latency, indent=2))
        (self.out / "run_info.json").write_text(json.dumps(self.run_info(), indent=2, default=str))

        self.logger.log(f"\nMain results (pooled out-of-fold):\n{main_table.round(3).to_string(index=False)}")
        self.logger.log(f"\nAll outputs written to {self.out}")

    def save_stat_tests(self, deep: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
        """ST-GAT vs fusion MLP: DeLong test on AUC, Wilcoxon test on fold RMSE."""
        tests = []
        if {"gat", "fusion_mlp"} <= set(self.args.models):     # "<=" on sets means "is a subset of"
            # DeLong: compare the two models on the same zones (seeds averaged).
            zones = (deep[deep.variant == "full"].groupby(["model", "trial_id", "zone_id"], as_index=False)
                     .agg(y_class=("y_class", "first"), probability=("probability", "mean")))
            gat = zones[zones.model == "gat"].sort_values(["trial_id", "zone_id"])
            mlp = zones[zones.model == "fusion_mlp"].sort_values(["trial_id", "zone_id"])
            auc_gat, auc_mlp, p = Metrics.delong_test(gat.y_class.to_numpy(int),
                                                      gat.probability.to_numpy(), mlp.probability.to_numpy())
            tests.append(dict(test="DeLong AUC", model_a="gat", model_b="fusion_mlp",
                              value_a=auc_gat, value_b=auc_mlp, p_value=p))

            # Wilcoxon signed-rank test on the paired fold RMSEs.
            full = fold_metrics[fold_metrics.variant == "full"]
            rmse_gat = full[full.model == "gat"].set_index("fold").rmse_um
            rmse_mlp = full[full.model == "fusion_mlp"].set_index("fold").rmse_um
            difference = (rmse_gat - rmse_mlp).dropna()
            if len(difference) >= 2 and np.any(difference != 0):
                w = stats.wilcoxon(difference)
                tests.append(dict(test="Wilcoxon fold RMSE", model_a="gat", model_b="fusion_mlp",
                                  value_a=rmse_gat.mean(), value_b=rmse_mlp.mean(), p_value=float(w.pvalue)))
        pd.DataFrame(tests).to_csv(self.out / "stat_tests.csv", index=False)

    def save_attribution_and_attention(self, attributions: pd.DataFrame, attention: pd.DataFrame) -> None:
        if not attributions.empty:
            # Percentage share of every sensor within each material (Table 5).
            total = attributions.groupby(["material_type", "modality"]).attribution.sum()
            share = total / total.groupby(level=0).transform("sum") * 100
            share = share.rename("share_percent").reset_index()
            share.to_csv(self.out / "attribution.csv", index=False)
        if not attention.empty:
            mean_attention = attention.groupby("material_type")[["kinematic_attention", "thermal_attention"]].mean()
            mean_attention.to_csv(self.out / "attention.csv")
            attention.to_csv(self.out / "attention_per_trial.csv", index=False)   # full matrices (figures)

    def run_info(self) -> dict:
        """Everything needed to describe and repeat this run."""
        a, cfg = self.args, self.config
        composition = pd.DataFrame([dict(trial=g["trial_id"], material=g["material_type"],
                                         unstable=int(g["y_class"].sum()), zones=len(g["y_class"]),
                                         windows=int(g["x"].shape[1])) for g in self.graphs])
        return {"protocol": a.protocol, "folds": len(self.folds), "seeds": a.seeds, "max_epochs": self.epochs,
                "patience": self.patience, "models": a.models, "ablations": a.ablations, "quick": a.quick,
                "ra_labels_imputed": any(g["ra_imputed"] for g in self.graphs),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "config": {"window_ms": cfg.signal.window_ms, "overlap": cfg.signal.overlap_fraction,
                           "hidden_dim": cfg.train.hidden_dim, "heads": cfg.train.attention_heads,
                           "dropout": cfg.train.dropout, "lr": cfg.train.learning_rate,
                           "weight_decay": cfg.train.weight_decay, "temporal_kernel": cfg.train.temporal_kernel,
                           "regression_weight": cfg.train.regression_weight,
                           "thermal_diffusivity_m2_s": cfg.graph.thermal_diffusivity_m2_s},
                "dataset": composition.to_dict("records"),
                "total_minutes": round((time.time() - self.start_time) / 60, 1)}


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

    ExperimentRunner(args).run()


if __name__ == "__main__":
    main()
