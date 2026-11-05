"""Leave-one-trial-out training for zone-level instability and Ra prediction."""
from pathlib import Path
import argparse
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from scipy.stats import wilcoxon
from config import CONFIG, PipelineConfig
from dataset_graph import WAAMGraphDataset
from models import SpatioTemporalGraphNet, MultimodalFusionMLP, BiLSTMFeatureBaseline, RawCNNBaseline


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def standardize_fold(train_data: list, other_data: list) -> tuple[list, list, float, float]:
    """Fit node-feature and Ra scaling on training trials only."""
    all_x = torch.cat([d.x.reshape(-1, d.x.shape[-1]) for d in train_data], dim=0)
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    valid_ra = torch.cat([d.y_reg[d.regression_mask] for d in train_data])
    if valid_ra.numel() == 0:
        raise ValueError("Training fold has no measured zone-level Ra labels")
    ra_mean = float(valid_ra.mean())
    ra_std = max(float(valid_ra.std(unbiased=False)), 1e-6)
    def transform(data):
        cloned = copy.copy(data)
        cloned.x = (data.x - mean) / std
        cloned.y_reg = (data.y_reg - ra_mean) / ra_std
        return cloned
    return [transform(d) for d in train_data], [transform(d) for d in other_data], ra_mean, ra_std


def make_model(kind: str, sample, config: PipelineConfig, variant: str = "full"):
    feature_dim = sample.x.shape[-1]
    force_channels = sample.force_raw.shape[-2]
    acceleration_channels = sample.acceleration_raw.shape[-2]
    use_thermal = bool(getattr(sample, "has_thermal", False)) and variant != "without_thermal_features"
    common = (feature_dim, force_channels, acceleration_channels)
    if kind in {"gcn", "gat"}:
        return SpatioTemporalGraphNet(*common, config.train.hidden_dim, config.train.attention_heads,
                                      config.train.dropout, kind=kind, temporal_enabled=variant != "without_temporal_module",
                                      use_thermal=use_thermal)
    if kind == "fusion_mlp":
        return MultimodalFusionMLP(*common, config.train.hidden_dim, config.train.dropout, use_thermal=use_thermal)
    if kind == "bilstm":
        return BiLSTMFeatureBaseline(feature_dim, config.train.hidden_dim, config.train.dropout)
    if kind == "cnn1d":
        return RawCNNBaseline(force_channels, acceleration_channels, config.train.hidden_dim, config.train.dropout)
    raise ValueError(f"Unsupported model: {kind}")


def loss_for(model, data, pos_weight: torch.Tensor, config: PipelineConfig, task: str = "joint") -> torch.Tensor:
    logits, ra_pred, _ = model(data)
    class_mask = torch.isfinite(data.y_class)
    if not class_mask.any():
        class_loss = logits.sum() * 0.0
    else:
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[class_mask], data.y_class[class_mask], pos_weight=pos_weight)
    reg_mask = data.regression_mask
    if reg_mask.any():
        reg_loss = torch.nn.functional.mse_loss(ra_pred[reg_mask], data.y_reg[reg_mask])
    else:
        reg_loss = ra_pred.sum() * 0.0
    if task == "classification_only":
        return class_loss
    if task == "roughness_only":
        return reg_loss
    return class_loss + config.train.regression_weight * reg_loss


def fit_one_fold(train_data: list, val_data, test_data, kind: str, config: PipelineConfig,
                 seed: int, device: torch.device, ra_mean: float, ra_std: float,
                 variant: str = "full") -> list[dict]:
    seed_everything(seed)
    model = make_model(kind, train_data[0], config, variant).to(device)
    classes = torch.cat([d.y_class[torch.isfinite(d.y_class)] for d in train_data])
    positives = float((classes == 1).sum())
    negatives = float((classes == 0).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(config.train.epochs):
        model.train()
        order = np.random.permutation(len(train_data))
        for idx in order:
            graph = train_data[int(idx)].to(device)
            optimizer.zero_grad(set_to_none=True)
            task = variant if variant in {"classification_only", "roughness_only"} else "joint"
            loss = loss_for(model, graph, pos_weight, config, task)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            task = variant if variant in {"classification_only", "roughness_only"} else "joint"
            val_losses = [float(loss_for(model, d.to(device), pos_weight, config, task)) for d in val_data]
        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.train.patience:
                break
    if best_state is None:
        raise RuntimeError("Inner-fold early stopping failed to select a model")
    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for graph in test_data:
            graph_device = graph.to(device)
            logits, ra_scaled, embeddings = model(graph_device)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            ra_values = (ra_scaled.cpu().numpy() * ra_std) + ra_mean
            edge_scores = edge_attention_summary(graph_device, model) if kind == "gat" else (float("nan"), float("nan"))
            for zone in range(len(probabilities)):
                rows.append({"trial_id": graph.trial_id, "material_type": graph.material_type, "zone_id": zone,
                             "y_class": float(graph.y_class[zone]), "probability": float(probabilities[zone]),
                             "y_reg": float(graph.y_reg[zone] * ra_std + ra_mean) if bool(graph.regression_mask[zone]) else np.nan,
                             "predicted_Ra_um": float(ra_values[zone]), "embedding": embeddings[zone].cpu().numpy().tolist(),
                             "kinematic_attention_mean": edge_scores[0], "thermal_attention_mean": edge_scores[1]})
    return rows


def edge_attention_summary(data, model) -> tuple[float, float]:
    """Average GAT coefficients over kinematic and thermal edge types."""
    if not model.last_edge_attention:
        return float("nan"), float("nan")
    original = {}
    for edge, attr in zip(data.edge_index.t().tolist(), data.edge_attr.tolist()):
        original[(edge[0], edge[1])] = attr
    sums = np.zeros(2, dtype=float)
    counts = np.zeros(2, dtype=float)
    for edge_index, alpha in model.last_edge_attention:
        coefficients = alpha.mean(dim=-1).cpu().numpy()
        for edge, value in zip(edge_index.t().cpu().tolist(), coefficients):
            types = original.get((edge[0], edge[1]))
            if types is None:
                continue
            for channel in range(2):
                if types[channel] > 0:
                    sums[channel] += float(value) * float(types[channel])
                    counts[channel] += float(types[channel])
    return tuple((sums / np.maximum(counts, 1e-12)).tolist())


def classification_metrics(frame: pd.DataFrame) -> dict:
    valid = frame[np.isfinite(frame.y_class)]
    if valid.empty:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "roc_auc": float("nan")}
    y = valid.y_class.astype(int).to_numpy()
    p = valid.probability.to_numpy()
    pred = p >= 0.5
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")}


def regression_metrics(frame: pd.DataFrame) -> dict:
    valid = frame[np.isfinite(frame.y_reg)]
    if valid.empty:
        return {"rmse_um": float("nan"), "mae_um": float("nan"), "r2": float("nan")}
    y, pred = valid.y_reg.to_numpy(), valid.predicted_Ra_um.to_numpy()
    return {"rmse_um": float(np.sqrt(mean_squared_error(y, pred))), "mae_um": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan")}


def bootstrap_fold_summary(metrics_frame: pd.DataFrame, output_path: Path, seed: int) -> pd.DataFrame:
    """Report fold mean, SD, and percentile bootstrap CI for each metric."""
    rng = np.random.default_rng(seed)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "rmse_um", "mae_um", "r2"]
    rows = []
    for (model, variant), group in metrics_frame.groupby(["model", "variant"]):
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            rows.append({"model": model, "variant": variant, "metric": metric, "fold_mean": float(values.mean()),
                         "fold_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                         "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                         "bootstrap_ci_high": float(np.quantile(draws, 0.975)), "folds": len(values)})
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def run_classical_baselines(graphs: list, output_dir: Path, seed: int) -> pd.DataFrame:
    """LOTO classical models on hand-crafted temporal-mean zone features."""
    classifiers = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        "random_forest_classifier": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed)),
    }
    regressors = {
        "ridge_regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest_regressor": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(C=1.0, kernel="rbf")),
    }
    output_rows = []
    for test_idx, test_graph in enumerate(graphs):
        train_graphs = [g for i, g in enumerate(graphs) if i != test_idx]
        train_x = np.concatenate([g.x.mean(dim=1).numpy() for g in train_graphs])
        test_x = test_graph.x.mean(dim=1).numpy()
        train_class = np.concatenate([g.y_class.numpy() for g in train_graphs])
        class_mask = np.isfinite(train_class)
        test_class = test_graph.y_class.numpy()
        for name, estimator in classifiers.items():
            if np.unique(train_class[class_mask]).size < 2:
                continue
            estimator.fit(train_x[class_mask], train_class[class_mask].astype(int))
            model = estimator[-1] if hasattr(estimator, "steps") else estimator
            if hasattr(model, "predict_proba"):
                score = estimator.predict_proba(test_x)[:, 1]
            else:
                score = estimator.decision_function(test_x)
                score = 1 / (1 + np.exp(-score))
            for zone in range(len(test_x)):
                output_rows.append({"trial_id": test_graph.trial_id, "material_type": test_graph.material_type,
                                    "zone_id": zone, "model": name, "y_class": test_class[zone],
                                    "probability": float(score[zone]), "y_reg": np.nan,
                                    "predicted_Ra_um": np.nan, "embedding": "[]"})
        train_ra = np.concatenate([g.y_reg.numpy() for g in train_graphs])
        ra_mask = np.isfinite(train_ra)
        for name, estimator in regressors.items():
            if ra_mask.sum() < 2:
                continue
            estimator.fit(train_x[ra_mask], train_ra[ra_mask])
            test_ra = test_graph.y_reg.numpy()
            predicted = estimator.predict(test_x)
            for zone in range(len(test_x)):
                output_rows.append({"trial_id": test_graph.trial_id, "material_type": test_graph.material_type,
                                    "zone_id": zone, "model": name, "y_class": np.nan,
                                    "probability": np.nan, "y_reg": test_ra[zone],
                                    "predicted_Ra_um": float(predicted[zone]), "embedding": "[]"})
    predictions = pd.DataFrame(output_rows)
    predictions.to_csv(output_dir / "classical_oof_predictions.csv", index=False)
    summaries = []
    for model, frame in predictions.groupby("model"):
        row = {"model": model}
        if frame.y_class.notna().any():
            row.update(classification_metrics(frame))
        if frame.y_reg.notna().any():
            row.update(regression_metrics(frame))
        summaries.append(row)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "classical_loto_metrics.csv", index=False)
    return summary_frame


def apply_variant(graph, variant: str):
    """Apply the manuscript's modality/edge ablations without changing targets."""
    allowed = {"full", "without_thermal_features", "without_force_features", "without_acceleration_features",
               "without_thermal_edges", "without_kinematic_edges", "no_spatial_edges", "without_temporal_module",
               "material_flag", "classification_only", "roughness_only"}
    if variant not in allowed:
        raise ValueError(f"Unknown ablation {variant!r}; supported values are {sorted(allowed)}")
    has_thermal = bool(getattr(graph, "has_thermal", False))
    if variant == "without_thermal_features" and not has_thermal:
        raise ValueError("Ablation 'without_thermal_features' requires a thermal modality, but this dataset has none (has_thermal=False)")
    if variant == "without_thermal_edges" and not bool((graph.edge_attr[:, 1] > 0).any()):
        raise ValueError("Ablation 'without_thermal_edges' requires thermal-diffusion graph edges, but none are present (enable graph.include_thermal_edges with a verified thermal_diffusivity_m2_s)")
    data = copy.copy(graph)
    clone_fields = ["x", "force_raw", "acceleration_raw", "edge_index", "edge_attr"]
    if has_thermal:
        clone_fields.append("thermal_raw")
    for name in clone_fields:
        setattr(data, name, getattr(graph, name).clone())
    force_end = graph.force_raw.shape[-2] * 9
    acceleration_end = force_end + graph.acceleration_raw.shape[-2] * 9
    thermal_end = acceleration_end + 4
    if variant == "without_force_features":
        data.x[:, :, :force_end] = 0
        data.force_raw.zero_()
    elif variant == "without_acceleration_features":
        data.x[:, :, force_end:acceleration_end] = 0
        data.acceleration_raw.zero_()
    elif variant == "without_thermal_features":
        data.x[:, :, acceleration_end:thermal_end] = 0
        data.thermal_raw.zero_()
    elif variant in {"without_thermal_edges", "without_kinematic_edges"}:
        channel = 1 if variant == "without_thermal_edges" else 0
        data.edge_attr[:, channel] = 0
        keep = data.edge_attr.sum(dim=1) > 0
        data.edge_index = data.edge_index[:, keep]
        data.edge_attr = data.edge_attr[keep]
    elif variant == "no_spatial_edges":
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0, 2), dtype=torch.float32)
    elif variant == "material_flag":
        flag = 1.0 if "waam" in graph.material_type.lower() else 0.0
        material = torch.full((*data.x.shape[:2], 1), flag, dtype=data.x.dtype)
        data.x = torch.cat((data.x, material), dim=-1)
    return data


def run_leave_one_trial_out(dataset: WAAMGraphDataset, config: PipelineConfig, model_kinds: tuple[str, ...],
                            variants: tuple[str, ...] = ("full",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataset) < 3:
        raise ValueError("Leave-one-trial-out with inner validation requires at least three trials")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_full = [dataset[i] for i in range(len(dataset))]
    trials = [str(graph.trial_id) for graph in raw_full]
    if len(set(trials)) != len(trials):
        raise ValueError("Expected one graph per unique trial_id")
    device = torch.device(config.device)
    classical_metrics = run_classical_baselines(raw_full, config.output_dir, config.train.seeds[0])
    if not classical_metrics.empty:
        print("Classical LOTO baselines:\n", classical_metrics.to_string(index=False))
    all_predictions, all_metrics = [], []
    rng = np.random.default_rng(config.train.seeds[0])
    for variant in variants:
        raw = [apply_variant(graph, variant) for graph in raw_full]
        for test_idx, test_graph in enumerate(raw):
            available = [i for i in range(len(raw)) if i != test_idx]
            validation_idx = int(rng.choice(available))
            train_idx = [i for i in available if i != validation_idx]
            train_std, eval_std, ra_mean, ra_std = standardize_fold([raw[i] for i in train_idx], [raw[validation_idx], test_graph])
            val_std, test_std = eval_std[0], eval_std[1]
            for kind in model_kinds:
                seed_predictions = []
                for seed in config.train.seeds:
                    rows = fit_one_fold(train_std, [val_std], [test_std], kind, config, seed, device, ra_mean, ra_std, variant)
                    for row in rows:
                        row["model"] = kind
                        row["variant"] = variant
                        row["seed"] = seed
                    seed_predictions.extend(rows)
                all_predictions.extend(seed_predictions)
                fold_frame = pd.DataFrame(seed_predictions)
                metrics = {**classification_metrics(fold_frame), **regression_metrics(fold_frame),
                           "model": kind, "variant": variant, "test_trial": trials[test_idx]}
                all_metrics.append(metrics)
                print(f"LOTO {test_idx + 1}/{len(raw)} {trials[test_idx]} {kind} {variant}: {metrics}")
    predictions = pd.DataFrame(all_predictions)
    metrics_frame = pd.DataFrame(all_metrics)
    predictions.to_csv(config.output_dir / "oof_zone_predictions.csv", index=False)
    material_metrics = []
    for (kind, variant, material), subset in predictions.groupby(["model", "variant", "material_type"]):
        # Average replicate seeds within each physical trial-zone before scoring.
        collapsed = subset.groupby(["trial_id", "zone_id", "y_class", "y_reg"], dropna=False, as_index=False).agg(
            probability=("probability", "mean"), predicted_Ra_um=("predicted_Ra_um", "mean"))
        material_metrics.append({**classification_metrics(collapsed), **regression_metrics(collapsed),
                                 "model": kind, "variant": variant, "material_type": material})
    pd.DataFrame(material_metrics).to_csv(config.output_dir / "loto_metrics_by_material.csv", index=False)
    metrics_frame.to_csv(config.output_dir / "loto_fold_metrics.csv", index=False)
    bootstrap_fold_summary(metrics_frame, config.output_dir / "loto_confidence_intervals.csv", config.train.seeds[0])
    paired_tests = []
    comparisons = [("gat", "fusion_mlp", variant, variant) for variant in variants]
    comparisons.extend(("gat", "gat", "full", variant) for variant in variants if variant != "full")
    for left_model, right_model, left_variant, right_variant in comparisons:
        left = metrics_frame[(metrics_frame.model == left_model) & (metrics_frame.variant == left_variant)].set_index("test_trial")
        right = metrics_frame[(metrics_frame.model == right_model) & (metrics_frame.variant == right_variant)].set_index("test_trial")
        common = left.index.intersection(right.index)
        for metric in ("f1", "rmse_um"):
            if len(common) < 2:
                continue
            a, b = left.loc[common, metric].to_numpy(), right.loc[common, metric].to_numpy()
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 2 or np.allclose(a[valid], b[valid]):
                continue
            statistic, p_value = wilcoxon(a[valid], b[valid], alternative="two-sided")
            paired_tests.append({"model_a": left_model, "variant_a": left_variant, "model_b": right_model,
                                 "variant_b": right_variant, "metric": metric, "n_trials": int(valid.sum()),
                                 "wilcoxon_statistic": float(statistic), "p_value": float(p_value)})
    pd.DataFrame(paired_tests).to_csv(config.output_dir / "paired_wilcoxon_tests.csv", index=False)
    numeric_columns = metrics_frame.select_dtypes(include=[np.number]).columns.tolist()
    summary = metrics_frame.groupby(["model", "variant"])[numeric_columns].agg(["mean", "std"]).round(6)
    summary.to_csv(config.output_dir / "loto_summary.csv")
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump({"models": model_kinds, "seeds": config.train.seeds, "outer_cv": config.train.outer_cv,
                   "config": {"window_ms": config.signal.window_ms, "acceleration_fs_hz": config.signal.acceleration_sampling_rate_hz,
                              "thermal_fps": config.signal.thermal_frame_rate_hz, "thermal_diffusivity_m2_s": config.graph.thermal_diffusivity_m2_s}}, stream, indent=2)
    return metrics_frame, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CONFIG.data_root)
    parser.add_argument("--manifest", type=Path, required=True, help="One row per trial; see dataset_graph.py NPZ schema")
    parser.add_argument("--labels", type=Path, required=True, help="Zone labels: trial_id, zone_id, chatter_label, measured_Ra_um")
    parser.add_argument("--output", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--models", nargs="+", choices=["gcn", "gat", "fusion_mlp", "cnn1d", "bilstm"], default=["gcn", "gat", "fusion_mlp", "cnn1d", "bilstm"])
    parser.add_argument("--ablations", nargs="+", default=["full"], help="full, without_thermal_features, without_force_features, without_acceleration_features, without_thermal_edges, without_kinematic_edges, no_spatial_edges, without_temporal_module, material_flag, classification_only, roughness_only")
    parser.add_argument("--window-ms", type=float, required=True, help="Fill the unresolved window duration from the acquisition protocol")
    parser.add_argument("--acceleration-fs", type=float, required=True)
    parser.add_argument("--include-thermal", action="store_true", help="Enable the thermal modality; requires --thermal-fps, --thermal-diffusivity, --ambient-c, and --thermal-radiometric")
    parser.add_argument("--thermal-fps", type=float)
    parser.add_argument("--thermal-diffusivity", type=float, help="Verified Al 5356 diffusivity in m^2/s; only used if --include-thermal and thermal edges are enabled")
    parser.add_argument("--ambient-c", type=float, help="Only required with --include-thermal")
    parser.add_argument("--thermal-radiometric", action="store_true", help="Confirm NPZ temperature arrays are radiometrically calibrated")
    args = parser.parse_args()
    if args.include_thermal and (args.thermal_fps is None or args.ambient_c is None or not args.thermal_radiometric):
        parser.error("--include-thermal requires --thermal-fps, --ambient-c, and --thermal-radiometric")
    CONFIG.output_dir = args.output
    CONFIG.signal.window_ms = args.window_ms
    CONFIG.signal.acceleration_sampling_rate_hz = args.acceleration_fs
    CONFIG.vision.include_thermal = args.include_thermal
    if args.include_thermal:
        CONFIG.signal.thermal_frame_rate_hz = args.thermal_fps
        CONFIG.vision.ambient_temperature_c = args.ambient_c
        CONFIG.vision.thermal_is_radiometric = args.thermal_radiometric
    if args.thermal_diffusivity is not None:
        CONFIG.graph.thermal_diffusivity_m2_s = args.thermal_diffusivity
        CONFIG.graph.include_thermal_edges = True
    dataset = WAAMGraphDataset(args.root, args.manifest, args.labels, CONFIG)
    run_leave_one_trial_out(dataset, CONFIG, tuple(args.models), tuple(args.ablations))


if __name__ == "__main__":
    main()
