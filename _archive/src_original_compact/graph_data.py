"""Step 2 helpers: turn the prepared caches into trial graphs (Section 2.5 of the draft).

A graph is a plain dict of tensors (see models.py for shapes) plus metadata.
Node features x = [force features (30) | acceleration features (30) | kinematic (4)].
"""
import numpy as np
import pandas as pd
import torch

from config import CONFIG, PipelineConfig, DATA_DIR, CACHE_DIR
from signal_features import FEATURES_PER_CHANNEL

# Column ranges of the node-feature vector, used by the ablations and attributions.
FORCE_COLS = slice(0, 3 * FEATURES_PER_CHANNEL)
ACCEL_COLS = slice(3 * FEATURES_PER_CHANNEL, 6 * FEATURES_PER_CHANNEL)
KINEMATIC_COLS = slice(6 * FEATURES_PER_CHANNEL, 6 * FEATURES_PER_CHANNEL + 4)
KINEMATIC_NAMES = ["spindle_speed", "depth_of_cut", "position_along_wall", "distance_from_wall_end"]


def zone_centres_mm(cfg: PipelineConfig = CONFIG) -> np.ndarray:
    return (np.arange(cfg.process.zone_count) + 0.5) * cfg.process.zone_length_mm


def build_adjacency(cfg: PipelineConfig = CONFIG) -> torch.Tensor:
    """Edge weights [2, N, N]: channel 0 kinematic (Eq. 8), channel 1 thermal heat kernel (Eq. 9).

    Thermal weight between zones i and j:
        w_ij = exp( -d_ij^2 / (4 * alpha * dt_ij) )
    with d_ij the centre distance and dt_ij the time between the tool's visits
    (d_ij / feed speed). Weights below the cut-off are set to zero.
    """
    n = cfg.process.zone_count
    centres = zone_centres_mm(cfg) / 1000.0                       # m
    feed_m_s = cfg.process.feed_rate_mm_min / 60.0 / 1000.0
    alpha = cfg.graph.thermal_diffusivity_m2_s
    adj = np.zeros((2, n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            adj[0, i, j] = 1.0 if abs(i - j) == 1 else 0.0
            if cfg.graph.use_thermal_edges:
                d = abs(centres[i] - centres[j])
                w = np.exp(-d ** 2 / (4 * alpha * d / feed_m_s))
                adj[1, i, j] = w if w >= cfg.graph.thermal_edge_cutoff else 0.0
    return torch.from_numpy(adj)


def load_graphs(cfg: PipelineConfig = CONFIG) -> list[dict]:
    """Load every prepared trial as a graph dict, in manifest order."""
    summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv")
    labels = pd.read_csv(DATA_DIR / "zone_labels.csv")
    adj = build_adjacency(cfg)
    centres = zone_centres_mm(cfg)
    graphs = []
    for row in summary[summary.included].itertuples():
        cache = np.load(CACHE_DIR / f"{row.trial_id}.npz")
        feats = cache["features"]                                          # [N, T, 60]
        n, t = feats.shape[:2]
        kin = np.stack([np.full(n, row.spindle_speed_rpm / 1400.0),
                        np.full(n, row.depth_of_cut_mm),
                        centres / cfg.process.wall_path_length_mm,
                        np.minimum(centres, cfg.process.wall_path_length_mm - centres)
                        / cfg.process.wall_path_length_mm], axis=1)       # [N, 4]
        x = np.concatenate([feats, np.repeat(kin[:, None, :], t, axis=1)], axis=-1)
        lab = labels[labels.trial_id == row.trial_id].sort_values("zone_id")
        graphs.append({
            "x": torch.from_numpy(np.nan_to_num(x).astype(np.float32)),
            "force_raw": torch.from_numpy(cache["force_raw"]),
            "acceleration_raw": torch.from_numpy(cache["acceleration_raw"]),
            "adj": adj.clone(),
            "y_class": torch.tensor(lab.chatter_label.to_numpy(), dtype=torch.float32),
            "y_reg": torch.tensor(lab.Ra_um.to_numpy(), dtype=torch.float32),
            "trial_id": row.trial_id,
            "material_type": row.material_type,
            "condition": f"{int(row.spindle_speed_rpm)}rpm_{row.depth_of_cut_mm:.2f}mm",
            "ra_imputed": bool(lab.Ra_source.str.startswith("IMPUTED").any()),
        })
    return graphs


def apply_variant(g: dict, variant: str) -> dict:
    """Return a copy of graph g modified for one ablation of Table 4 (targets unchanged)."""
    out = dict(g)
    if variant == "without_force_features":
        out["x"] = g["x"].clone()
        out["x"][..., FORCE_COLS] = 0
        out["force_raw"] = torch.zeros_like(g["force_raw"])
    elif variant == "without_acceleration_features":
        out["x"] = g["x"].clone()
        out["x"][..., ACCEL_COLS] = 0
        out["acceleration_raw"] = torch.zeros_like(g["acceleration_raw"])
    elif variant == "without_thermal_edges":
        out["adj"] = g["adj"].clone()
        out["adj"][1] = 0
    elif variant == "without_kinematic_edges":
        out["adj"] = g["adj"].clone()
        out["adj"][0] = 0
    elif variant == "no_spatial_edges":
        out["adj"] = torch.zeros_like(g["adj"])
    elif variant == "material_flag":
        flag = torch.full((*g["x"].shape[:2], 1), 1.0 if g["material_type"] == "WAAM" else 0.0)
        out["x"] = torch.cat([g["x"], flag], dim=-1)
    elif variant not in ("full", "without_temporal_module", "classification_only", "roughness_only"):
        raise ValueError(f"unknown ablation {variant!r}")
    return out
