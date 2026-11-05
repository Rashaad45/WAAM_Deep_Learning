"""
graph_data.py - one graph per trial: nodes = the 3 zones, ONE edge type (kinematic: consecutive zones).

Node features x [N, T, F]: 30 force + 30 acceleration window features + 2 kinematic/geometric numbers
(measured spindle speed, zone-centre position along the wall). Depth of cut is NOT used: its
trial mapping is unverified (DECISION.md). No thermal edges, no Ra target (AUDIT.md section H).
"""
import numpy as np
import pandas as pd
import torch

from config import CONFIG, PipelineConfig, DATA_DIR, CACHE_DIR
from signal_features import FEATURES_PER_CHANNEL

FORCE_COLS = slice(0, 3 * FEATURES_PER_CHANNEL)
ACCEL_COLS = slice(3 * FEATURES_PER_CHANNEL, 6 * FEATURES_PER_CHANNEL)
KINEMATIC_COLS = slice(6 * FEATURES_PER_CHANNEL, 6 * FEATURES_PER_CHANNEL + 2)
VARIANTS = ("full", "no_edges", "no_force", "no_acceleration")


def kinematic_adjacency(n: int) -> torch.Tensor:
    """A_kin[i, j] = 1 if |i - j| = 1 (consecutive zones along the tool path), else 0."""
    a = torch.zeros(n, n)
    i = torch.arange(n - 1)
    a[i, i + 1] = 1.0
    a[i + 1, i] = 1.0
    return a


class TrialGraphBuilder:
    def __init__(self, config: PipelineConfig = CONFIG, threshold_db: float = 15.0):
        self.cfg = config
        self.threshold_db = threshold_db

    def load_all(self, exclude: tuple = ()) -> list[dict]:
        summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv")
        labels = pd.read_csv(DATA_DIR / "zone_labels.csv")
        col = f"label_{int(self.threshold_db)}db"
        graphs = []
        for row in summary[summary.included.astype(bool)].itertuples():
            if row.trial_id in exclude:
                continue
            cache = np.load(CACHE_DIR / f"{row.trial_id}.npz")
            feats = cache["features"]
            n, t, _ = feats.shape
            lab = labels[labels.trial_id == row.trial_id].sort_values("zone_id")
            assert list(lab.zone_id) == list(range(n)), "zone order mismatch"
            centres = ((lab.zone_start_s + lab.zone_end_s) / 2 - row.cut_start_s).to_numpy() / row.cut_duration_s
            kin = np.stack([np.full(n, row.spindle_rpm_measured / 1000.0), centres], axis=1)
            x = np.concatenate([feats, np.repeat(kin[:, None, :], t, axis=1)], axis=-1)
            graphs.append({
                "x": torch.from_numpy(np.nan_to_num(x).astype(np.float32)),
                "force_raw": torch.from_numpy(cache["force_raw"]),
                "acceleration_raw": torch.from_numpy(cache["acceleration_raw"]),
                "adj": kinematic_adjacency(n),
                "y": torch.tensor(lab[col].to_numpy(), dtype=torch.float32),
                "trial_id": row.trial_id,
                "source": "WAAM_LAB",
                "accel_ok": bool(row.accel_ok),
            })
        return graphs


def apply_variant(graph: dict, variant: str) -> dict:
    """Return a modified copy for an ablation (targets unchanged)."""
    g = dict(graph)
    if variant == "no_edges":
        g["adj"] = torch.zeros_like(graph["adj"])
    elif variant == "no_force":
        g["x"] = graph["x"].clone(); g["x"][..., FORCE_COLS] = 0
        g["force_raw"] = torch.zeros_like(graph["force_raw"])
    elif variant == "no_acceleration":
        g["x"] = graph["x"].clone(); g["x"][..., ACCEL_COLS] = 0
        g["acceleration_raw"] = torch.zeros_like(graph["acceleration_raw"])
    elif variant != "full":
        raise ValueError(f"unknown variant {variant!r}")
    return g
