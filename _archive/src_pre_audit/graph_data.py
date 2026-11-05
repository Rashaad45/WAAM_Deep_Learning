"""
graph_data.py
=============
Turns the feature files of Step 1 into one GRAPH per trial (Section 2.5 of the draft).

What is a trial graph?
----------------------
* NODES  = the 6 zones along the wall (N = 6).
* EDGES  = which zones influence each other. There are two edge types:
     channel 0 - kinematic edges: neighbouring zones along the tool path (Eq. 8)
     channel 1 - thermal edges:   heat-diffusion kernel between zones   (Eq. 9)
* NODE FEATURES = for every zone and every window t, a vector of 64 numbers:
     [ 30 force features | 30 acceleration features | 4 kinematic numbers ]

In the code a graph is a normal Python dict, so you can look inside it with
graph["x"], graph["adj"], ... The keys are:

    "x"                 tensor [N, T, F]     node features per window
    "force_raw"         tensor [N, T, 3, L]  down-sampled raw force windows
    "acceleration_raw"  tensor [N, T, 3, L]  down-sampled raw acceleration windows
    "adj"               tensor [2, N, N]     edge weights (kinematic, thermal)
    "y_class"           tensor [N]           chatter label of every zone (0 / 1)
    "y_reg"             tensor [N]           Ra of every zone in micrometre
    "trial_id", "material_type", "condition", "ra_imputed"   plain information
"""
import numpy as np
import pandas as pd
import torch

from config import CONFIG, PipelineConfig, DATA_DIR, CACHE_DIR
from signal_features import FEATURES_PER_CHANNEL

# Which columns of the feature vector belong to which sensor.
# slice(a, b) means "columns a, a+1, ..., b-1".
FORCE_COLS = slice(0, 3 * FEATURES_PER_CHANNEL)                                       # columns 0-29
ACCEL_COLS = slice(3 * FEATURES_PER_CHANNEL, 6 * FEATURES_PER_CHANNEL)                # columns 30-59
KINEMATIC_COLS = slice(6 * FEATURES_PER_CHANNEL, 6 * FEATURES_PER_CHANNEL + 4)        # columns 60-63
KINEMATIC_NAMES = ["spindle_speed", "depth_of_cut", "position_along_wall", "distance_from_wall_end"]

# All ablation names that apply_variant() understands (Table 4 of the draft).
# Variants listed in ONLY_CHANGE_THE_MODEL do not change the graph itself;
# they are handled when the model or the loss is built.
ONLY_CHANGE_THE_MODEL = ("full", "without_temporal_module", "classification_only", "roughness_only")


class TrialGraphBuilder:
    """Builds the list of trial graphs from data/processed/ and data/zone_labels.csv."""

    def __init__(self, config: PipelineConfig = CONFIG):
        self.config = config

    def zone_centres_mm(self) -> np.ndarray:
        """Position of the centre of every zone along the wall: 2.5, 7.5, ..., 27.5 mm."""
        zone_numbers = np.arange(self.config.process.zone_count)      # 0, 1, ..., 5
        return (zone_numbers + 0.5) * self.config.process.zone_length_mm

    def build_adjacency(self) -> torch.Tensor:
        """Edge weights [2, N, N]: channel 0 kinematic (Eq. 8), channel 1 thermal (Eq. 9).

        Thermal weight between zones i and j (heat kernel):
            w_ij = exp( -d_ij^2 / (4 * alpha * dt_ij) )
        d_ij  = distance between the zone centres (m)
        dt_ij = time between the tool's visits = d_ij / feed speed (s)
        Weights below the cut-off (0.05) are set to zero.
        """
        n = self.config.process.zone_count
        centres_m = self.zone_centres_mm() / 1000.0                     # mm -> m
        feed_m_per_s = self.config.process.feed_rate_mm_min / 60.0 / 1000.0
        alpha = self.config.graph.thermal_diffusivity_m2_s

        adjacency = np.zeros((2, n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue                                            # no self-edges here

                # Kinematic edge: 1 for direct neighbours along the tool path.
                adjacency[0, i, j] = 1.0 if abs(i - j) == 1 else 0.0

                # Thermal edge: heat kernel with dt = distance / feed speed.
                if self.config.graph.use_thermal_edges:
                    distance = abs(centres_m[i] - centres_m[j])
                    weight = np.exp(-distance ** 2 / (4 * alpha * distance / feed_m_per_s))
                    adjacency[1, i, j] = weight if weight >= self.config.graph.thermal_edge_cutoff else 0.0
        return torch.from_numpy(adjacency)

    def kinematic_features(self, spindle_rpm: float, depth_mm: float) -> np.ndarray:
        """4 kinematic numbers per zone -> array [N, 4] (same for every window)."""
        n = self.config.process.zone_count
        centres = self.zone_centres_mm()
        wall = self.config.process.wall_path_length_mm
        columns = [
            np.full(n, spindle_rpm / 1400.0),                        # speed, scaled so 1400 rpm = 1
            np.full(n, depth_mm),                                    # depth of cut (mm)
            centres / wall,                                          # position along the wall (0..1)
            np.minimum(centres, wall - centres) / wall,              # distance to the nearest wall end
        ]
        return np.stack(columns, axis=1)

    def load_all(self) -> list[dict]:
        """Load every prepared trial as a graph dict, in the order of dataset_summary.csv."""
        summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv")
        labels = pd.read_csv(DATA_DIR / "zone_labels.csv")
        adjacency = self.build_adjacency()                          # same edges for every trial

        graphs = []
        for row in summary[summary.included].itertuples():
            cache = np.load(CACHE_DIR / f"{row.trial_id}.npz")
            features = cache["features"]                            # [N, T, 60]
            n_windows = features.shape[1]

            # Repeat the 4 kinematic numbers for every window and glue them to the features.
            kinematic = self.kinematic_features(row.spindle_speed_rpm, row.depth_of_cut_mm)   # [N, 4]
            kinematic_per_window = np.repeat(kinematic[:, None, :], n_windows, axis=1)      # [N, T, 4]
            x = np.concatenate([features, kinematic_per_window], axis=-1)                  # [N, T, 64]

            # Labels of this trial, sorted by zone number.
            trial_labels = labels[labels.trial_id == row.trial_id].sort_values("zone_id")

            graphs.append({
                "x": torch.from_numpy(np.nan_to_num(x).astype(np.float32)),     # NaN -> 0
                "force_raw": torch.from_numpy(cache["force_raw"]),
                "acceleration_raw": torch.from_numpy(cache["acceleration_raw"]),
                "adj": adjacency.clone(),
                "y_class": torch.tensor(trial_labels.chatter_label.to_numpy(), dtype=torch.float32),
                "y_reg": torch.tensor(trial_labels.Ra_um.to_numpy(), dtype=torch.float32),
                "trial_id": row.trial_id,
                "material_type": row.material_type,
                "condition": f"{int(row.spindle_speed_rpm)}rpm_{row.depth_of_cut_mm:.2f}mm",
                "ra_imputed": bool(trial_labels.Ra_source.str.startswith("IMPUTED").any()),
            })
        return graphs


class AblationVariant:
    """Changes a graph for one ablation of Table 4 (the targets stay the same)."""

    @staticmethod
    def apply(graph: dict, variant: str) -> dict:
        """Return a modified COPY of the graph; the original is never changed."""
        new = dict(graph)                         # shallow copy: we replace keys, never edit tensors

        if variant == "without_force_features":
            new["x"] = graph["x"].clone()
            new["x"][..., FORCE_COLS] = 0         # "..." = all leading axes (zones, windows)
            new["force_raw"] = torch.zeros_like(graph["force_raw"])

        elif variant == "without_acceleration_features":
            new["x"] = graph["x"].clone()
            new["x"][..., ACCEL_COLS] = 0
            new["acceleration_raw"] = torch.zeros_like(graph["acceleration_raw"])

        elif variant == "without_thermal_edges":
            new["adj"] = graph["adj"].clone()
            new["adj"][1] = 0                     # remove channel 1 (thermal)

        elif variant == "without_kinematic_edges":
            new["adj"] = graph["adj"].clone()
            new["adj"][0] = 0                     # remove channel 0 (kinematic)

        elif variant == "no_spatial_edges":
            new["adj"] = torch.zeros_like(graph["adj"])   # only self-loops remain inside the layers

        elif variant == "material_flag":
            # Add one extra feature column: 1 for WAAM, 0 for wrought.
            n_zones, n_windows = graph["x"].shape[:2]
            flag_value = 1.0 if graph["material_type"] == "WAAM" else 0.0
            flag = torch.full((n_zones, n_windows, 1), flag_value)
            new["x"] = torch.cat([graph["x"], flag], dim=-1)

        elif variant not in ONLY_CHANGE_THE_MODEL:
            raise ValueError(f"unknown ablation {variant!r}")
        return new
