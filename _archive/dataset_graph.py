"""Trial graphs: six physical cutting zones, temporal windows, and zone labels.

The trial manifest points to synchronised per-trial NPZ tensors. This keeps raw
instrument export decoding separate from scientifically consequential timing,
calibration, and label decisions that are not specified in the draft.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from config import CONFIG, PipelineConfig
from utils_signal import load_numeric_csv, multichannel_features, thermal_summary
from utils_vision import track_interface_roi


MANIFEST_COLUMNS = {"trial_id", "material_type", "spindle_speed_rpm", "depth_of_cut_mm", "feature_path"}
LABEL_COLUMNS = {"trial_id", "zone_id", "chatter_label", "measured_Ra_um"}


def heat_kernel_edges(config: PipelineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Create kinematic and heat-diffusion edge weights between zone centres."""
    graph = config.graph
    pairs, attrs = [], []
    if graph.include_thermal_edges and graph.thermal_diffusivity_m2_s is None:
        raise ValueError("Set graph.thermal_diffusivity_m2_s from a verified material source, or disable thermal edges explicitly")
    feed_mm_s = graph.feed_rate_mm_min / 60.0
    coords = (np.arange(graph.zone_count) + 0.5) * graph.zone_length_mm
    for i in range(graph.zone_count):
        for j in range(graph.zone_count):
            if i == j:
                continue
            distance_m = abs(coords[i] - coords[j]) / 1000.0
            kinematic = graph.kinematic_edge_weight if abs(i - j) == 1 else 0.0
            thermal = 0.0
            if graph.include_thermal_edges:
                delta_t = abs(coords[i] - coords[j]) / feed_mm_s
                alpha = float(graph.thermal_diffusivity_m2_s)
                thermal = float(np.exp(-(distance_m ** 2) / max(4 * alpha * delta_t, 1e-20)))
                if thermal < graph.thermal_edge_cutoff:
                    thermal = 0.0
            if kinematic > 0 or thermal > 0:
                pairs.append((i, j))
                attrs.append((kinematic, thermal))
    return torch.tensor(pairs, dtype=torch.long).t().contiguous(), torch.tensor(attrs, dtype=torch.float32)


def make_zone_graph(features: np.ndarray, force_raw: np.ndarray, acceleration_raw: np.ndarray,
                    thermal_raw: np.ndarray | None, chatter: np.ndarray, roughness: np.ndarray,
                    metadata: dict, config: PipelineConfig = CONFIG) -> Data:
    """Build one graph per trial; features are [zone, window, feature].

    thermal_raw is None when no calibrated thermal modality is available; the
    graph is then built force+acceleration-only rather than fabricating a
    thermal tensor.
    """
    n_zones = config.graph.zone_count
    if features.ndim != 3 or features.shape[0] != n_zones:
        raise ValueError(f"Expected features [zones={n_zones}, windows, features], got {features.shape}")
    if len(chatter) != n_zones or len(roughness) != n_zones:
        raise ValueError("Zone-level label count must equal configured zone count")
    edge_index, edge_attr = heat_kernel_edges(config)
    data = Data(x=torch.tensor(features, dtype=torch.float32),
                force_raw=torch.tensor(force_raw, dtype=torch.float32),
                acceleration_raw=torch.tensor(acceleration_raw, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr,
                y_class=torch.tensor(chatter, dtype=torch.float32),
                y_reg=torch.tensor(roughness, dtype=torch.float32),
                regression_mask=torch.tensor(np.isfinite(roughness), dtype=torch.bool),
                zone_pos=torch.tensor((np.arange(n_zones) + 0.5) * config.graph.zone_length_mm, dtype=torch.float32),
                trial_id=str(metadata["trial_id"]), material_type=str(metadata["material_type"]),
                has_thermal=thermal_raw is not None)
    if thermal_raw is not None:
        data.thermal_raw = torch.tensor(thermal_raw, dtype=torch.float32)
    return data


def features_from_npz(path: Path, row: pd.Series, config: PipelineConfig = CONFIG) -> tuple[np.ndarray, ...]:
    """Extract paper features from aligned raw windows in the documented cache schema.

    NPZ keys: force_windows [6,T,3,Lf], acceleration_windows [6,T,3,La],
    window_bounds_s [T,2], and optionally temperature_windows_c [6,T,H,W].
    Arrays must be synchronised and calibrated before packaging; no guessed
    resampling is done. Thermal is only read when config.vision.include_thermal
    is set, so a dataset with no calibrated IR data trains force+acceleration-only
    instead of raising or fabricating thermal features.
    """
    with np.load(path) as data:
        needed = {"force_windows", "acceleration_windows", "window_bounds_s"}
        missing = needed - set(data.files)
        if missing:
            raise ValueError(f"{path} lacks NPZ arrays: {sorted(missing)}")
        force = data["force_windows"].astype(np.float32)
        acceleration = data["acceleration_windows"].astype(np.float32)
        bounds = data["window_bounds_s"].astype(float)
        has_thermal_array = "temperature_windows_c" in data.files
        thermal = data["temperature_windows_c"].astype(np.float32) if has_thermal_array else None
    use_thermal = config.vision.include_thermal
    if use_thermal and not has_thermal_array:
        raise ValueError(f"{path} lacks 'temperature_windows_c' but config.vision.include_thermal=True")
    expected = config.graph.zone_count
    if force.shape[:2] != acceleration.shape[:2] or force.shape[0] != expected:
        raise ValueError("Force and acceleration tensors must share [zone, window] dimensions")
    if force.ndim != 4 or acceleration.ndim != 4 or force.shape[2] != 3 or acceleration.shape[2] != 3:
        raise ValueError("Force and acceleration arrays must have shape [6, windows, 3 axes, samples]")
    if bounds.shape != (force.shape[1], 2):
        raise ValueError("window_bounds_s must have shape [windows, 2]")
    fs_force = config.signal.force_sampling_rate_hz
    fs_acc = config.signal.acceleration_sampling_rate_hz
    if fs_acc is None:
        raise ValueError("Set acceleration_sampling_rate_hz from the recorded CSV timestamps or acquisition log")
    if config.signal.window_ms is None:
        raise ValueError("Set signal.window_ms from the study protocol; the draft leaves it as [X] ms")
    observed_ms = float(np.median(bounds[:, 1] - bounds[:, 0]) * 1000.0)
    if not np.isclose(observed_ms, config.signal.window_ms, rtol=0.05, atol=0.25):
        raise ValueError(f"Configured window_ms={config.signal.window_ms:g} does not match NPZ window duration {observed_ms:g} ms")
    if use_thermal:
        if thermal.shape[:2] != force.shape[:2]:
            raise ValueError("Temperature tensor must share [zone, window] dimensions with force/acceleration")
        if thermal.ndim != 4 or min(thermal.shape[-2:]) < 2:
            raise ValueError("Temperature arrays must have shape [6, windows, height, width]")
        fs_thermal = config.signal.thermal_frame_rate_hz
        if fs_thermal is None:
            raise ValueError("Set thermal_frame_rate_hz from the IR acquisition metadata")
        if not config.vision.thermal_is_radiometric:
            raise ValueError("Thermal arrays need confirmed radiometric calibration before use")
        if config.vision.ambient_temperature_c is None:
            raise ValueError("Set ambient_temperature_c from the measured laboratory conditions")
    force_feats, acc_feats, thermal_feats = [], [], []
    for z in range(expected):
        row_f, row_a, row_t = [], [], []
        for t in range(force.shape[1]):
            row_f.append(multichannel_features(force[z, t].T, fs_force, float(row.spindle_speed_rpm), config.signal))
            row_a.append(multichannel_features(acceleration[z, t].T, fs_acc, float(row.spindle_speed_rpm), config.signal))
            if use_thermal:
                row_t.append(thermal_summary(thermal[z, t], ambient_c=config.vision.ambient_temperature_c,
                                             window_seconds=(bounds[t, 1] - bounds[t, 0])))
        force_feats.append(row_f)
        acc_feats.append(row_a)
        if use_thermal:
            thermal_feats.append(row_t)
    n_windows = force.shape[1]
    positions = (np.arange(expected) + 0.5) * config.graph.zone_length_mm
    kinematic = np.broadcast_to(np.column_stack((np.full(expected, float(row.spindle_speed_rpm) / 1400.0),
                                                 np.full(expected, float(row.depth_of_cut_mm)),
                                                 positions / config.graph.wall_path_length_mm,
                                                 np.abs(positions - config.graph.clamp_position_mm) / config.graph.wall_path_length_mm)),
                                (n_windows, expected, 4)).transpose(1, 0, 2)
    blocks = [np.asarray(force_feats), np.asarray(acc_feats)]
    if use_thermal:
        blocks.append(np.asarray(thermal_feats))
    blocks.append(kinematic)
    handcrafted = np.concatenate(blocks, axis=-1)
    # Raw calibrated modality tensors are retained for trainable 1D/2D encoders.
    thermal_raw = thermal[:, :, None, :, :] if use_thermal else None
    return handcrafted.astype(np.float32), force, acceleration, thermal_raw


class WAAMGraphDataset(Dataset):
    """Load one graph per trial from `trial_manifest.csv` and `zone_labels.csv`."""
    def __init__(self, root: str | Path, manifest_path: str | Path, labels_path: str | Path,
                 config: PipelineConfig = CONFIG):
        self.root = Path(root)
        self.config = config
        self.manifest = pd.read_csv(manifest_path)
        self.labels = pd.read_csv(labels_path)
        for name, frame, required in (("manifest", self.manifest, MANIFEST_COLUMNS), ("labels", self.labels, LABEL_COLUMNS)):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if self.manifest.trial_id.astype(str).duplicated().any():
            raise ValueError("trial_manifest.csv must contain exactly one row per trial")
        if self.labels.duplicated(["trial_id", "zone_id"]).any():
            raise ValueError("zone_labels.csv has duplicate trial_id/zone_id labels")
        super().__init__(root=None)
        # PyG's Dataset initializer assigns ``self.root`` from its ``root``
        # argument. Since this class manages its own manifest/cache layout,
        # restore the caller-supplied feature directory after initialization.
        # Otherwise relative feature paths resolve against None/garbled values
        # on Windows (for example ``???\\trial_04_...npz``).
        self.root = Path(root).expanduser().resolve()

    def len(self) -> int:
        return len(self.manifest)

    def get(self, index: int) -> Data:
        row = self.manifest.iloc[index]
        trial_id = str(row.trial_id)
        path = Path(str(row.feature_path))
        if not path.is_absolute():
            path = self.root / path
        labels = self.labels[self.labels.trial_id.astype(str) == trial_id].sort_values("zone_id")
        if len(labels) != self.config.graph.zone_count or not np.array_equal(labels.zone_id.to_numpy(), np.arange(self.config.graph.zone_count)):
            raise ValueError(f"Trial {trial_id} needs one label row per zone_id 0..{self.config.graph.zone_count - 1}")
        features, force, acceleration, thermal = features_from_npz(path, row, self.config)
        chatter = labels.chatter_label.to_numpy(dtype=float)
        roughness = labels.measured_Ra_um.to_numpy(dtype=float)
        if np.any(~np.isin(chatter[np.isfinite(chatter)], [0.0, 1.0])):
            raise ValueError(f"Trial {trial_id} chatter labels must be 0, 1, or missing")
        meta = row.to_dict()
        meta["trial_id"] = trial_id
        return make_zone_graph(features, force, acceleration, thermal, chatter, roughness, meta, self.config)


def _read_timed_channels(path: Path, fs: float, time_offset_s: float, skip_unit_row: bool) -> tuple[np.ndarray, np.ndarray]:
    # DynoWare exports may start with a metadata line ("Cycle No:,..."), then
    # a channel header, a units row, and numeric samples. Keep the channel names
    # so Fx/Fy/Fz are selected instead of accidentally taking the trailing moment
    # channels (Mx/My/Mz).
    first_line = path.open("r", encoding="utf-8-sig", errors="replace").readline()
    if "cycle no" in first_line.lower():
        frame = pd.read_csv(path, skiprows=[0, 2], encoding="utf-8-sig")
        names = [str(column).strip() for column in frame.columns]
        frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        values = frame.to_numpy(dtype=float)
    else:
        values, names = load_numeric_csv(path, skip_unit_row=skip_unit_row)
    time_col = next((i for i, name in enumerate(names) if "time" in name.lower()), None)
    normalized = {name.strip().lower(): i for i, name in enumerate(names)}
    channel_names = ("fx", "fy", "fz")
    channel_cols = [normalized[name] for name in channel_names if name in normalized]
    if len(channel_cols) != 3:
        channel_names = ("acc-x", "acc-y", "acc-z")
        channel_cols = [normalized[name] for name in channel_names if name in normalized]
    if time_col is None:
        times = np.arange(len(values), dtype=float) / fs
        channels = values[:, channel_cols] if len(channel_cols) == 3 else values[:, -3:]
    else:
        times = values[:, time_col]
        channels = (values[:, channel_cols] if len(channel_cols) == 3
                    else np.delete(values, time_col, axis=1)[:, -3:])
    if channels.shape[1] != 3 or not np.all(np.diff(times) > 0):
        raise ValueError(f"{path} must contain three channels and increasing timestamps")
    return times + time_offset_s, channels


def _sample_window(times: np.ndarray, values: np.ndarray, start: float, end: float, fs: float) -> np.ndarray:
    count = int(round((end - start) * fs))
    target = start + np.arange(count, dtype=float) / fs
    if target[0] < times[0] or target[-1] > times[-1]:
        raise ValueError(f"Sensor file does not cover requested interval {start:.3f}–{end:.3f} s")
    return np.stack([np.interp(target, times, values[:, axis]) for axis in range(values.shape[1])], axis=0).astype(np.float32)


def prepare_raw_trial_cache(raw_manifest_path: str | Path, output_dir: str | Path,
                            config: PipelineConfig = CONFIG) -> Path:
    """Synchronise timed CSV signals and radiometric NPZ frames into fixed 6-zone windows.

    Raw manifest columns: trial_id, material_type, spindle_speed_rpm,
    depth_of_cut_mm, force_csv, acceleration_csv, cut_start_time_s, and
    optionally thermal_npz. Optional stream offsets and thermal_time_key
    support separate acquisition clocks. Kistler binary DWD and proprietary
    IRGD files must first be exported to numeric CSV and calibrated
    temperature NPZ, respectively. When config.vision.include_thermal is
    False (the default -- no calibrated thermal data is available for this
    dataset), thermal_npz is ignored even if present and no temperature
    array is written to the cache.
    """
    source = pd.read_csv(raw_manifest_path)
    required = {"trial_id", "material_type", "spindle_speed_rpm", "depth_of_cut_mm", "force_csv",
                "acceleration_csv", "cut_start_time_s"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Raw manifest missing columns: {sorted(missing)}")
    use_thermal = config.vision.include_thermal
    if config.signal.window_ms is None or config.signal.acceleration_sampling_rate_hz is None:
        raise ValueError("Set window_ms and acceleration_sampling_rate_hz before preparation")
    if use_thermal:
        if config.signal.thermal_frame_rate_hz is None:
            raise ValueError("Set thermal_frame_rate_hz before preparation (or disable vision.include_thermal)")
        if not config.vision.thermal_is_radiometric:
            raise ValueError("Confirm the infrared export is radiometric and set thermal_is_radiometric=True")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared_rows = []
    width_s = config.signal.window_ms / 1000.0
    hop_s = width_s * (1 - config.signal.overlap_fraction)
    zone_duration = config.graph.zone_length_mm / (config.graph.feed_rate_mm_min / 60.0)
    windows_per_zone = int(np.floor((zone_duration - width_s) / hop_s)) + 1
    if windows_per_zone < 1:
        raise ValueError("Configured time window exceeds the 5 mm zone traversal duration")
    for _, row in source.iterrows():
        resolve = lambda column: (Path(str(row[column])) if Path(str(row[column])).is_absolute()
                                  else Path(raw_manifest_path).parent / str(row[column]))
        try:
            force_t, force = _read_timed_channels(resolve("force_csv"), config.signal.force_sampling_rate_hz,
                                                  float(row.get("force_time_offset_s", 0.0)), bool(row.get("force_skip_unit_row", False)))
            accel_t, acceleration = _read_timed_channels(resolve("acceleration_csv"), config.signal.acceleration_sampling_rate_hz,
                                                          float(row.get("acceleration_time_offset_s", 0.0)), bool(row.get("acceleration_skip_unit_row", True)))
            thermal_cube = thermal_t = None
            if use_thermal:
                thermal_path = resolve("thermal_npz")
                with np.load(thermal_path) as archive:
                    if "temperature_c" not in archive:
                        raise ValueError(f"{thermal_path} requires a calibrated 'temperature_c' array")
                    thermal_cube = archive["temperature_c"].astype(np.float32)
                    thermal_t = archive["time_s"].astype(float) if "time_s" in archive else np.arange(len(thermal_cube)) / config.signal.thermal_frame_rate_hz
                thermal_t = thermal_t + float(row.get("thermal_time_offset_s", 0.0))
                if thermal_cube.ndim != 3 or len(thermal_t) != len(thermal_cube):
                    raise ValueError("thermal_npz must contain [frames,height,width] temperature_c and matching time_s")
            force_zones, acc_zones, temp_zones, bounds = [], [], [], []
            cut_start = float(row.cut_start_time_s)
            for zone in range(config.graph.zone_count):
                f_windows, a_windows, t_windows = [], [], []
                zone_start = cut_start + zone * zone_duration
                for step in range(windows_per_zone):
                    start = zone_start + step * hop_s
                    end = start + width_s
                    f_windows.append(_sample_window(force_t, force, start, end, config.signal.force_sampling_rate_hz))
                    a_windows.append(_sample_window(accel_t, acceleration, start, end, config.signal.acceleration_sampling_rate_hz))
                    if use_thermal:
                        frame_ix = np.flatnonzero((thermal_t >= start) & (thermal_t < end))
                        if frame_ix.size == 0:
                            frame_ix = np.asarray([int(np.argmin(np.abs(thermal_t - (start + end) / 2)))])
                        cropped = []
                        for frame_index in frame_ix:
                            image = thermal_cube[frame_index]
                            image, _ = track_interface_roi(image, config.vision.roi_xywh)
                            cropped.append(image)
                        t_windows.append(np.mean(cropped, axis=0).astype(np.float32))
                    if zone == 0:
                        bounds.append((start, end))
                force_zones.append(f_windows)
                acc_zones.append(a_windows)
                if use_thermal:
                    temp_zones.append(t_windows)
            feature_path = out / f"{row.trial_id}_synchronized_windows.npz"
            arrays = dict(force_windows=np.asarray(force_zones, dtype=np.float32),
                          acceleration_windows=np.asarray(acc_zones, dtype=np.float32),
                          window_bounds_s=np.asarray(bounds, dtype=np.float64))
            if use_thermal:
                arrays["temperature_windows_c"] = np.asarray(temp_zones, dtype=np.float32)
            np.savez_compressed(feature_path, **arrays)
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare trial {row.trial_id}: {exc}") from exc
        prepared_rows.append({"trial_id": str(row.trial_id), "material_type": row.material_type,
                              "spindle_speed_rpm": float(row.spindle_speed_rpm),
                              "depth_of_cut_mm": float(row.depth_of_cut_mm), "feature_path": feature_path.name})
    result_path = out / "trial_manifest.csv"
    pd.DataFrame(prepared_rows).to_csv(result_path, index=False)
    return result_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CONFIG.data_root)
    parser.add_argument("--manifest", type=Path, help="Prepared trial manifest; not needed with --prepare-raw-manifest")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--prepare-raw-manifest", type=Path, help="Prepare synchronized windows from timed CSV and calibrated thermal NPZ")
    parser.add_argument("--cache-output", type=Path, default=Path("processed_features"))
    parser.add_argument("--window-ms", type=float)
    parser.add_argument("--acceleration-fs", type=float)
    parser.add_argument("--include-thermal", action="store_true", help="Attempt to read a calibrated thermal_npz column; requires --thermal-fps and --thermal-radiometric")
    parser.add_argument("--thermal-fps", type=float)
    parser.add_argument("--thermal-radiometric", action="store_true")
    args = parser.parse_args()
    if args.prepare_raw_manifest:
        if args.window_ms is None or args.acceleration_fs is None:
            parser.error("raw preparation requires --window-ms and --acceleration-fs")
        if args.include_thermal and args.thermal_fps is None:
            parser.error("--include-thermal requires --thermal-fps")
        CONFIG.signal.window_ms = args.window_ms
        CONFIG.signal.acceleration_sampling_rate_hz = args.acceleration_fs
        CONFIG.vision.include_thermal = args.include_thermal
        if args.include_thermal:
            CONFIG.signal.thermal_frame_rate_hz = args.thermal_fps
            CONFIG.vision.thermal_is_radiometric = args.thermal_radiometric
        output = prepare_raw_trial_cache(args.prepare_raw_manifest, args.cache_output, CONFIG)
        print(f"Prepared manifest: {output}")
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --prepare-raw-manifest is used")
    if args.labels is None:
        parser.error("--labels is required unless --prepare-raw-manifest is used")
    dataset = WAAMGraphDataset(args.root, args.manifest, args.labels)
    print(f"Loaded {len(dataset)} trial records; graphs are built lazily when indexed.")


if __name__ == "__main__":
    main()
