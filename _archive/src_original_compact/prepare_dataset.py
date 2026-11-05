"""Step 1 of the pipeline: raw WAAM exports  ->  per-trial feature caches + zone labels.

For every trial listed in data/trial_manifest.csv this script

  1. reads the Kistler force export (Fx, Fy, Fz) and the accelerometer CSV,
  2. finds the cut inside the 40 s force record from its RMS envelope,
  3. aligns the accelerometer to the force record by cross-correlating the two
     envelopes (the two DAQ PCs had unsynchronised clocks),
  4. splits the cut into 6 zones of equal length (5 mm each) and each zone
     into T overlapping windows,
  5. computes the hand-crafted node features of Section 2.4 for every window,
     and keeps decimated raw windows for the CNN encoders,
  6. labels every zone with the pre-defined chatter rule (signal_features.py),
  7. attaches Ra per zone: measured values from data/ra_measurements.csv when
     present, otherwise the published material mean (flagged as IMPUTED).

Outputs
  data/processed/<trial>.npz          features and raw windows of one trial
  data/processed/dataset_summary.csv  cut times, alignment quality, inclusion
  data/zone_labels.csv                one row per trial x zone, with provenance

Usage:  python src/prepare_dataset.py [--workers 3] [--only trial_04,trial_14]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from config import CONFIG, PipelineConfig, PROJECT_DIR, DATA_DIR, CACHE_DIR
from signal_features import window_features, zone_chatter_label


# --------------------------------------------------------------------------
# Raw file readers
# --------------------------------------------------------------------------
def read_force(path: Path) -> np.ndarray:
    """Kistler DynoWare CSV export -> [samples, 3] array of Fx, Fy, Fz in N.

    Row 0 is 'Cycle No:,1', row 1 holds channel names, row 2 holds units.
    """
    frame = pd.read_csv(path, skiprows=[0, 2], usecols=["Fx", "Fy", "Fz"], dtype=np.float32)
    return frame.to_numpy()


def read_acceleration(path: Path) -> tuple[np.ndarray, float]:
    """Accelerometer CSV (Time, Acc-X, Acc-Y, Acc-Z in g) -> ([samples, 3], fs)."""
    frame = pd.read_csv(path, skiprows=[1], dtype=np.float64)
    times = frame.iloc[:, 0].to_numpy()
    fs = 1.0 / float(np.median(np.diff(times[:10_000])))
    return frame.iloc[:, 1:4].to_numpy(dtype=np.float32), fs


# --------------------------------------------------------------------------
# Cut detection and alignment
# --------------------------------------------------------------------------
def rms_envelope(x: np.ndarray, fs: float, bin_s: float) -> np.ndarray:
    """RMS of the vector magnitude in consecutive bins of `bin_s` seconds."""
    magnitude = np.linalg.norm(x, axis=1)
    n = int(round(fs * bin_s))
    k = len(magnitude) // n
    return np.sqrt((magnitude[: k * n].reshape(k, n) ** 2).mean(axis=1))


def detect_cut(force: np.ndarray, cfg: PipelineConfig) -> tuple[float, float, np.ndarray]:
    """Return (start_s, end_s) of the longest sustained burst of cutting force.

    The first second of every force record is an air cut (spindle on, no
    contact); its median is removed as the dynamometer offset. The onset is the
    first 10 ms bin where the smoothed envelope exceeds
        baseline + threshold_fraction * (plateau - baseline).
    """
    fs, c = cfg.signal.force_fs_hz, cfg.cut
    offset = np.median(force[: int(c.baseline_s * fs)], axis=0)
    env = rms_envelope(force - offset, fs, c.envelope_bin_s)
    smooth = signal.medfilt(env, 51)                       # 0.5 s median filter
    baseline = float(np.median(smooth[: int(c.baseline_s / c.envelope_bin_s)]))
    plateau = float(np.percentile(smooth, 90))
    above = smooth > baseline + c.threshold_fraction * (plateau - baseline)
    # Bridge short dips (tooth-passing fluctuation at 1400 rpm), then keep the longest run.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(int), [0]))))
    runs = edges.reshape(-1, 2)
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if (s - merged[-1][1]) * c.envelope_bin_s < c.max_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = np.asarray(merged)
    longest = merged[np.argmax(merged[:, 1] - merged[:, 0])]
    start, end = longest * c.envelope_bin_s
    if end - start < c.min_cut_s:
        raise ValueError(f"cut burst only {end - start:.1f} s long (force record starts or ends mid-cut)")
    return float(start), float(end), env


def align_acceleration(start: float, end: float, n_force_bins: int, accel: np.ndarray,
                       accel_fs: float, cfg: PipelineConfig) -> tuple[float, float]:
    """Lag (s) such that  accel_time = force_time + lag, and the correlation at that lag.

    The accelerometer envelope (log RMS, 10 ms bins) is matched against a step
    template that is 1 while the force record shows cutting and 0 otherwise.
    Only lags for which the accelerometer record contains the whole cut are
    searched (the operator started each vibration recording before the cut
    and stopped it after the cut).
    """
    c = cfg.cut
    acc_env = rms_envelope(accel - accel.mean(axis=0), accel_fs, c.envelope_bin_s)
    a = np.log(signal.medfilt(acc_env, 21) + 1e-6)
    a = (a - a.mean()) / (a.std() + 1e-9)
    t_bins = np.arange(n_force_bins) * c.envelope_bin_s
    template = ((t_bins >= start) & (t_bins <= end)).astype(float)
    acc_len = len(a) * c.envelope_bin_s
    lo_lag, hi_lag = max(-start, -c.max_align_lag_s), min(acc_len - end, c.max_align_lag_s)
    if hi_lag < lo_lag:
        raise ValueError(f"accelerometer record ({acc_len:.1f} s) is shorter than the cut ({end - start:.1f} s)")
    best_lag, best_r = 0.0, -np.inf
    for lag_bins in range(int(np.ceil(lo_lag / c.envelope_bin_s)), int(np.floor(hi_lag / c.envelope_bin_s)) + 1):
        # accel bin i  <->  force bin i - lag_bins
        idx = np.arange(len(a)) - lag_bins
        valid = (idx >= 0) & (idx < n_force_bins)
        tpl = template[idx[valid]]
        if tpl.std() == 0:
            continue
        r = float(np.corrcoef(a[valid], tpl)[0, 1])
        if r > best_r:
            best_lag, best_r = lag_bins * c.envelope_bin_s, r
    return best_lag, best_r


# --------------------------------------------------------------------------
# Windowing and features
# --------------------------------------------------------------------------
def cut_windows(x: np.ndarray, fs: float, starts_s: np.ndarray, width_s: float) -> np.ndarray:
    """Slice [samples, 3] into [len(starts), 3, width] windows."""
    n = int(round(width_s * fs))
    idx = np.round(starts_s * fs).astype(int)
    if idx.min() < 0 or idx.max() + n > len(x):
        raise ValueError("record does not cover the requested window")
    return np.stack([x[i:i + n].T for i in idx]).astype(np.float32)


def wall_clock_lag(row: dict) -> float:
    """Lag implied by the two PCs' recorded start times (accel_time = force_time + lag)."""
    to_s = lambda hms: sum(int(p) * m for p, m in zip(str(hms).split(":"), (3600, 60, 1)))
    return float(to_s(row["force_start_time"]) - to_s(row["acceleration_start_time"]))


def locate_trial(row: dict, cfg: PipelineConfig = CONFIG) -> dict:
    """Pass 1: find the cut and the envelope-correlation lag of one trial."""
    out = {"trial_id": row["trial_id"]}
    try:
        force = read_force(PROJECT_DIR / row["force_csv"])
        accel, accel_fs = read_acceleration(PROJECT_DIR / row["acceleration_csv"])
        start, end, force_env = detect_cut(force, cfg)
        lag, corr = align_acceleration(start, end, len(force_env), accel, accel_fs, cfg)
        out.update(cut_start_s=start, cut_end_s=end, xcorr_lag_s=lag, xcorr_r=corr,
                   wall_clock_lag_s=wall_clock_lag(row), accel_fs=accel_fs,
                   accel_record_s=len(accel) / accel_fs, error="")
    except Exception as exc:
        out["error"] = str(exc)
    return out


def choose_lags(located: list[dict], cfg: PipelineConfig) -> dict:
    """Final accelerometer lag of every trial, with a note on how reliable it is."""
    decisions = {}
    for l in located:
        if l["error"]:
            decisions[l["trial_id"]] = (None, l["error"])
        elif l["xcorr_r"] >= cfg.cut.min_alignment_r:
            decisions[l["trial_id"]] = (l["xcorr_lag_s"], "step-template match of accelerometer envelope")
        else:
            decisions[l["trial_id"]] = (l["xcorr_lag_s"], "WEAK: accelerometer barely responds to the cut; "
                                                          "best lag inside the feasible range used")
    return decisions


def prepare_trial(row: dict, located: dict, lag: float | None, lag_source: str,
                  cfg: PipelineConfig = CONFIG) -> dict:
    """Pass 2: cut windows, compute features and labels, write the cache file."""
    summary = {"trial_id": row["trial_id"], "material_type": row["material_type"],
               "spindle_speed_rpm": row["spindle_speed_rpm"], "depth_of_cut_mm": row["depth_of_cut_mm"],
               "force_pairing_confidence": row.get("force_pairing_confidence", "")}
    try:
        if lag is None:
            raise ValueError(lag_source)
        force = read_force(PROJECT_DIR / row["force_csv"])
        accel, accel_fs = read_acceleration(PROJECT_DIR / row["acceleration_csv"])
        fs = cfg.signal.force_fs_hz
        if abs(accel_fs - cfg.signal.acceleration_fs_hz) > 1:
            raise ValueError(f"accelerometer sampled at {accel_fs:.0f} Hz, expected {cfg.signal.acceleration_fs_hz:.0f}")
        start, end = located["cut_start_s"], located["cut_end_s"]
        summary.update(cut_start_s=round(start, 3), cut_end_s=round(end, 3), cut_duration_s=round(end - start, 3),
                       accel_lag_s=round(lag, 3), accel_lag_source=lag_source,
                       accel_alignment_r=round(located["xcorr_r"], 3),
                       accel_record_s=round(len(accel) / accel_fs, 2))
        if start + lag < 0 or end + lag > len(accel) / accel_fs:
            raise ValueError(f"accelerometer record ({len(accel) / accel_fs:.1f} s) does not cover the cut "
                             f"({start + lag:.1f}-{end + lag:.1f} s in accelerometer time)")

        # Zones: equal division of the detected cut; T windows spread over each zone.
        width = cfg.signal.window_ms / 1000.0
        n_zones, n_win = cfg.process.zone_count, cfg.windows_per_zone
        zone_len = (end - start) / n_zones
        offsets = np.linspace(0.0, zone_len - width, n_win)
        rpm = float(row["spindle_speed_rpm"])
        decim = cfg.signal.encoder_decimation
        feats, f_raw, a_raw, labels = [], [], [], []
        for z in range(n_zones):
            z0 = start + z * zone_len
            f_win = cut_windows(force, fs, z0 + offsets, width)
            a_win = cut_windows(accel, accel_fs, z0 + offsets + lag, width)
            feats.append([np.concatenate([window_features(fw, fs, rpm, cfg.signal),
                                          window_features(aw, accel_fs, rpm, cfg.signal)])
                          for fw, aw in zip(f_win, a_win)])
            # Anti-aliased decimation for the CNN encoders (25 kHz -> 12.5 kHz).
            f_raw.append(signal.resample_poly(f_win, 1, decim, axis=-1))
            a_raw.append(signal.resample_poly(a_win, 1, decim, axis=-1))
            zone_force = force[int(z0 * fs): int((z0 + zone_len) * fs)].T
            labels.append({"zone_id": z, **zone_chatter_label(zone_force, fs, rpm, cfg.signal, cfg.labels)})

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_DIR / f"{row['trial_id']}.npz",
                            features=np.asarray(feats, dtype=np.float32),          # [6, T, 60]
                            force_raw=np.asarray(f_raw, dtype=np.float32),         # [6, T, 3, L]
                            acceleration_raw=np.asarray(a_raw, dtype=np.float32),  # [6, T, 3, L]
                            zone_start_s=start + np.arange(n_zones) * zone_len)
        summary.update(included=True, exclusion_reason="", windows_per_zone=n_win, zone_labels=labels)
    except Exception as exc:  # keep going; the reason is written to the summary
        summary.update(included=False, exclusion_reason=str(exc), zone_labels=[])
    return summary


# --------------------------------------------------------------------------
# Labels table
# --------------------------------------------------------------------------
def build_label_table(summaries: list[dict], cfg: PipelineConfig) -> pd.DataFrame:
    """Combine chatter labels with measured or imputed Ra, one row per zone."""
    measured_path = DATA_DIR / "ra_measurements.csv"
    measured = pd.read_csv(measured_path) if measured_path.exists() else pd.DataFrame(columns=["trial_id", "zone_id", "Ra_um"])
    measured = measured.dropna(subset=["Ra_um"])
    lookup = {(r.trial_id, int(r.zone_id)): float(r.Ra_um) for r in measured.itertuples()}
    rows = []
    for s in summaries:
        for lab in s["zone_labels"]:
            key = (s["trial_id"], lab["zone_id"])
            if key in lookup:
                ra, source = lookup[key], "measured (data/ra_measurements.csv)"
            else:
                ra = cfg.labels.ra_material_mean_um[s["material_type"]]
                source = "IMPUTED: material mean from ref. [1], not a zone measurement"
            rows.append({"trial_id": s["trial_id"], "zone_id": lab["zone_id"], "material_type": s["material_type"],
                         "chatter_label": lab["chatter_label"], "Ra_um": ra, "Ra_source": source,
                         "label_peak_hz": round(lab["peak_hz"], 1),
                         "label_peak_prominence_db": round(lab["peak_prominence_db"], 1),
                         "label_peak_is_spindle_harmonic": lab["peak_is_spindle_harmonic"],
                         "label_ci_zone": round(lab["ci_zone"], 3)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3, help="parallel trials (each needs ~1 GB RAM)")
    parser.add_argument("--only", type=str, default="", help="comma-separated trial ids to process")
    args = parser.parse_args()

    manifest = pd.read_csv(DATA_DIR / "trial_manifest.csv")
    if args.only:
        manifest = manifest[manifest.trial_id.isin(args.only.split(","))]
    rows = manifest.to_dict("records")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        print(f"Pass 1/2: locating the cut and aligning sensors in {len(rows)} trials...")
        located = list(pool.map(locate_trial, rows))
        decisions = choose_lags(located, CONFIG)
        print(f"Pass 2/2: features and labels ({CONFIG.windows_per_zone} windows/zone, "
              f"{CONFIG.signal.window_ms:g} ms windows)...")
        jobs = [(row, loc, *decisions[row["trial_id"]]) for row, loc in zip(rows, located)]
        summaries = []
        for s in pool.map(prepare_trial, *zip(*jobs)):
            status = "OK " if s["included"] else "SKIP"
            extra = (f"cut {s['cut_start_s']:.2f}-{s['cut_end_s']:.2f} s, accel lag {s['accel_lag_s']:+.2f} s "
                     f"(r={s['accel_alignment_r']:.2f}), unstable zones "
                     f"{sum(l['chatter_label'] for l in s['zone_labels'])}/6") if s["included"] else s["exclusion_reason"]
            print(f"  [{status}] {s['trial_id']}: {extra}")
            summaries.append(s)

    summary_frame = pd.DataFrame([{k: v for k, v in s.items() if k != "zone_labels"} for s in summaries])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(CACHE_DIR / "dataset_summary.csv", index=False)
    labels = build_label_table([s for s in summaries if s["included"]], CONFIG)
    if labels.empty:
        raise SystemExit("No trial could be prepared - see the SKIP reasons above.")
    if args.only and (DATA_DIR / "zone_labels.csv").exists():
        # Partial rerun: replace only the processed trials in the existing table.
        old = pd.read_csv(DATA_DIR / "zone_labels.csv")
        labels = pd.concat([old[~old.trial_id.isin(labels.trial_id)], labels]).sort_values(["trial_id", "zone_id"])
    labels.to_csv(DATA_DIR / "zone_labels.csv", index=False)
    print(f"\nWrote {CACHE_DIR / 'dataset_summary.csv'} and {DATA_DIR / 'zone_labels.csv'}")
    print(f"Unstable zones: {int(labels.chatter_label.sum())} / {len(labels)}")


if __name__ == "__main__":
    main()
