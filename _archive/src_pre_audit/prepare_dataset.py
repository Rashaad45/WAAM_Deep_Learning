"""
prepare_dataset.py   -   STEP 1 of the pipeline
================================================
Raw WAAM lab files  ->  one feature file per trial  +  a table of zone labels.

For every trial listed in data/trial_manifest.csv this script

  1. reads the Kistler force export (Fx, Fy, Fz) and the accelerometer CSV,
  2. finds the cut inside the 40 s force record from its RMS envelope,
  3. aligns the accelerometer to the force record (the two lab PCs had
     unsynchronised clocks) by matching the two envelopes,
  4. splits the cut into 6 zones of equal length (5 mm each) and every zone
     into T overlapping windows,
  5. computes the 10 hand-crafted features per channel for every window and
     keeps a down-sampled copy of the raw window for the CNN encoders,
  6. labels every zone with the chatter rule (signal_features.py),
  7. attaches Ra per zone: measured values from data/ra_measurements.csv when
     present, otherwise the published material mean (flagged as IMPUTED).

Files written
  data/processed/<trial>.npz          features and raw windows of one trial
  data/processed/dataset_summary.csv  cut times, alignment quality, included or not
  data/zone_labels.csv                one row per trial x zone

How to run (from the project folder)
  python run_pipeline.py prepare
or directly
  python src/prepare_dataset.py                       all trials, 3 in parallel
  python src/prepare_dataset.py --workers 1           one trial at a time (less RAM)
  python src/prepare_dataset.py --only trial_04,trial_14
"""
import argparse                                          # reads options typed after the file name
from concurrent.futures import ProcessPoolExecutor       # runs several trials in parallel
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from config import CONFIG, PipelineConfig, PROJECT_DIR, DATA_DIR, CACHE_DIR
from signal_features import WindowFeatureExtractor, ZoneChatterLabeler


# ===========================================================================
# 1. Reading the raw files
# ===========================================================================
class RawFileReader:
    """Reads the two kinds of CSV files exported in the lab."""

    @staticmethod
    def read_force(path: Path) -> np.ndarray:
        """Kistler DynoWare CSV -> array [samples, 3] with Fx, Fy, Fz in newton.

        File layout: row 0 = 'Cycle No:,1', row 1 = column names, row 2 = units.
        We skip rows 0 and 2 so that pandas uses row 1 as the header.
        """
        frame = pd.read_csv(path, skiprows=[0, 2], usecols=["Fx", "Fy", "Fz"], dtype=np.float32)
        return frame.to_numpy()

    @staticmethod
    def read_acceleration(path: Path) -> tuple[np.ndarray, float]:
        """Accelerometer CSV (Time, Acc-X, Acc-Y, Acc-Z in g) -> (array [samples, 3], sampling rate)."""
        frame = pd.read_csv(path, skiprows=[1], dtype=np.float64)    # row 1 holds the units
        times = frame.iloc[:, 0].to_numpy()                          # first column = time in s
        # Sampling rate = 1 / (typical time step between two samples).
        sampling_rate = 1.0 / float(np.median(np.diff(times[:10_000])))
        acceleration = frame.iloc[:, 1:4].to_numpy(dtype=np.float32)  # columns 1, 2, 3 = X, Y, Z
        return acceleration, sampling_rate


# ===========================================================================
# 2. Finding where the cut is inside the force record
# ===========================================================================
class CutDetector:
    """Locates the cutting period inside a 40 s force recording."""

    def __init__(self, config: PipelineConfig):
        self.fs = config.signal.force_fs_hz      # force sampling rate (25 kHz)
        self.settings = config.cut               # CutDetectionConfig from config.py

    @staticmethod
    def rms_envelope(x: np.ndarray, fs: float, bin_s: float) -> np.ndarray:
        """RMS of the 3-axis vector length in consecutive bins of `bin_s` seconds."""
        magnitude = np.linalg.norm(x, axis=1)              # sqrt(x^2 + y^2 + z^2) for every sample
        samples_per_bin = int(round(fs * bin_s))           # e.g. 25000 * 0.01 = 250
        bin_count = len(magnitude) // samples_per_bin      # number of complete bins
        # Cut off the incomplete last bin, reshape to [bins, samples_per_bin]
        # and take the RMS of every row.
        usable = magnitude[: bin_count * samples_per_bin].reshape(bin_count, samples_per_bin)
        return np.sqrt((usable ** 2).mean(axis=1))

    @staticmethod
    def merge_short_gaps(runs: np.ndarray, bin_s: float, max_gap_s: float) -> np.ndarray:
        """Join consecutive [start, end] runs separated by less than `max_gap_s`.

        Short dips are caused by tooth-passing fluctuation, not by the end of the cut.
        """
        merged = [list(runs[0])]
        for start, end in runs[1:]:
            gap_s = (start - merged[-1][1]) * bin_s
            if gap_s < max_gap_s:
                merged[-1][1] = end          # extend the previous run
            else:
                merged.append([start, end])  # start a new run
        return np.asarray(merged)

    def detect(self, force: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return (start_s, end_s, envelope) of the longest sustained burst of cutting force.

        The first second of each record is an air cut (spindle on, no contact); its
        median is the dynamometer offset. The cut starts in the first 10 ms bin where
        the smoothed envelope exceeds  baseline + 30 % * (plateau - baseline).
        """
        s = self.settings

        # --- remove the sensor offset and build the 10 ms RMS envelope -------
        offset = np.median(force[: int(s.baseline_s * self.fs)], axis=0)
        envelope = self.rms_envelope(force - offset, self.fs, s.envelope_bin_s)
        smooth = signal.medfilt(envelope, 51)                  # 51 bins = 0.5 s median filter

        # --- threshold between the "air" level and the "cutting" level -------
        baseline = float(np.median(smooth[: int(s.baseline_s / s.envelope_bin_s)]))
        plateau = float(np.percentile(smooth, 90))
        above = smooth > baseline + s.threshold_fraction * (plateau - baseline)

        # --- find runs of consecutive True values ------------------------------
        # Padding with 0 on both sides and taking the difference gives +1 where a
        # run starts and -1 where it ends; flatnonzero returns those positions.
        padded = np.concatenate(([0], above.astype(int), [0]))
        edges = np.flatnonzero(np.diff(padded))
        runs = edges.reshape(-1, 2)                            # rows of [start_bin, end_bin]

        # --- bridge short dips, then keep the longest run ---------------------
        merged = self.merge_short_gaps(runs, s.envelope_bin_s, s.max_gap_s)
        lengths = merged[:, 1] - merged[:, 0]
        longest = merged[np.argmax(lengths)]
        start, end = longest * s.envelope_bin_s                # bins -> seconds

        if end - start < s.min_cut_s:
            raise ValueError(f"cut burst only {end - start:.1f} s long (force record starts or ends mid-cut)")
        return float(start), float(end), envelope


# ===========================================================================
# 3. Aligning the accelerometer to the force record
# ===========================================================================
class SensorAligner:
    """Finds the time shift between the accelerometer clock and the force clock."""

    def __init__(self, config: PipelineConfig):
        self.settings = config.cut

    def find_lag(self, cut_start: float, cut_end: float, n_force_bins: int,
                 acceleration: np.ndarray, accel_fs: float) -> tuple[float, float]:
        """Return (lag in s, correlation) such that  accel_time = force_time + lag.

        The accelerometer envelope (log RMS, 10 ms bins) is compared with a "step"
        template that is 1 while the force record shows cutting and 0 otherwise.
        Only lags for which the accelerometer record contains the whole cut are
        tried (the operator started each vibration recording before the cut and
        stopped it after the cut).
        """
        s = self.settings
        bin_s = s.envelope_bin_s

        # --- normalised log envelope of the acceleration ----------------------
        accel_env = CutDetector.rms_envelope(acceleration - acceleration.mean(axis=0), accel_fs, bin_s)
        a = np.log(signal.medfilt(accel_env, 21) + 1e-6)
        a = (a - a.mean()) / (a.std() + 1e-9)                  # zero mean, unit spread

        # --- step template on the force time axis ------------------------------
        force_times = np.arange(n_force_bins) * bin_s
        template = ((force_times >= cut_start) & (force_times <= cut_end)).astype(float)

        # --- range of lags that keeps the whole cut inside the accel record -----
        accel_length_s = len(a) * bin_s
        lowest_lag = max(-cut_start, -s.max_align_lag_s)
        highest_lag = min(accel_length_s - cut_end, s.max_align_lag_s)
        if highest_lag < lowest_lag:
            raise ValueError(f"accelerometer record ({accel_length_s:.1f} s) is shorter than "
                             f"the cut ({cut_end - cut_start:.1f} s)")

        # --- try every lag (in whole bins) and keep the best correlation --------
        best_lag, best_r = 0.0, -np.inf
        first_bin = int(np.ceil(lowest_lag / bin_s))
        last_bin = int(np.floor(highest_lag / bin_s))
        for lag_bins in range(first_bin, last_bin + 1):
            # Accelerometer bin i corresponds to force bin (i - lag_bins).
            force_index = np.arange(len(a)) - lag_bins
            valid = (force_index >= 0) & (force_index < n_force_bins)
            shifted_template = template[force_index[valid]]
            if shifted_template.std() == 0:                    # all 0 or all 1: correlation undefined
                continue
            r = float(np.corrcoef(a[valid], shifted_template)[0, 1])
            if r > best_r:
                best_lag, best_r = lag_bins * bin_s, r
        return best_lag, best_r


# ===========================================================================
# 4. Processing one trial
# ===========================================================================
def clock_to_seconds(hms) -> int:
    """'14:03:27' -> 14*3600 + 3*60 + 27 seconds."""
    parts = str(hms).split(":")
    multipliers = (3600, 60, 1)
    return sum(int(part) * mult for part, mult in zip(parts, multipliers))


def cut_windows(x: np.ndarray, fs: float, starts_s: np.ndarray, width_s: float) -> np.ndarray:
    """Slice a [samples, 3] record into windows -> array [len(starts), 3, width_samples]."""
    width = int(round(width_s * fs))                           # window length in samples
    start_indices = np.round(starts_s * fs).astype(int)
    if start_indices.min() < 0 or start_indices.max() + width > len(x):
        raise ValueError("record does not cover the requested window")
    # .T turns [width, 3] into [3, width] so channels come first.
    windows = [x[i:i + width].T for i in start_indices]
    return np.stack(windows).astype(np.float32)


class TrialProcessor:
    """Runs the two passes over one trial: (1) locate the cut, (2) features + labels.

    One object of this class is sent to each worker process, so it only stores
    settings and helper objects (no big arrays).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.reader = RawFileReader()
        self.cut_detector = CutDetector(config)
        self.aligner = SensorAligner(config)
        self.feature_extractor = WindowFeatureExtractor(config.signal)
        self.labeler = ZoneChatterLabeler(config.signal, config.labels)

    # -----------------------------------------------------------------------
    # Pass 1
    # -----------------------------------------------------------------------
    def locate(self, row: dict) -> dict:
        """Find the cut and the accelerometer lag of one trial (row of the manifest)."""
        result = {"trial_id": row["trial_id"]}
        try:
            force = self.reader.read_force(PROJECT_DIR / row["force_csv"])
            accel, accel_fs = self.reader.read_acceleration(PROJECT_DIR / row["acceleration_csv"])
            start, end, force_env = self.cut_detector.detect(force)
            lag, corr = self.aligner.find_lag(start, end, len(force_env), accel, accel_fs)
            # Lag implied by the clocks of the two PCs (stored only for comparison).
            wall_clock_lag = float(clock_to_seconds(row["force_start_time"])
                                   - clock_to_seconds(row["acceleration_start_time"]))
            result.update(cut_start_s=start, cut_end_s=end, xcorr_lag_s=lag, xcorr_r=corr,
                          wall_clock_lag_s=wall_clock_lag, accel_fs=accel_fs,
                          accel_record_s=len(accel) / accel_fs, error="")
        except Exception as exc:              # any problem -> remember the message, keep going
            result["error"] = str(exc)
        return result

    def choose_lag(self, located: dict) -> tuple[float | None, str]:
        """Final accelerometer lag of a trial, with a note on how reliable it is."""
        if located["error"]:
            return None, located["error"]
        if located["xcorr_r"] >= self.config.cut.min_alignment_r:
            return located["xcorr_lag_s"], "step-template match of accelerometer envelope"
        return located["xcorr_lag_s"], ("WEAK: accelerometer barely responds to the cut; "
                                        "best lag inside the feasible range used")

    # -----------------------------------------------------------------------
    # Pass 2
    # -----------------------------------------------------------------------
    def prepare(self, row: dict, located: dict, lag: float | None, lag_source: str) -> dict:
        """Cut windows, compute features and labels, and write data/processed/<trial>.npz."""
        cfg = self.config
        summary = {"trial_id": row["trial_id"], "material_type": row["material_type"],
                   "spindle_speed_rpm": row["spindle_speed_rpm"], "depth_of_cut_mm": row["depth_of_cut_mm"],
                   "force_pairing_confidence": row.get("force_pairing_confidence", "")}
        try:
            if lag is None:                                    # pass 1 failed for this trial
                raise ValueError(lag_source)

            # --- read the raw data again --------------------------------------
            force = self.reader.read_force(PROJECT_DIR / row["force_csv"])
            accel, accel_fs = self.reader.read_acceleration(PROJECT_DIR / row["acceleration_csv"])
            fs = cfg.signal.force_fs_hz
            if abs(accel_fs - cfg.signal.acceleration_fs_hz) > 1:
                raise ValueError(f"accelerometer sampled at {accel_fs:.0f} Hz, "
                                 f"expected {cfg.signal.acceleration_fs_hz:.0f}")

            start, end = located["cut_start_s"], located["cut_end_s"]
            accel_length_s = len(accel) / accel_fs
            summary.update(cut_start_s=round(start, 3), cut_end_s=round(end, 3),
                           cut_duration_s=round(end - start, 3),
                           accel_lag_s=round(lag, 3), accel_lag_source=lag_source,
                           accel_alignment_r=round(located["xcorr_r"], 3),
                           accel_record_s=round(accel_length_s, 2))
            if start + lag < 0 or end + lag > accel_length_s:
                raise ValueError(f"accelerometer record ({accel_length_s:.1f} s) does not cover the cut "
                                 f"({start + lag:.1f}-{end + lag:.1f} s in accelerometer time)")

            # --- zone and window layout ----------------------------------------
            # The detected cut is divided into 6 equal zones; inside each zone
            # T windows are spread evenly from its start to its end.
            window_s = cfg.signal.window_ms / 1000.0
            n_zones = cfg.process.zone_count
            n_windows = cfg.windows_per_zone
            zone_length_s = (end - start) / n_zones
            window_offsets = np.linspace(0.0, zone_length_s - window_s, n_windows)
            rpm = float(row["spindle_speed_rpm"])
            decimation = cfg.signal.encoder_decimation

            features, force_raw, accel_raw, zone_labels = [], [], [], []
            for zone in range(n_zones):
                zone_start = start + zone * zone_length_s

                # Windows of force and acceleration (accel shifted by the lag).
                force_windows = cut_windows(force, fs, zone_start + window_offsets, window_s)
                accel_windows = cut_windows(accel, accel_fs, zone_start + window_offsets + lag, window_s)

                # 60 features per window: 30 from force (3 axes x 10) + 30 from acceleration.
                zone_features = []
                for fw, aw in zip(force_windows, accel_windows):
                    f_feat = self.feature_extractor.window_features(fw, fs, rpm)
                    a_feat = self.feature_extractor.window_features(aw, accel_fs, rpm)
                    zone_features.append(np.concatenate([f_feat, a_feat]))
                features.append(zone_features)

                # Down-sampled raw windows for the CNN encoders (25 kHz -> 12.5 kHz,
                # resample_poly applies an anti-aliasing filter first).
                force_raw.append(signal.resample_poly(force_windows, 1, decimation, axis=-1))
                accel_raw.append(signal.resample_poly(accel_windows, 1, decimation, axis=-1))

                # Chatter label from the FULL force record of this zone.
                zone_force = force[int(zone_start * fs): int((zone_start + zone_length_s) * fs)].T
                label = self.labeler.label_zone(zone_force, fs, rpm)
                zone_labels.append({"zone_id": zone, **label})

            # --- save everything of this trial in one compressed file -----------
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE_DIR / f"{row['trial_id']}.npz",
                                features=np.asarray(features, dtype=np.float32),          # [6, T, 60]
                                force_raw=np.asarray(force_raw, dtype=np.float32),        # [6, T, 3, L]
                                acceleration_raw=np.asarray(accel_raw, dtype=np.float32), # [6, T, 3, L]
                                zone_start_s=start + np.arange(n_zones) * zone_length_s)
            summary.update(included=True, exclusion_reason="", windows_per_zone=n_windows,
                           zone_labels=zone_labels)
        except Exception as exc:              # keep going; the reason goes into the summary
            summary.update(included=False, exclusion_reason=str(exc), zone_labels=[])
        return summary


# ===========================================================================
# 5. The table of zone labels
# ===========================================================================
class LabelTableBuilder:
    """Combines chatter labels with measured (or imputed) Ra -> one row per zone."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_measured_ra(self) -> dict:
        """Read data/ra_measurements.csv -> {(trial_id, zone_id): Ra}. Empty if no values."""
        path = DATA_DIR / "ra_measurements.csv"
        if path.exists():
            measured = pd.read_csv(path)
        else:
            measured = pd.DataFrame(columns=["trial_id", "zone_id", "Ra_um"])
        measured = measured.dropna(subset=["Ra_um"])          # ignore empty cells
        return {(r.trial_id, int(r.zone_id)): float(r.Ra_um) for r in measured.itertuples()}

    def build(self, summaries: list[dict]) -> pd.DataFrame:
        measured = self.load_measured_ra()
        rows = []
        for trial in summaries:
            for lab in trial["zone_labels"]:
                key = (trial["trial_id"], lab["zone_id"])
                if key in measured:
                    ra = measured[key]
                    source = "measured (data/ra_measurements.csv)"
                else:
                    ra = self.config.labels.ra_material_mean_um[trial["material_type"]]
                    source = "IMPUTED: material mean from ref. [1], not a zone measurement"
                rows.append({"trial_id": trial["trial_id"], "zone_id": lab["zone_id"],
                             "material_type": trial["material_type"],
                             "chatter_label": lab["chatter_label"], "Ra_um": ra, "Ra_source": source,
                             "label_peak_hz": round(lab["peak_hz"], 1),
                             "label_peak_prominence_db": round(lab["peak_prominence_db"], 1),
                             "label_peak_is_spindle_harmonic": lab["peak_is_spindle_harmonic"],
                             "label_ci_zone": round(lab["ci_zone"], 3)})
        return pd.DataFrame(rows)


# ===========================================================================
# 6. The whole of Step 1
# ===========================================================================
class DatasetPreparer:
    """Runs pass 1 and pass 2 over all trials and writes the summary and label tables."""

    def __init__(self, config: PipelineConfig, workers: int, only: str):
        self.config = config
        self.workers = workers                  # how many trials are processed at the same time
        self.only = only                        # e.g. "trial_04,trial_14" or "" for all
        self.processor = TrialProcessor(config)
        self.label_builder = LabelTableBuilder(config)

    def load_manifest(self) -> list[dict]:
        """Rows of data/trial_manifest.csv as a list of dicts (one dict per trial)."""
        manifest = pd.read_csv(DATA_DIR / "trial_manifest.csv")
        if self.only:
            manifest = manifest[manifest.trial_id.isin(self.only.split(","))]
        return manifest.to_dict("records")

    @staticmethod
    def describe(summary: dict) -> str:
        """One progress line for the terminal."""
        if not summary["included"]:
            return f"  [SKIP] {summary['trial_id']}: {summary['exclusion_reason']}"
        unstable = sum(label["chatter_label"] for label in summary["zone_labels"])
        return (f"  [OK ] {summary['trial_id']}: cut {summary['cut_start_s']:.2f}-{summary['cut_end_s']:.2f} s, "
                f"accel lag {summary['accel_lag_s']:+.2f} s (r={summary['accel_alignment_r']:.2f}), "
                f"unstable zones {unstable}/6")

    def run(self) -> None:
        rows = self.load_manifest()

        # A process pool runs the trials in parallel (each worker needs ~1 GB RAM).
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            print(f"Pass 1/2: locating the cut and aligning sensors in {len(rows)} trials...")
            located = list(pool.map(self.processor.locate, rows))
            lags = [self.processor.choose_lag(loc) for loc in located]   # list of (lag, source)

            print(f"Pass 2/2: features and labels ({self.config.windows_per_zone} windows/zone, "
                  f"{self.config.signal.window_ms:g} ms windows)...")
            lag_values = [lag for lag, _ in lags]
            lag_sources = [source for _, source in lags]
            summaries = []
            # pool.map with several lists calls prepare(rows[i], located[i], lag_values[i], lag_sources[i]).
            for summary in pool.map(self.processor.prepare, rows, located, lag_values, lag_sources):
                print(self.describe(summary))
                summaries.append(summary)

        self.save(summaries)

    def save(self, summaries: list[dict]) -> None:
        # dataset_summary.csv: everything except the nested list of zone labels.
        summary_rows = [{k: v for k, v in s.items() if k != "zone_labels"} for s in summaries]
        summary_table = pd.DataFrame(summary_rows)
        summary_path = CACHE_DIR / "dataset_summary.csv"
        if self.only and summary_path.exists():
            # Partial rerun: keep the rows of the trials that were NOT processed this time,
            # otherwise the next training run would only see the re-processed trials.
            old = pd.read_csv(summary_path)
            kept_old_rows = old[~old.trial_id.isin(summary_table.trial_id)]
            summary_table = pd.concat([kept_old_rows, summary_table]).sort_values("trial_id")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        summary_table.to_csv(summary_path, index=False)

        # zone_labels.csv: only the trials that could be prepared.
        labels = self.label_builder.build([s for s in summaries if s["included"]])
        if labels.empty:
            raise SystemExit("No trial could be prepared - see the SKIP reasons above.")

        label_path = DATA_DIR / "zone_labels.csv"
        if self.only and label_path.exists():
            # Partial rerun: replace only the processed trials in the existing table.
            old = pd.read_csv(label_path)
            kept_old_rows = old[~old.trial_id.isin(labels.trial_id)]
            labels = pd.concat([kept_old_rows, labels]).sort_values(["trial_id", "zone_id"])
        labels.to_csv(label_path, index=False)

        print(f"\nWrote {CACHE_DIR / 'dataset_summary.csv'} and {label_path}")
        print(f"Unstable zones: {int(labels.chatter_label.sum())} / {len(labels)}")


def main() -> None:
    # Options that can be typed after the file name.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3, help="parallel trials (each needs ~1 GB RAM)")
    parser.add_argument("--only", type=str, default="", help="comma-separated trial ids to process")
    args = parser.parse_args()

    preparer = DatasetPreparer(CONFIG, workers=args.workers, only=args.only)
    preparer.run()


# This block runs only when the file is started directly (python prepare_dataset.py),
# not when another file imports it. On Windows it is REQUIRED for the process pool.
if __name__ == "__main__":
    main()
