"""
prepare_dataset.py - STEP 1: raw force + acceleration CSVs -> zone features and zone labels.

Per trial:
  1. read force (Kistler CSV) and acceleration (KiNOVA CSV);
  2. detect the cut in the force envelope (unchanged from the pre-audit pipeline);
  3. align the accelerometer to the force record by envelope matching; r < 0.7 -> accel_ok = False;
  4. MEASURE the spindle-rotation frequency from the force spectrum (harmonic comb, ~0.001 Hz precision);
     the harmonic test at ~3.5 kHz (harmonic number ~150) needs this: a +/-2 Hz tolerance there
     corresponds to a 0.06 % speed accuracy that the nominal rpm cannot provide;
  5. split the detected cut into 3 equal zones and drop 5 mm (one tool radius) at each end;
  6. T windows of 100 ms evenly spread over the retained span of each zone; features per window
     (chatter indicator on a 0.5 s context); decimated raw windows for the CNN encoders;
  7. label every zone at 15 / 20 / 25 dB from ONE high-resolution PSD of its retained span.

Writes data/processed/<trial>.npz, data/processed/dataset_summary.csv, data/zone_labels.csv.
No Ra target is produced (ZONE_RA = no) and no thermal data is read (IR_EXPORT = no).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import signal

from config import CONFIG, PipelineConfig, PROJECT_DIR, DATA_DIR, CACHE_DIR
from signal_features import WindowFeatureExtractor, zone_spectrum_peak, chatter_label, harmonic_mask


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------
def read_force(path) -> np.ndarray:
    """Kistler DynoWare CSV (row0 'Cycle No', row1 names, row2 units) -> [samples, 3] Fx, Fy, Fz (N)."""
    return pd.read_csv(path, skiprows=[0, 2], usecols=["Fx", "Fy", "Fz"], dtype=np.float32).to_numpy()


def read_acceleration(path) -> tuple[np.ndarray, float]:
    """Accelerometer CSV (Time, X, Y, Z in g; row 1 = units) -> ([samples, 3], fs)."""
    frame = pd.read_csv(path, skiprows=[1], dtype=np.float64)
    t = frame.iloc[:, 0].to_numpy()
    return frame.iloc[:, 1:4].to_numpy(dtype=np.float32), 1.0 / float(np.median(np.diff(t[:10_000])))


# ---------------------------------------------------------------------------
# Cut detection and sensor alignment (unchanged logic, see AUDIT.md)
# ---------------------------------------------------------------------------
def rms_envelope(x: np.ndarray, fs: float, bin_s: float) -> np.ndarray:
    mag = np.linalg.norm(x, axis=1)
    n = int(round(fs * bin_s))
    k = len(mag) // n
    return np.sqrt((mag[: k * n].reshape(k, n) ** 2).mean(axis=1))


def detect_cut(force: np.ndarray, cfg: PipelineConfig) -> tuple[float, float, np.ndarray]:
    s, fs = cfg.cut, cfg.signal.force_fs_hz
    offset = np.median(force[: int(s.baseline_s * fs)], axis=0)
    env = rms_envelope(force - offset, fs, s.envelope_bin_s)
    smooth = signal.medfilt(env, 51)
    base = float(np.median(smooth[: int(s.baseline_s / s.envelope_bin_s)]))
    plateau = float(np.percentile(smooth, 90))
    above = smooth > base + s.threshold_fraction * (plateau - base)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(int), [0])))).reshape(-1, 2)
    merged = [list(edges[0])]
    for a, b in edges[1:]:
        if (a - merged[-1][1]) * s.envelope_bin_s < s.max_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = np.asarray(merged)
    a, b = merged[np.argmax(merged[:, 1] - merged[:, 0])] * s.envelope_bin_s
    if b - a < s.min_cut_s:
        raise ValueError(f"cut burst only {b - a:.1f} s long (force record starts or ends mid-cut)")
    return float(a), float(b), env


def align_acceleration(cut_start, cut_end, n_force_bins, accel, accel_fs, cfg) -> tuple[float, float]:
    s = cfg.cut
    b = s.envelope_bin_s
    env = rms_envelope(accel - accel.mean(axis=0), accel_fs, b)
    a = np.log(signal.medfilt(env, 21) + 1e-6)
    a = (a - a.mean()) / (a.std() + 1e-9)
    ft = np.arange(n_force_bins) * b
    template = ((ft >= cut_start) & (ft <= cut_end)).astype(float)
    lo = max(-cut_start, -s.max_align_lag_s)
    hi = min(len(a) * b - cut_end, s.max_align_lag_s)
    if hi < lo:
        raise ValueError("accelerometer record shorter than the cut")
    best = (0.0, -np.inf)
    for lag in range(int(np.ceil(lo / b)), int(np.floor(hi / b)) + 1):
        idx = np.arange(len(a)) - lag
        ok = (idx >= 0) & (idx < n_force_bins)
        tpl = template[idx[ok]]
        if tpl.std() == 0:
            continue
        r = float(np.corrcoef(a[ok], tpl)[0, 1])
        if r > best[1]:
            best = (lag * b, r)
    return best


# ---------------------------------------------------------------------------
# Spindle frequency from the force spectrum
# ---------------------------------------------------------------------------
def measure_spindle_hz(force_cut: np.ndarray, fs: float, search_hz: tuple = (12.0, 30.0),
                       n_harmonics: int = 60) -> tuple[float, float]:
    """Harmonic-comb estimate of the spindle-rotation frequency, independent of the nominal rpm.

    Periodogram of the summed, de-meaned 3-axis force over the whole retained cut (~0.07 Hz bins);
    the comb score of f0 is the mean log-PSD at n * f0, n = 1..n_harmonics (linear interpolation).
    The search covers 12-30 Hz (720-1800 rpm), i.e. both nominal speeds with a wide margin; a coarse
    0.0005 Hz grid is refined to 0.00002 Hz around the best value.
    Returns (f0, comb contrast in dB = best score minus the median score over the coarse grid).
    """
    x = (force_cut - force_cut.mean(axis=0)).sum(axis=1)
    freq, psd = signal.periodogram(x, fs=fs, window="hann")
    logp = np.log10(psd + 1e-30)
    n = np.arange(1, n_harmonics + 1)

    def score(grid):
        return np.interp(np.outer(grid, n), freq, logp).mean(axis=1)

    coarse = np.arange(search_hz[0], search_hz[1], 0.0005)
    sc = score(coarse)
    k = int(np.argmax(sc))
    fine = np.arange(coarse[k] - 0.001, coarse[k] + 0.001, 0.00002)
    sf = score(fine)
    j = int(np.argmax(sf))
    return float(fine[j]), float(10 * (sf[j] - np.median(sc)))


def cut_windows(x: np.ndarray, fs: float, starts_s: np.ndarray, width_s: float) -> np.ndarray:
    w = int(round(width_s * fs))
    idx = np.round(starts_s * fs).astype(int)
    if idx.min() < 0 or idx.max() + w > len(x):
        raise ValueError("record does not cover the requested window")
    return np.stack([x[i:i + w].T for i in idx]).astype(np.float32)


def zone_spans(cut_start: float, cut_end: float, cfg: PipelineConfig) -> list[tuple[float, float]]:
    """Retained [start, end] (s, force clock) of every zone: thirds of the detected cut, minus
    `end_exclusion_s` at the cut start and end (zone 1 and zone 3 are shortened, zone 2 is complete)."""
    n = cfg.process.zone_count
    edges = cut_start + np.arange(n + 1) * (cut_end - cut_start) / n
    lo, hi = cut_start + cfg.process.end_exclusion_s, cut_end - cfg.process.end_exclusion_s
    spans = [(max(edges[z], lo), min(edges[z + 1], hi)) for z in range(n)]
    if any(b - a <= cfg.signal.context_s for a, b in spans):
        raise ValueError(f"a retained zone span is shorter than the {cfg.signal.context_s} s context")
    return spans


def window_starts(span: tuple[float, float], cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Start times of the T windows and of their 0.5 s contexts (context centred on the window,
    shifted to stay inside the span)."""
    w, c = cfg.signal.window_ms / 1000.0, cfg.signal.context_s
    starts = np.linspace(span[0], span[1] - w, cfg.signal.windows_per_zone)
    ctx = np.clip(starts + w / 2 - c / 2, span[0], span[1] - c)
    return starts, ctx


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------
class TrialProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.fx = WindowFeatureExtractor(cfg.signal)

    def run(self, row: dict) -> dict:
        cfg = self.cfg
        out = {"trial_id": row["trial_id"], "nominal_rpm": row["spindle_speed_rpm"],
               "force_pairing_confidence": row.get("force_pairing_confidence", "")}
        try:
            force = read_force(PROJECT_DIR / row["force_csv"])
            accel, afs = read_acceleration(PROJECT_DIR / row["acceleration_csv"])
            fs = cfg.signal.force_fs_hz
            if abs(afs - cfg.signal.acceleration_fs_hz) > 1:
                raise ValueError(f"accelerometer sampled at {afs:.0f} Hz")
            start, end, env = detect_cut(force, cfg)
            lag, r = align_acceleration(start, end, len(env), accel, afs, cfg)
            spans = zone_spans(start, end, cfg)
            retained = force[int(spans[0][0] * fs): int(spans[-1][1] * fs)]
            f0, contrast = measure_spindle_hz(retained, fs)
            out.update(cut_start_s=round(start, 3), cut_end_s=round(end, 3), cut_duration_s=round(end - start, 3),
                       cut_length_mm=round((end - start) * cfg.process.feed_mm_s, 2),
                       accel_lag_s=round(lag, 3), accel_alignment_r=round(r, 3),
                       accel_ok=bool(r >= cfg.cut.alignment_ok_r),
                       spindle_hz_measured=round(f0, 4), spindle_rpm_measured=round(60 * f0, 2),
                       spindle_comb_contrast_db=round(contrast, 2),
                       mean_force_resultant_n=round(float(np.linalg.norm(
                           retained - np.median(force[: int(cfg.cut.baseline_s * fs)], axis=0), axis=1).mean()), 3))
            if start + lag < 0 or end + lag > len(accel) / afs:
                raise ValueError("accelerometer record does not cover the cut")

            feats, f_raw, a_raw, labels = [], [], [], []
            w = cfg.signal.window_ms / 1000.0
            for z, span in enumerate(spans):
                st, ctx = window_starts(span, cfg)
                fw = cut_windows(force, fs, st, w)
                fc = cut_windows(force, fs, ctx, cfg.signal.context_s)
                aw = cut_windows(accel, afs, st + lag, w)
                ac = cut_windows(accel, afs, ctx + lag, cfg.signal.context_s)
                feats.append([np.concatenate([self.fx.window_features(fw[t], fc[t], fs, f0),
                                              self.fx.window_features(aw[t], ac[t], afs, f0)])
                              for t in range(len(st))])
                f_raw.append(signal.resample_poly(fw, 1, cfg.signal.encoder_decimation, axis=-1))
                a_raw.append(signal.resample_poly(aw, 1, cfg.signal.encoder_decimation, axis=-1))

                zone_force = force[int(span[0] * fs): int(span[1] * fs)].T
                peak, prom = zone_spectrum_peak(zone_force, fs, cfg.signal.chatter_band_hz, cfg.labels.label_nperseg)
                n_h = round(peak / f0)
                lab = {"zone_id": z, "zone_start_s": round(span[0], 3), "zone_end_s": round(span[1], 3),
                       "zone_duration_s": round(span[1] - span[0], 3), "peak_hz": round(peak, 2),
                       "prominence_db": round(prom, 2), "harmonic_number": int(n_h),
                       "dist_to_spindle_harmonic_hz": round(abs(peak - n_h * f0), 3),
                       "dist_to_nominal_harmonic_hz": round(abs(peak - round(peak / (out["nominal_rpm"] / 60))
                                                                * out["nominal_rpm"] / 60), 3)}
                for thr in cfg.labels.thresholds_db:
                    lab[f"label_{int(thr)}db"] = chatter_label(peak, prom, f0, cfg.signal.harmonic_tolerance_hz, thr)
                labels.append(lab)

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE_DIR / f"{row['trial_id']}.npz",
                                features=np.asarray(feats, np.float32),     # [N, T, 60]
                                force_raw=np.asarray(f_raw, np.float32),    # [N, T, 3, L]
                                acceleration_raw=np.asarray(a_raw, np.float32))
            out.update(included=True, exclusion_reason="", zone_labels=labels)
        except Exception as exc:
            out.update(included=False, exclusion_reason=str(exc), zone_labels=[])
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    manifest = pd.read_csv(DATA_DIR / "trial_manifest.csv")
    if args.only:
        manifest = manifest[manifest.trial_id.isin(args.only.split(","))]
    proc = TrialProcessor(CONFIG)
    with ProcessPoolExecutor(args.workers) as pool:
        summaries = list(pool.map(proc.run, manifest.to_dict("records")))
    for s in summaries:
        if s["included"]:
            n20 = sum(l["label_20db"] for l in s["zone_labels"])
            print(f"[OK ] {s['trial_id']} rpm {s['spindle_rpm_measured']:.1f} (nominal {s['nominal_rpm']:.0f}), "
                  f"comb {s['spindle_comb_contrast_db']:.1f} dB, r={s['accel_alignment_r']:.2f}, unstable@20dB {n20}/3")
        else:
            print(f"[SKIP] {s['trial_id']}: {s['exclusion_reason']}")
    pd.DataFrame([{k: v for k, v in s.items() if k != "zone_labels"} for s in summaries]).to_csv(
        CACHE_DIR / "dataset_summary.csv", index=False)
    rows = [{"trial_id": s["trial_id"], **lab} for s in summaries if s["included"] for lab in s["zone_labels"]]
    pd.DataFrame(rows).to_csv(DATA_DIR / "zone_labels.csv", index=False)
    print("wrote", CACHE_DIR / "dataset_summary.csv", "and", DATA_DIR / "zone_labels.csv")


if __name__ == "__main__":
    main()
