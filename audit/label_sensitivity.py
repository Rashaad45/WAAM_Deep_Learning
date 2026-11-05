"""Item 11: label sensitivity to harmonic tolerance and prominence threshold (read-only).

Recomputes the zone PSD exactly as ZoneChatterLabeler does (same cut times, same 6 zones),
then re-applies the rule on a grid. Writes nothing into the project.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
from config import CONFIG  # noqa: E402
from prepare_dataset import RawFileReader  # noqa: E402

summary = pd.read_csv(ROOT / "data/processed/dataset_summary.csv")
manifest = pd.read_csv(ROOT / "data/trial_manifest.csv").set_index("trial_id")
base = pd.read_csv(ROOT / "data/zone_labels.csv")
fs = CONFIG.signal.force_fs_hz
lo, hi = CONFIG.signal.chatter_band_hz
rows = []
for r in summary[summary.included].itertuples():
    force = RawFileReader.read_force(ROOT / manifest.loc[r.trial_id, "force_csv"])
    zl = (r.cut_end_s - r.cut_start_s) / 6
    f0 = r.spindle_speed_rpm / 60
    for z in range(6):
        zs = r.cut_start_s + z * zl
        x = force[int(zs * fs): int((zs + zl) * fs)].T
        x = x - x.mean(axis=1, keepdims=True)
        freq, psd = signal.welch(x, fs=fs, nperseg=min(32768, x.shape[1]), axis=1)
        psd = psd.sum(0)
        band = (freq >= lo) & (freq <= hi)
        bf, bp = freq[band], psd[band]
        k = int(np.argmax(bp))
        peak = float(bf[k])
        prom = float(10 * np.log10(bp[k] / np.median(bp)))
        dist = abs(peak - np.round(peak / f0) * f0)
        tooth = 4 * f0
        dist_tooth = abs(peak - np.round(peak / tooth) * tooth)
        row = dict(trial_id=r.trial_id, zone_id=z, material_type=r.material_type, rpm=r.spindle_speed_rpm,
                   peak_hz=peak, prominence_db=prom, dist_to_spindle_harmonic_hz=dist,
                   dist_to_tooth_harmonic_hz=dist_tooth, df_hz=float(freq[1] - freq[0]))
        for tol in (0.5, 1.0, 2.0):
            for thr in (15, 20, 25):
                row[f"lab_tol{tol}_prom{thr}"] = int(dist > tol and prom >= thr)
        rows.append(row)
    print("done", r.trial_id, flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / "label_sensitivity_zones.csv", index=False)
m = d.merge(base[["trial_id", "zone_id", "chatter_label"]], on=["trial_id", "zone_id"])
print("\nreproduces stored labels at tol=2/prom=20:", bool((m.chatter_label == m["lab_tol2.0_prom20"]).all()))
grid = []
for tol in (0.5, 1.0, 2.0):
    for thr in (15, 20, 25):
        c = f"lab_tol{tol}_prom{thr}"
        grid.append(dict(tolerance_hz=tol, prominence_db=thr, n_unstable=int(m[c].sum()),
                         n_unstable_waam=int(m[m.material_type == "WAAM"][c].sum()),
                         n_unstable_wrought=int(m[m.material_type == "WROUGHT"][c].sum()),
                         n_changed_vs_baseline=int((m[c] != m.chatter_label).sum()),
                         trials_with_positives=int(m[m[c] == 1].trial_id.nunique())))
g = pd.DataFrame(grid)
g.to_csv(OUT / "label_sensitivity_grid.csv", index=False)
print(g.to_string(index=False))
print("\nzones with prominence within +/-2 dB of the 20 dB threshold:",
      int(((d.prominence_db > 18) & (d.prominence_db < 22)).sum()), "/", len(d))
print("baseline-stable zones rejected ONLY by the harmonic test (prom>=20 but on harmonic):",
      int(((d.prominence_db >= 20) & (d.dist_to_spindle_harmonic_hz <= 2)).sum()))
print(d[(d.trial_id == "trial_16")][["zone_id", "peak_hz", "prominence_db", "dist_to_spindle_harmonic_hz"]].round(2).to_string(index=False))
