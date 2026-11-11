"""Thermal metrics from the extracted WAAM FEA results (plain Python 3).

    python fea_analysis.py                      # baseline (12 layers, original mesh)
    python fea_analysis.py --run r2_L1 --refine 2 --layers 1

Reads fea_results/<run>_field.npz and <run>_probes.csv (from extract_odb.py) and writes
  fea_results/<run>_layer_metrics.csv   per layer: probe peak T, cooling rate, time above
                                        solidus, re-melting by later layers, interpass T
  fea_results/<run>_meltpool.csv        per deposition segment: melt-pool length / depth,
                                        re-melt depth into the previous layer, fused width
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_waam_inp import (LAYER_H, N_SEG, SEG_L, SUB_T, WALL_X0, WALL_Y0, WALL_Y1,  # noqa: E402
                               Mesh)

RES = HERE.parent / "fea_results"
T_LIQ, T_SOL = 635.0, 571.0          # ER5356 liquidus / solidus used in the model (degC)
YC = 0.0375                          # wall centre line


class Result:
    """Field + probe results of one run, with the mesh rebuilt by the generator."""

    def __init__(self, run: str = "baseline", refine: int = 1, layers: int = 12, nseg: int = N_SEG,
                 layer_h: float = LAYER_H, rows: int = 2):
        self.run, self.refine, self.layers, self.nseg = run, refine, layers, nseg
        self.mesh = Mesh(refine, layers, nseg, layer_h, rows)
        self.lh = layer_h
        d = np.load(RES / f"{run}_field.npz")
        order = np.argsort(d["labels"])
        self.T = d["T"][:, order]
        self.time, self.step = d["time"], d["step"].astype(str)
        self.probes = pd.read_csv(RES / f"{run}_probes.csv")
        X = self.mesh.coords
        self.key = {tuple(np.round(p, 7)): i for i, p in enumerate(X)}
        self.xs = np.unique(np.round(X[:, 0], 7))
        self.ys = np.unique(np.round(X[:, 1], 7))
        self.zs = np.unique(np.round(X[:, 2], 7))

    # -------------------------------------------------------------- helpers
    def node(self, x, y, z):
        return self.key.get((round(x, 7), round(y, 7), round(z, 7)))

    def line(self, frame, axis, fixed):
        """Temperatures along a grid line: axis 0/1/2, fixed = the other two coordinates."""
        grid = (self.xs, self.ys, self.zs)[axis]
        pts, vals = [], []
        for c in grid:
            p = list(fixed)
            p.insert(axis, c)
            n = self.node(*p)
            if n is not None:
                pts.append(c)
                vals.append(self.T[frame, n])
        return np.array(pts), np.array(vals)

    def probe_series(self, name):
        n = self.mesh.nsets[name][0]
        return self.probes["time_s"].to_numpy(), self.probes[f"node_{n}"].to_numpy()

    def step_start(self, name):
        hit = self.probes.index[self.probes.step == name]
        return float(self.probes.time_s[hit[0]]) if len(hit) else np.nan

    # -------------------------------------------------------------- melt pool
    def meltpool(self):
        rows = []
        for f, s in enumerate(self.step):
            if not s.startswith("L") or "_S" not in s:
                continue
            l, k = int(s[1:3]), int(s[5:7])
            ztop = SUB_T + l * self.lh
            xc, xend = WALL_X0 + (k - 0.5) * SEG_L, WALL_X0 + k * SEG_L
            # length along the top centre line (the deposit ends at xend)
            x, t = self.line(f, 0, (YC, ztop))
            ok = (x <= xend + 1e-9) & (x >= WALL_X0 - 1e-9)
            x, t = x[ok], t[ok]
            rear = _crossing(x[::-1], t[::-1], xc, T_LIQ)
            front = _crossing(x, t, xc, T_LIQ)
            # depth along the vertical centre line through the source
            z, tz = self.line(f, 2, (xc, YC))
            ok = z <= ztop + 1e-9
            zb = _crossing(z[ok][::-1], tz[ok][::-1], ztop, T_LIQ)
            depth = ztop - zb if np.isfinite(zb) else 0.0
            # fused width at the interface with the layer below (or the substrate)
            zi = ztop - self.lh
            y, ty = self.line(f, 1, (xc, zi))
            ok = (y >= WALL_Y0 - 1e-9) & (y <= WALL_Y1 + 1e-9)
            width_int = _width(y[ok], ty[ok], T_SOL)
            rows.append(dict(layer=l, segment=k, step=s, time_s=self.time[f],
                             peak_T=float(np.nanmax(self.T[f])),
                             length_mm=1e3 * ((front if np.isfinite(front) else xc) -
                                              (rear if np.isfinite(rear) else xc)),
                             rear_mm=1e3 * (xc - rear) if np.isfinite(rear) else np.nan,
                             depth_mm=1e3 * depth, remelt_depth_mm=1e3 * (depth - self.lh),
                             fused_width_interface_mm=1e3 * width_int))
        return pd.DataFrame(rows)

    # -------------------------------------------------------------- per layer
    def layer_metrics(self):
        rows = []
        for l in range(1, self.layers + 1):
            name = f"P_WALL_L{l:02d}"
            t, T = self.probe_series(name)
            t0 = self.step_start(f"L{l:02d}_S01")
            t_next = self.step_start(f"L{l + 1:02d}_S01") if l < self.layers else np.nan
            own = (t >= t0) & ((t < t_next) if np.isfinite(t_next) else True)
            i_pk = np.argmax(np.where(own, T, -np.inf))
            after = np.arange(len(t)) > i_pk
            # first cooling through the solidification range
            c = after & own
            tl = _first_time_below(t[c], T[c], T_LIQ)
            ts = _first_time_below(t[c], T[c], T_SOL)
            t5 = _first_time_below(t[c], T[c], 500.0)
            t3 = _first_time_below(t[c], T[c], 300.0)
            above = (T >= T_SOL) & own
            dt = np.diff(t, prepend=t[0])
            # re-heating by later layers
            later = t >= t_next if np.isfinite(t_next) else np.zeros_like(t, bool)
            reheat_peaks = _peaks_by_layer(self, t, T, l)
            remelts = sum(1 for v in reheat_peaks.values() if v >= T_LIQ)
            interpass = np.interp(t_next, t, T) if np.isfinite(t_next) else np.nan
            rows.append(dict(
                layer=l, z_mm=1e3 * (SUB_T + l * self.lh),
                peak_T=T[i_pk],
                solidification_rate_C_s=(T_LIQ - T_SOL) / (ts - tl) if np.isfinite(ts - tl) and ts > tl else np.nan,
                cooling_rate_500_300_C_s=200.0 / (t3 - t5) if np.isfinite(t3 - t5) and t3 > t5 else np.nan,
                time_above_solidus_s=float((dt * above).sum()),
                reheat_peak_next_layer=reheat_peaks.get(l + 1, np.nan),
                n_remelts_by_later_layers=remelts,
                n_reheats_above_200C=sum(1 for v in reheat_peaks.values() if v >= 200),
                interpass_T_at_next_layer=interpass,
                T_before_own_layer=np.interp(t0, t, T) if np.isfinite(t0) else np.nan,
                wall_top_mean_T_at_next_layer=self._mean_top_before(l + 1),
                substrate_bottom_T_at_next_layer=(np.interp(t_next, *self.probe_series("P_SUB_BOT"))
                                                  if np.isfinite(t_next) else np.nan),
                any_later=bool(later.any())))
        return pd.DataFrame(rows).drop(columns="any_later")

    def _mean_top_before(self, l_next):
        """Mean temperature of the wall top surface just before layer l_next starts."""
        if l_next > self.layers:
            return np.nan
        frames = np.where(self.step == f"Interpass_L{l_next - 1:02d}")[0]
        if not len(frames):
            return np.nan
        f = frames[-1]
        z = SUB_T + (l_next - 1) * self.lh
        x, t = self.line(f, 0, (YC, z))
        return float(t[(x >= WALL_X0) & (x <= WALL_X0 + self.nseg * SEG_L)].mean())


def _crossing(x, t, x0, level):
    """Walk from x0 along x (array ordered in the walking direction) to where t drops below level."""
    start = np.searchsorted(x, x0) if x[0] <= x[-1] else np.searchsorted(-x, -x0)
    start = min(max(start, 0), len(x) - 1)
    if t[start] < level:
        # x0 itself may sit between nodes; check the neighbour
        if start > 0 and t[start - 1] >= level:
            start -= 1
        else:
            return np.nan
    for i in range(start, len(x) - 1):
        if t[i] >= level > t[i + 1]:
            return x[i] + (level - t[i]) * (x[i + 1] - x[i]) / (t[i + 1] - t[i])
    return x[-1]


def _width(y, t, level):
    hot = t >= level
    if not hot.any():
        return 0.0
    yy = np.linspace(y.min(), y.max(), 601)
    return float((np.interp(yy, y, t) >= level).mean() * (y.max() - y.min()))


def _first_time_below(t, T, level):
    idx = np.where(T < level)[0]
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return t[0]
    return t[i - 1] + (level - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1])


def _peaks_by_layer(res: Result, t, T, l):
    out = {}
    for m in range(l + 1, res.layers + 1):
        a = res.step_start(f"L{m:02d}_S01")
        b = res.step_start(f"L{m + 1:02d}_S01") if m < res.layers else t[-1] + 1
        sel = (t >= a) & (t < b)
        if sel.any():
            out[m] = float(T[sel].max())
    return out


def steady_segments(mp: pd.DataFrame, nseg: int = N_SEG) -> pd.DataFrame:
    """Segments away from the start/end transients (10-37 of 46 for the full wall)."""
    lo, hi = round(0.2 * nseg) + 1, round(0.8 * nseg)
    return mp[(mp.segment >= lo) & (mp.segment <= hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--nseg", type=int, default=N_SEG)
    ap.add_argument("--layer-height", type=float, default=LAYER_H * 1e3)
    ap.add_argument("--rows", type=int, default=2)
    a = ap.parse_args()
    r = Result(a.run, a.refine, a.layers, a.nseg, a.layer_height * 1e-3, a.rows)
    lm = r.layer_metrics()
    mp = r.meltpool()
    lm.to_csv(RES / f"{a.run}_layer_metrics.csv", index=False, float_format="%.4g")
    mp.to_csv(RES / f"{a.run}_meltpool.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(lm.round(2).to_string(index=False))
    steady = steady_segments(mp, a.nseg)
    print(steady.groupby("layer")[["peak_T", "length_mm", "depth_mm", "remelt_depth_mm",
                                   "fused_width_interface_mm"]].agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()
