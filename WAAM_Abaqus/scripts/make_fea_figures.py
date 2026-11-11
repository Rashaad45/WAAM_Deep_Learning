"""Journal figures for the WAAM thermal FEA (plain Python 3 + matplotlib).

    python make_fea_figures.py --run final --goldak-a 5     # production run (12 layers)
    python make_fea_figures.py --verification ...            # see main() for the options

Figures are written to WAAM_Abaqus/figures/<name>.png (300 dpi) and .pdf (vector), in the same
style as waam_gnn_pipeline/src/make_journal_figures.py (Arial 8 pt, no top/right spines).

  fig_fea_setup            mesh, probe positions, Goldak heat-source distribution
  fig_fea_sections         longitudinal mid-plane temperature at layers 1, 6 and 12
  fig_fea_fusion           transverse sections through the pool, re-melt map, re-melt depth
  fig_fea_thermal_cycles   probe thermal cycles (full build and first three passes)
  fig_fea_layer_metrics    peak T, melt-pool size, interpass T, cooling rate vs. layer
  fig_fea_verification     mesh refinement, heat-input and boundary-condition sensitivity
  fig_fea_calibration      Goldak depth calibration sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fea_analysis import RES, T_LIQ, T_SOL, YC, Result, steady_segments  # noqa: E402
from generate_waam_inp import (GOLDAK, LAYER_H, N_SEG, SEG_L, SUB_T, WALL_X0, WALL_Y0,  # noqa: E402
                               WALL_Y1, Mesh)

FIG = HERE.parent / "figures"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
                     "lines.linewidth": 1.2, "savefig.facecolor": "white"})
# validated categorical slots 1-3 (dataviz reference palette, light mode)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
TCMAP = "inferno"
T_LEVELS = np.arange(25, 801, 25)          # solid-state detail; > 800 degC saturates (extend="max")
T_TICKS = [25, 200, 400, 571, 635, 800]


SUFFIX = ""          # appended to every figure name (--suffix)


def save(fig, name, tight=True):
    FIG.mkdir(parents=True, exist_ok=True)
    name = name + SUFFIX
    for suf in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suf}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def panel(ax, letter, x=-0.02):
    ax.text(x, 1.02, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def layer_ticks(n):
    step = 1 if n <= 12 else 3 if n <= 30 else 5
    return [1] + list(range(step, n + 1, step)) if step > 1 else list(range(1, n + 1))


def layer_colors(n):
    return plt.get_cmap("Reds")(np.linspace(0.35, 0.95, n))


# ====================================================================== activation
def active_extent(step: str, nl: int, nseg: int):
    """(complete layers, current layer, x_end of current layer) for a step name."""
    if step.startswith("L") and "_S" in step:
        l, s = int(step[1:3]), int(step[5:7])
        return l - 1, l, WALL_X0 + s * SEG_L
    if step.startswith("Interpass"):
        return int(step[-2:]), None, None
    if step.startswith("Final"):
        return nl, None, None
    return 0, None, None


def cell_active(xc, zc, done, cur, xend, lh=LAYER_H):
    if zc <= SUB_T:
        return True
    lay = int(np.ceil((zc - SUB_T) / lh - 1e-9))
    if lay <= done:
        return True
    return cur is not None and lay == cur and xc <= xend + 1e-9


def section(res: Result, frame: int, plane: str, value: float, values=None):
    """Triangulated mesh plane ('y' = longitudinal, 'x' = transverse) restricted to active cells.

    Returns (Triangulation in mm, nodal values). `values` overrides the frame temperatures.
    """
    xs, ys, zs = res.xs, res.ys, res.zs
    a = xs if plane == "y" else ys
    vals = res.T[frame] if values is None else values
    idx = -np.ones((len(zs), len(a)), int)
    for k, z in enumerate(zs):
        for i, c in enumerate(a):
            n = res.node(c, value, z) if plane == "y" else res.node(value, c, z)
            if n is not None:
                idx[k, i] = n
    done, cur, xend = active_extent(res.step[frame], res.layers, res.nseg)
    tris = []
    for k in range(len(zs) - 1):
        for i in range(len(a) - 1):
            q = idx[k:k + 2, i:i + 2]
            if (q < 0).any():
                continue
            cx = 0.5 * (a[i] + a[i + 1]) if plane == "y" else value
            cz = 0.5 * (zs[k] + zs[k + 1])
            if plane == "x" and cz > SUB_T and not (WALL_Y0 - 1e-9 <= 0.5 * (a[i] + a[i + 1]) <= WALL_Y1 + 1e-9):
                continue
            if cell_active(cx, cz, done, cur, xend, res.lh):
                p00, p01 = k * len(a) + i, k * len(a) + i + 1
                p10, p11 = (k + 1) * len(a) + i, (k + 1) * len(a) + i + 1
                tris += [(p00, p01, p11), (p00, p11, p10)]
    A, Z = np.meshgrid(a * 1e3, zs * 1e3)
    v = np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0.0).ravel()
    return mtri.Triangulation(A.ravel(), Z.ravel(), np.array(tris)), v


def frame_of(res: Result, step: str):
    hit = np.where(res.step == step)[0]
    return int(hit[-1])


def draw_section(ax, res, frame, plane, value, norm, outline=True, values=None, iso=True):
    tri, v = section(res, frame, plane, value, values)
    cs = ax.tricontourf(tri, np.clip(v, T_LEVELS[0], None), levels=T_LEVELS, cmap=TCMAP, norm=norm,
                        extend="max")
    if iso:
        ax.tricontour(tri, v, levels=[T_SOL, T_LIQ], colors=["#9ad0ff", "white"], linewidths=[0.6, 0.9],
                      linestyles=["--", "-"])
    if outline:
        top = (SUB_T + res.layers * res.lh) * 1e3
        lo, hi = ((WALL_X0, WALL_X0 + res.nseg * SEG_L) if plane == "y" else (WALL_Y0, WALL_Y1))
        ax.plot([lo * 1e3, lo * 1e3, hi * 1e3, hi * 1e3], [SUB_T * 1e3, top, top, SUB_T * 1e3],
                color="#b5b4ae", lw=0.5, ls=":")
    ax.set_aspect("equal")
    return cs


def remelt_field(res: Result):
    """Per node: max temperature reached AFTER the node's own layer was completed.

    A node on the interface between layers k and k+1 belongs to layer k, so its value is the
    temperature it reaches when layer k+1 (and later layers) are deposited: >= liquidus means
    that the interface is re-melted, i.e. metallurgically bonded.
    """
    z = res.mesh.coords[:, 2]
    lay = np.where(z <= SUB_T + 1e-9, 0, np.ceil((z - SUB_T) / res.lh - 1e-9)).astype(int)
    out = np.full(len(z), np.nan)
    for k in range(0, res.layers):
        start = np.where(res.step == f"L{k + 1:02d}_S01")[0][0]
        sel = lay == k
        out[sel] = np.nanmax(res.T[start:, sel], axis=0)
    out[lay == res.layers] = np.nan
    return out


def norm_T():
    return BoundaryNorm(T_LEVELS, 256, extend="max")


# ====================================================================== figures
def fig_setup(res: Result):
    m = Mesh(1, res.layers, res.nseg, res.lh, res.mesh.rows)
    fig = plt.figure(figsize=(7.2, 4.7))
    ax = fig.add_axes([-0.07, 0.38, 1.14, 0.60], projection="3d")
    X = m.coords * 1e3
    faces, cols = [], []
    sub_top = len(m.zs) - 2
    for e, (reg, i, j, k) in enumerate(m.info):
        c = X[m.conn[e] - 1]
        quads = []
        if reg == "S":
            if k == sub_top:
                quads.append(c[[4, 5, 6, 7]])
            if j == 0:
                quads.append(c[[0, 1, 5, 4]])
            if i == 0:
                quads.append(c[[0, 3, 7, 4]])
        else:
            if k == len(m.zw) - 1:
                quads.append(c[[4, 5, 6, 7]])
            if j == m.iy0:
                quads.append(c[[0, 1, 5, 4]])
            if i == m.ix0:
                quads.append(c[[0, 3, 7, 4]])
        for q in quads:
            faces.append(q)
            cols.append("#d9dde3" if reg == "S" else "#f3c9a8")
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors="#5f6670", linewidths=0.1))
    ax.quiver(20, 37.5, 70, 60, 0, 0, color=C2, arrow_length_ratio=0.15, lw=1.3)
    ax.text(85, 37.5, 68, "torch travel 800 mm min$^{-1}$, every layer from x = 5 mm", color=INK, fontsize=6.5)
    ax.set_box_aspect((240, 75, 61), zoom=1.35)
    ax.set_xlim(0, 240), ax.set_ylim(0, 75), ax.set_zlim(0, 61)
    ax.set_xticks([0, 60, 120, 180, 240]), ax.set_yticks([0, 75]), ax.set_zticks([0, 25, 61])
    ax.view_init(elev=16, azim=-72)
    ax.set_xlabel("x (mm)", labelpad=2), ax.set_ylabel("y (mm)", labelpad=-2)
    ax.set_zlabel("z (mm)", labelpad=-4)
    ax.tick_params(pad=0, labelsize=6.5)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False)
        a._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    fig.text(0.04, 0.97, "(a)", fontsize=9, fontweight="bold", va="top")
    lh, nl = res.lh * 1e3, res.layers
    dz = lh / m.rows
    fig.text(0.09, 0.965, f"Al 5083 substrate 240 x 75 x 25 mm (grey); ER5356 wall 230 x 6 mm, {nl} layers x "
             f"{lh:g} mm (orange);\n{len(m.conn):,} linear heat-transfer bricks (DC3D8), {len(m.coords):,} nodes; "
             f"wall elements 2.5 x 1.5 x {dz:g} mm".replace(",", " "), fontsize=7, va="top")
    # (b) probe positions on the longitudinal mid-plane
    ax = fig.add_axes([0.08, 0.08, 0.43, 0.30])
    ax.add_patch(Rectangle((0, 0), 240, 25, fc="#d9dde3", ec=MUTED, lw=0.6))
    for l in range(nl):
        ax.add_patch(Rectangle((5, 25 + lh * l), 230, lh, fc="#f3c9a8", ec="#c98b62", lw=0.25))
    for name, (n,) in m.nsets.items():
        p = X[n - 1]
        wall = name.startswith("P_WALL")
        ax.plot(p[0], p[2], "o", ms=2.6 if nl > 12 else 3.2, color=C1 if wall else C3, mec="white", mew=0.3)
    ax.annotate(f"layer-top probes L1–L{nl}\n(x = 120 mm, wall centre-line)", xy=(121, 46), xytext=(135, 68),
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.annotate("substrate probes\n(top, mid-depth, bottom)", xy=(121, 14), xytext=(135, 4),
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.set_xlim(-2, 242), ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"), ax.set_ylabel("z (mm)")
    panel(ax, "b")
    # (c) heat source distribution on the longitudinal centre plane, close-up
    ax2 = fig.add_axes([0.62, 0.08, 0.27, 0.27])
    g = res.goldak
    lsrc = max(3, int(round(18 / lh)))              # a layer about 18 mm above the substrate
    top = lsrc * lh                                  # mm above substrate
    xc, zc = 0.120, SUB_T + top * 1e-3
    xx = np.linspace(xc - 9e-3, xc + 6e-3, 400)
    zz = np.linspace(zc - 9e-3, zc, 300)
    Xg, Zg = np.meshgrid(xx, zz)
    dx = Xg - xc
    c = np.where(dx >= 0, g["cf"], g["cr"])
    f = np.where(dx >= 0, g["ff"], g["fr"])
    r2 = dx ** 2 / c ** 2 + (Zg - zc) ** 2 / g["a"] ** 2
    q = 6 * np.sqrt(3) * f * res.Q / (g["a"] * g["b"] * c * np.pi ** 1.5) * np.exp(-3 * r2)
    q = np.where(r2 <= 2.5, q, np.nan)
    im = ax2.contourf((Xg - xc) * 1e3, (Zg - SUB_T) * 1e3, q / 1e9, levels=16, cmap="Oranges")
    for l in range(lsrc - int(9 / lh) - 1, lsrc + 1):
        ax2.axhline(l * lh, color=MUTED, lw=0.5)
        if l > lsrc - 3 and l > 0:
            ax2.text(-8.8, (l - 0.5) * lh, f"layer {l}", fontsize=6, color=MUTED, va="center")
    for zg in np.arange(top - 9, top, dz):
        ax2.axhline(zg, color=MUTED, lw=0.2, alpha=0.4)
    for xg in np.arange(-10, 7.5, 2.5):
        ax2.axvline(xg, color=MUTED, lw=0.25, alpha=0.5)
    ax2.annotate("", xy=(0.2, top - g["a"] * 1e3), xytext=(0.2, top), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax2.text(0.5, top - g["a"] * 6e2, f"a = {g['a'] * 1e3:g} mm", fontsize=6.5)
    ax2.annotate("", xy=(-g["cr"] * 1e3, top + 0.5), xytext=(0, top + 0.5), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax2.text(-g["cr"] * 1e3 / 2, top + 0.8, f"c$_r$ = {g['cr'] * 1e3:g}", ha="center", fontsize=6.5)
    ax2.annotate("", xy=(g["cf"] * 1e3, top + 0.5), xytext=(0, top + 0.5), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax2.text(g["cf"] * 1e3 / 2, top + 0.8, f"c$_f$ = {g['cf'] * 1e3:g}", ha="center", fontsize=6.5)
    ax2.set_ylim(top - 9, top + 2)
    ax2.set_xlabel("x − x$_{torch}$ (mm)"), ax2.set_ylabel("z − 25 mm (mm)")
    ax2.set_aspect("equal")
    cax = fig.add_axes([0.905, 0.08, 0.012, 0.27])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("q (GW m$^{-3}$)")
    ax2.text(0.0, 1.13, f"Goldak source, centre plane: Q = {res.Q:.0f} W,\nb = {g['b'] * 1e3:g} mm, "
             f"f$_f$ = {g['ff']:g}, f$_r$ = {g['fr']:g}; grey lines = mesh",
             transform=ax2.transAxes, fontsize=6.5, va="bottom")
    panel(ax2, "c", x=-0.2)
    save(fig, "fig_fea_setup")


def fig_sections(res: Result):
    norm = norm_T()
    seg = res.nseg // 2
    nl = res.layers
    picks = [(f"L01_S{seg:02d}", "layer 1"), (f"L{nl // 2:02d}_S{seg:02d}", f"layer {nl // 2}"),
             (f"L{nl:02d}_S{seg:02d}", f"layer {nl}"), (f"Interpass_L{nl - 1:02d}", None)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(7.2, 6.0), sharex=True)
    t0 = res.step_start("L01_S01")
    for ax, (st, lab), letter in zip(axes, picks, "abcd"):
        f = frame_of(res, st)
        cs = draw_section(ax, res, f, "y", YC, norm)
        t = res.time[f] - t0
        title = (f"torch at mid-length of {lab}, t = {t:.0f} s" if lab
                 else f"end of the 240 s dwell after layer {nl - 1} (t = {t:.0f} s): interpass state")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_ylabel("z (mm)")
        ax.set_ylim(0, 63)
        panel(ax, letter)
    axes[-1].set_xlabel("x (mm)")
    cb = fig.colorbar(cs, ax=axes, shrink=0.85, pad=0.015, ticks=T_TICKS)
    cb.set_label("temperature (°C);  > 800 °C saturated")
    save(fig, "fig_fea_sections")


def fig_fusion(res: Result, mp: pd.DataFrame):
    norm = norm_T()
    seg = res.nseg // 2
    xs = WALL_X0 + (seg - 0.5) * SEG_L
    picks = sorted({1, 3, res.layers // 2, res.layers})
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.30, wspace=0.35)
    for c, l in enumerate(picks):
        ax = fig.add_subplot(gs[0, c])
        f = frame_of(res, f"L{l:02d}_S{seg:02d}")
        cs = draw_section(ax, res, f, "x", xs, norm)
        zi = (SUB_T + (l - 1) * res.lh) * 1e3
        ax.axhline(zi, color=C3, lw=0.9, ls="--")
        ax.set_xlim(28, 47)
        ax.set_ylim(zi - 8, zi + 4)
        ax.set_title(f"layer {l}", fontsize=7.5)
        ax.set_xlabel("y (mm)")
        if c == 0:
            ax.set_ylabel("z (mm)")
            panel(ax, "a", x=-0.25)
    cb = fig.colorbar(cs, ax=fig.axes[:4], shrink=0.9, pad=0.02, ticks=T_TICKS)
    cb.set_label("T (°C)")
    # re-melt map in the longitudinal plane
    ax = fig.add_subplot(gs[1, 0:3])
    rm = remelt_field(res)
    last = len(res.step) - 1
    tri, v = section(res, last, "y", YC, values=np.nan_to_num(rm, nan=0.0))
    # drop triangles of the top layer (no later pass) for this map
    zt = tri.y[tri.triangles].mean(1)
    tri.set_mask(zt > (SUB_T + (res.layers - 1) * res.lh) * 1e3)
    im = ax.tricontourf(tri, np.clip(v, 25, None), levels=T_LEVELS, cmap=TCMAP, norm=norm, extend="max")
    ax.tricontour(tri, v, levels=[T_LIQ], colors="white", linewidths=0.8)
    ax.set_xlim(0, (2 * WALL_X0 + res.nseg * SEG_L) * 1e3)
    ax.set_ylim(15, (SUB_T + (res.layers - 1) * res.lh) * 1e3 + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"), ax.set_ylabel("z (mm)")
    ax.set_title("re-heating map: highest temperature reached after a layer is finished\n"
                 "(white = liquidus: re-melted by the following pass)", fontsize=7, loc="left")
    panel(ax, "b", x=-0.06)
    # re-melt depth vs layer
    ax = fig.add_subplot(gs[1, 3])
    st = steady_segments(mp, res.nseg).groupby("layer")
    md = st.remelt_depth_mm.agg(["mean", "std"])
    lo = min(-1.5, md["mean"].min() - 0.3)
    ax.axvspan(lo, 0, color="#f4d6d2", lw=0)
    ax.text(lo + 0.08, res.layers + 0.4, "lack of\nfusion", ha="left", va="bottom", fontsize=6.3, color="#9b2c22")
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.plot(md["mean"], md.index, "o", color=C1, ms=3.5)
    ax.set_ylim(res.layers + 0.7, -0.6)
    ax.set_xlim(lo, max(1.5, md["mean"].max() + 0.4))
    ax.set_yticks(layer_ticks(res.layers))
    ax.set_xlabel("re-melt depth into the\nlayer below (mm)")
    ax.set_ylabel("layer")
    ax.grid(axis="x", color=GRID, lw=0.5)
    panel(ax, "c", x=-0.15)
    save(fig, "fig_fea_fusion")


def fig_cycles(res: Result):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    t0 = res.step_start("L01_S01")
    nl = res.layers
    show = sorted({1, max(1, nl // 4), nl // 2, 3 * nl // 4, nl})
    cols = layer_colors(len(show))
    for l, c in zip(show, cols):
        t, T = res.probe_series(f"P_WALL_L{l:02d}")
        a1.plot((t - t0) / 60, T, color=c, lw=0.7, label=f"layer {l} top")
    t, T = res.probe_series("P_SUB_BOT")
    a1.plot((t - t0) / 60, T, color="#7f8c8d", lw=1.0, ls="--", label="substrate bottom")
    for ax in (a1, a2):
        ax.axhline(T_LIQ, color=MUTED, lw=0.5, ls="-")
        ax.axhline(T_SOL, color=MUTED, lw=0.5, ls="--")
        ax.grid(axis="y", color=GRID, lw=0.5)
    xr = (res.probes.time_s.max() - t0) / 60
    a1.text(0.99 * xr, T_LIQ + 15, "liquidus", fontsize=6, color=MUTED, ha="right")
    a1.text(0.99 * xr, T_SOL - 55, "solidus", fontsize=6, color=MUTED, ha="right")
    a1.set_xlim(0, (res.probes.time_s.max() - t0) / 60)
    a1.set_xlabel("time from start of deposition (min)"), a1.set_ylabel("temperature (°C)")
    a1.legend(frameon=False, ncol=2, loc="upper right", fontsize=6.3)
    a1.set_ylim(0, 1350)
    panel(a1, "a")
    t, T = res.probe_series("P_WALL_L01")
    sel = (t - t0) <= 3 * 257.25 + 40
    a2.plot(t[sel] - t0, T[sel], color=cols[0], lw=0.9)
    a2.set_yscale("log")
    a2.set_yticks([30, 50, 100, 200, 400, 635, 1000])
    a2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a2.set_ylim(20, 1500)
    a2.set_xlabel("time (s)"), a2.set_ylabel("temperature (°C, log scale)")
    for k in range(4):
        ts = res.step_start(f"L{k + 1:02d}_S01") - t0
        if np.isfinite(ts):
            a2.axvspan(ts, ts + res.nseg * SEG_L / (0.8 / 60), color="#fbe3d6", lw=0)
            a2.text(ts + 8, 24, f"pass {k + 1}", fontsize=6, color=MUTED)
    a2.set_title("layer-1 probe: own pass and re-heating by later passes", fontsize=7)
    panel(a2, "b")
    save(fig, "fig_fea_thermal_cycles")


def fig_layer_metrics(res: Result, lm: pd.DataFrame, mp: pd.DataFrame):
    st = steady_segments(mp, res.nseg).groupby("layer")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    axes = axes.ravel()
    L = lm.layer
    ax = axes[0]
    ax.plot(L, lm.peak_T, "o-", color=C1, ms=3.5, label="layer-top probe (x = 120 mm)")
    ax.plot(st.peak_T.mean().index, st.peak_T.mean(), "s--", color=C2, ms=3,
            label="field maximum (steady segments)")
    ax.set_ylabel("peak temperature (°C)")
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    ax = axes[1]
    ax.plot(st.depth_mm.mean().index, st.depth_mm.mean(), "o-", color=C1, ms=3.5, label="depth below layer top")
    ax.plot(st.length_mm.mean().index, st.length_mm.mean(), "s--", color=C2, ms=3, label="length")
    ax.axhline(res.lh * 1e3, color=MUTED, lw=0.6, ls=":")
    ax.text(res.layers, res.lh * 1e3 + 0.3, f"layer height {res.lh * 1e3:g} mm", fontsize=6, color=MUTED, ha="right")
    ax.set_ylabel("melt pool, liquidus isotherm (mm)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=6.3, loc="center right")
    ax = axes[2]
    ok = lm.interpass_T_at_next_layer.notna()
    ax.plot(L[ok], lm.interpass_T_at_next_layer[ok], "o-", color=C1, ms=3.5, label="layer top (mid-length)")
    ax.plot(L[ok], lm.substrate_bottom_T_at_next_layer[ok], "s--", color=C3, ms=3, label="substrate bottom")
    ax.set_ylabel("temperature when the\nnext layer starts (°C)")
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    ax = axes[3]
    ax.semilogy(L, lm.solidification_rate_C_s, "o-", color=C1, ms=3.5, label="635→571 °C (solidification)")
    ax.semilogy(L, lm.cooling_rate_500_300_C_s, "s--", color=C2, ms=3, label="500→300 °C")
    ax.set_ylabel("cooling rate (°C s$^{-1}$)")
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(frameon=False, fontsize=6.3)
    for a, letter in zip(axes, "abcd"):
        a.set_xlabel("layer")
        a.set_xticks(layer_ticks(res.layers))
        a.grid(axis="y", color=GRID, lw=0.5)
        panel(a, letter)
    fig.tight_layout()
    save(fig, "fig_fea_layer_metrics")


def fig_verification(sens: dict, probe="P_WALL_L03", n_layers=4):
    """sens: {label: (Result, layer_metrics, meltpool)}; first entry = reference (production run)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    colors = ["#8c8b86", C1, C2, C3, "#6250d6"]
    styles = ["-", "--", "-.", ":", (0, (5, 1))]
    ax = axes[0]
    for k, (lab, (r, lm, mp)) in enumerate(sens.items()):
        t0 = r.step_start("L01_S01")
        t, T = r.probe_series(probe)
        sel = (t - t0) <= n_layers * 257.25
        ax.plot(t[sel] - t0, T[sel], color=colors[k], ls=styles[k], lw=0.9, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([30, 100, 300, 635, 1000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axhline(T_LIQ, color=MUTED, lw=0.5)
    ax.set_xlabel("time (s)"), ax.set_ylabel("temperature (°C, log scale)")
    ax.set_title(f"layer-{int(probe[-2:])} top probe, first {n_layers} passes", fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.5)
    panel(ax, "a")
    labels = list(sens)
    w = 0.84 / len(labels)
    for ax, key, ylab, letter in [(axes[1], "peak_T", "probe peak temperature (°C)", "b"),
                                  (axes[2], "remelt", "re-melt depth into layer below (mm)", "c")]:
        for k, lab in enumerate(labels):
            r, lm, mp = sens[lab]
            if key == "peak_T":
                y = lm.set_index("layer").peak_T
            else:
                y = steady_segments(mp, r.nseg).groupby("layer").remelt_depth_mm.mean()
            y = y[y.index <= n_layers]
            ax.bar(y.index + (k - (len(labels) - 1) / 2) * w, y.values, w * 0.92, color=colors[k],
                   label=lab, edgecolor="white", linewidth=0.3)
        ax.set_xticks(range(1, n_layers + 1))
        ax.set_xlabel("layer"), ax.set_ylabel(ylab)
        ax.axhline(0, color=INK, lw=0.5)
        ax.grid(axis="y", color=GRID, lw=0.5)
        panel(ax, letter)
    axes[1].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    save(fig, "fig_fea_verification")


def fig_calibration(sweep: dict, layer_runs: dict, nseg: int):
    """sweep: {a_mm: meltpool} for 3 mm layers; layer_runs: {label: (meltpool, n_layers)}."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a_vals = sorted(sweep)
    for l, c in zip([1, 2, 3], layer_colors(3)):
        md = []
        for av in a_vals:
            s = steady_segments(sweep[av], nseg)
            md.append(s.loc[s.layer == l, "remelt_depth_mm"].mean())
        axes[0].plot(a_vals, md, "o-", color=c, ms=3.5, label=f"layer {l}")
    axes[0].set_xlabel("Goldak depth semi-axis a (mm)")
    axes[0].set_xticks(a_vals)
    axes[0].set_title("3 mm layers: no setting of a fuses the layers", fontsize=7)
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower left")
    for (lab, (mp, n)), c, mk in zip(layer_runs.items(), [C2, C1, C3], ["s", "o", "^"]):
        md = steady_segments(mp, nseg).groupby("layer").remelt_depth_mm.mean()
        axes[1].plot(md.index, md.values, marker=mk, color=c, ms=3.5, label=lab)
    axes[1].set_xlabel("layer")
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_title("effect of layer height (a = 3 mm)", fontsize=7)
    axes[1].legend(frameon=False, fontsize=6.3, loc="lower right")
    for ax, letter in zip(axes, "ab"):
        lo = min(-2.1, ax.get_ylim()[0])
        ax.axhspan(lo, 0, color="#f4d6d2", lw=0, zorder=0)
        ax.set_ylim(lo, max(1.0, ax.get_ylim()[1]))
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel("re-melt depth into\nthe layer below (mm)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        panel(ax, letter)
    axes[0].text(a_vals[-1], lo + 0.1, "lack of fusion", ha="right", va="bottom", fontsize=6.3, color="#9b2c22")
    fig.tight_layout()
    save(fig, "fig_fea_calibration")


# ====================================================================== main
def load(run, refine=1, layers=12, nseg=N_SEG, goldak_a=None, q_scale=1.0, layer_h=LAYER_H, rows=2):
    r = Result(run, refine, layers, nseg, layer_h, rows)
    r.goldak = dict(GOLDAK)
    if goldak_a is not None:
        r.goldak["a"] = goldak_a * 1e-3
    r.Q = 0.8 * 18 * 102 * q_scale
    lm = pd.read_csv(RES / f"{run}_layer_metrics.csv")
    mp = pd.read_csv(RES / f"{run}_meltpool.csv")
    return r, lm, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--layer-height", type=float, default=1.5)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--goldak-a", type=float, default=None)
    ap.add_argument("--nseg", type=int, default=N_SEG)
    ap.add_argument("--only", nargs="*", default=None,
                    help="setup sections fusion cycles metrics verification calibration")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    global SUFFIX
    SUFFIX = a.suffix
    todo = a.only or ["setup", "sections", "fusion", "cycles", "metrics", "verification", "calibration"]
    lh = a.layer_height * 1e-3
    main_figs = [k for k in todo if k in ("setup", "sections", "fusion", "cycles", "metrics")]
    if main_figs:
        r, lm, mp = load(a.run, layers=a.layers, nseg=a.nseg, goldak_a=a.goldak_a, layer_h=lh, rows=a.rows)
        for key, fn in [("setup", lambda: fig_setup(r)), ("sections", lambda: fig_sections(r)),
                        ("fusion", lambda: fig_fusion(r, mp)), ("cycles", lambda: fig_cycles(r)),
                        ("metrics", lambda: fig_layer_metrics(r, lm, mp))]:
            if key in main_figs:
                fn()
    if "verification" in todo:
        runs = [(f"reference ({a.run})", a.run, 1, a.layers, 1.0), ("mesh refined x2", "v_r2_L4", 2, 4, 1.0),
                ("Q +10 %", "v_q110_L4", 1, 4, 1.1), ("Q −10 %", "v_q090_L4", 1, 4, 0.9),
                ("h$_{bottom}$ = 100", "v_hb100_L4", 1, 4, 1.0)]
        sens = {}
        for lab, run, ref, nl, qs in runs:
            if (RES / f"{run}_meltpool.csv").exists():
                sens[lab] = load(run, refine=ref, layers=nl, nseg=a.nseg, q_scale=qs, layer_h=lh, rows=a.rows)
        fig_verification(sens)
    if "calibration" in todo:
        sweep = {av: pd.read_csv(RES / f"cal_a{av}_meltpool.csv") for av in (3, 4, 5, 6)}
        layer_runs = {"3 mm layers": (sweep[3], 3),
                      "1.5 mm layers": (pd.read_csv(RES / "cal_h15_meltpool.csv"), 4),
                      "1.5 mm layers, 2 element rows": (pd.read_csv(RES / "cal_h15r2_meltpool.csv"), 4)}
        fig_calibration(sweep, layer_runs, 16)


if __name__ == "__main__":
    main()
