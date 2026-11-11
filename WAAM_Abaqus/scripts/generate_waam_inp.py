"""Parametric generator for the WAAM Al 5356 thermal model (Abaqus/Standard, DC3D8).

With default arguments it rebuilds waam_al5356.inp (same nodes, elements, sets, element-birth
sequence, Goldak fluxes, film/radiation history and outputs), which is checked by
verify_generator.py. Options give the variants used for verification:

    --refine 2          halves the element edge in the wall and the adjacent substrate
    --layers 3          builds and deposits only the first N layers
    --q-scale 1.1       scales the net arc power (sensitivity to the assumed U and eta)
    --h-bottom 100      film coefficient on the substrate underside (clamping/table contact)
    --final-cooling 240 length of the cooling step after the last layer (s)

Model (Units: m, s, kg, J, W, degC):
  substrate Al 5083 240 x 75 x 25 mm; wall ER5356 230 x 6 mm, 12 layers x 3 mm
  arc I = 102 A, U = 18 V, eta = 0.8 -> Q = 1468.8 W; travel 800 mm/min; interpass 240 s
  Goldak double ellipsoid a = b = cf = 3 mm, cr = 6 mm, ff = 0.6, fr = 1.4, evaluated at
  element centroids (r^2 <= 2.5), renormalised so that sum(q V) = Q for every segment
  segment-wise moving source: 46 segments of 5 mm per layer, 0.375 s each
  h = 10 W/m2K on all free faces, emissivity 0.2, ambient 25 degC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ fixed model data
SUB_L, SUB_W, SUB_T = 0.240, 0.075, 0.025
WALL_X0, WALL_X1 = 0.005, 0.235
WALL_Y0, WALL_Y1 = 0.0345, 0.0405
LAYER_H, N_SEG, SEG_L = 0.003, 46, 0.005
TRAVEL = 0.8 / 60.0                       # m/s
I_ARC, U_ARC, ETA = 102.0, 18.0, 0.8
GOLDAK = dict(a=3e-3, b=3e-3, cf=3e-3, cr=6e-3, ff=0.6, fr=1.4)
R2_CUT = 2.5
INTERPASS = 240.0
T_AMB, T0 = 25.0, 25.0

MAT = {
    "AL5356": dict(k=[(116, 25), (121, 100), (129, 200), (136, 300), (142, 400), (147, 500),
                      (149, 571), (90, 635), (90, 800)], rho=2640,
                   cp=[(900, 25), (950, 100), (1000, 200), (1045, 300), (1090, 400), (1130, 500),
                       (1160, 571), (1180, 635), (1180, 800)], L=(390000, 571, 635)),
    "AL5083": dict(k=[(117, 25), (122, 100), (130, 200), (137, 300), (143, 400), (148, 500),
                      (150, 574), (90, 638), (90, 800)], rho=2660,
                   cp=[(900, 25), (950, 100), (1000, 200), (1045, 300), (1090, 400), (1130, 500),
                       (1160, 574), (1180, 638), (1180, 800)], L=(390000, 574, 638)),
}
SUB_PROBES = {"P_SUB_TOP_10MM": (0.12, 0.023844, SUB_T), "P_SUB_MID": (0.12, 0.0375, 0.014344),
              "P_SUB_BOT": (0.12, 0.0375, 0.0)}


# ------------------------------------------------------------------ mesh
def graded(extent: float, s0: float, ratio: float = 1.4) -> list[float]:
    """Element sizes growing by `ratio` from s0; the last element takes the remainder."""
    sizes, cum, s = [], 0.0, s0
    while True:
        rem = extent - cum
        if rem < s * ratio + 1e-12:
            sizes.append(rem)
            return sizes
        sizes.append(s)
        cum += s
        s *= ratio


def grid(refine: int, n_layers: int, n_seg: int = N_SEG, layer_h: float = LAYER_H, rows: int = 2):
    r = refine
    end = [SEG_L / r] * r                                   # 5 mm end strips
    xw = [SEG_L / 2 / r] * (n_seg * 2 * r)                  # 2.5 mm / r in the wall
    x = np.concatenate([[0.0], np.cumsum(end + xw + end)])
    yw = [0.0015 / r] * (4 * r)
    yg = graded(WALL_Y0, 0.0015 / r)
    y = np.concatenate([[0.0], np.cumsum(yg[::-1] + yw + yg)])
    zg = graded(SUB_T, 0.0015 / r)
    zs = np.concatenate([[0.0], np.cumsum(zg[::-1])])
    zw = SUB_T + np.arange(1, rows * r * n_layers + 1) * (layer_h / (rows * r))
    for arr in (x, y, zs):
        arr[-1] = round(arr[-1], 9)
    return x, y, zs, zw


class Mesh:
    def __init__(self, refine: int = 1, n_layers: int = 12, n_seg: int = N_SEG, layer_h: float = LAYER_H,
                 rows: int = 2):
        self.r, self.nl, self.nseg, self.layer_h, self.rows = refine, n_layers, n_seg, layer_h, rows
        x, y, zs, zw = grid(refine, n_layers, n_seg, layer_h, rows)
        self.x, self.y, self.zs, self.zw = x, y, zs, zw
        nx, ny, nzs = len(x), len(y), len(zs)
        # substrate nodes: full grid
        coords, sub_id = [], np.zeros((nx, ny, nzs), int)
        nid = 0
        for k in range(nzs):
            for j in range(ny):
                for i in range(nx):
                    nid += 1
                    sub_id[i, j, k] = nid
                    coords.append((x[i], y[j], zs[k]))
        # wall nodes above the substrate top
        self.ix0 = int(np.argmin(abs(x - WALL_X0)))
        self.ix1 = int(np.argmin(abs(x - (WALL_X0 + n_seg * SEG_L))))
        self.iy0 = int(np.argmin(abs(y - WALL_Y0)))
        self.iy1 = int(np.argmin(abs(y - WALL_Y1)))
        wi, wj = range(self.ix0, self.ix1 + 1), range(self.iy0, self.iy1 + 1)
        wall_id = {}
        for i in wi:
            for j in wj:
                wall_id[(i, j, 0)] = sub_id[i, j, nzs - 1]
        for k in range(1, len(zw) + 1):
            for j in wj:
                for i in wi:
                    nid += 1
                    wall_id[(i, j, k)] = nid
                    coords.append((x[i], y[j], zw[k - 1]))
        self.coords = np.array(coords)
        # elements (x fastest, then y, then z); node order: bottom face CCW, then top face
        conn, info = [], []          # info: (region, i, j, k)
        for k in range(nzs - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    c = [sub_id[i, j, k], sub_id[i + 1, j, k], sub_id[i + 1, j + 1, k], sub_id[i, j + 1, k]]
                    conn.append(c + [sub_id[i, j, k + 1], sub_id[i + 1, j, k + 1],
                                     sub_id[i + 1, j + 1, k + 1], sub_id[i, j + 1, k + 1]])
                    info.append(("S", i, j, k))
        self.n_sub = len(conn)
        for k in range(len(zw)):
            for j in range(self.iy0, self.iy1):
                for i in range(self.ix0, self.ix1):
                    g = lambda a, b, c: wall_id[(a, b, c)]
                    conn.append([g(i, j, k), g(i + 1, j, k), g(i + 1, j + 1, k), g(i, j + 1, k),
                                 g(i, j, k + 1), g(i + 1, j, k + 1), g(i + 1, j + 1, k + 1), g(i, j + 1, k + 1)])
                    info.append(("W", i, j, k))
        self.conn = np.array(conn)
        self.info = info
        self.cent = self.coords[self.conn - 1].mean(axis=1)
        lo, hi = self.coords[self.conn - 1].min(1), self.coords[self.conn - 1].max(1)
        self.vol = np.prod(hi - lo, axis=1)
        self._sets()

    def _sets(self):
        r, nx, ny, nzs = self.r, len(self.x), len(self.y), len(self.zs)
        el = {}
        S = [e for e, t in enumerate(self.info, 1) if t[0] == "S"]
        W = [e for e, t in enumerate(self.info, 1) if t[0] == "W"]
        el["SUB_ALL"], el["WALL_ALL"] = S, W
        inf = self.info
        top = nzs - 2
        under_fp = lambda i, j: self.ix0 <= i < self.ix1 and self.iy0 <= j < self.iy1
        el["SUB_BOT"] = [e for e in S if inf[e - 1][3] == 0]
        el["SUB_X0"] = [e for e in S if inf[e - 1][1] == 0]
        el["SUB_X1"] = [e for e in S if inf[e - 1][1] == nx - 2]
        el["SUB_Y0"] = [e for e in S if inf[e - 1][2] == 0]
        el["SUB_Y1"] = [e for e in S if inf[e - 1][2] == ny - 2]
        el["SUB_TOPFREE"] = [e for e in S if inf[e - 1][3] == top and not under_fp(*inf[e - 1][1:3])]
        el["SUB_FP"] = [e for e in S if inf[e - 1][3] == top and under_fp(*inf[e - 1][1:3])]
        rows = self.rows * r               # element rows per layer
        cols = 2 * r                       # element columns per segment
        seg = lambda e: (inf[e - 1][1] - self.ix0) // cols + 1
        lay = lambda e: inf[e - 1][3] // rows + 1
        for l in range(1, self.nl + 1):
            Wl = [e for e in W if lay(e) == l]
            toprow = [e for e in Wl if inf[e - 1][3] == l * rows - 1]
            if l == 1:
                below = el["SUB_FP"]
            else:
                below = [e for e in W if inf[e - 1][3] == (l - 1) * rows - 1]
            y0 = [e for e in Wl if inf[e - 1][2] == self.iy0]
            y1 = [e for e in Wl if inf[e - 1][2] == self.iy1 - 1]
            el[f"W_X0_L{l:02d}"] = [e for e in Wl if inf[e - 1][1] == self.ix0]
            el[f"W_X1_L{l:02d}"] = [e for e in Wl if inf[e - 1][1] == self.ix1 - 1]
            for s in range(1, self.nseg + 1):
                el[f"W_L{l:02d}_S{s:02d}"] = [e for e in Wl if seg(e) == s]
                el[f"W_TOP_L{l:02d}_C{s:02d}"] = [e for e in toprow if seg(e) <= s]
                el[f"W_Y0_L{l:02d}_C{s:02d}"] = [e for e in y0 if seg(e) <= s]
                el[f"W_Y1_L{l:02d}_C{s:02d}"] = [e for e in y1 if seg(e) <= s]
                if s < self.nseg:
                    segb = (lambda e: (inf[e - 1][1] - self.ix0) // cols + 1)
                    el[f"REMTOP_L{l:02d}_C{s:02d}"] = [e for e in below if segb(e) > s]
        self.elsets = el
        # probes: nearest node to the nominal coordinates
        ns = {}
        xmid = WALL_X0 + self.nseg * SEG_L / 2          # probes at mid-length of the wall
        probes = {f"P_WALL_L{l:02d}": (0.0, 0.0375, SUB_T + l * self.layer_h) for l in range(1, self.nl + 1)}
        probes.update(SUB_PROBES)
        for name, p in probes.items():
            p = (xmid, p[1], p[2])
            ns[name] = [int(np.argmin(((self.coords - np.array(p)) ** 2).sum(1))) + 1]
        self.nsets = ns


# ------------------------------------------------------------------ heat source
def goldak_flux(mesh: Mesh, active: np.ndarray, xc: float, yc: float, zc: float, Q: float,
                g: dict = GOLDAK):
    d = mesh.cent[active] - np.array([xc, yc, zc])
    front = d[:, 0] >= 0
    c = np.where(front, g["cf"], g["cr"])
    f = np.where(front, g["ff"], g["fr"])
    r2 = d[:, 0] ** 2 / c ** 2 + d[:, 1] ** 2 / g["b"] ** 2 + d[:, 2] ** 2 / g["a"] ** 2
    q = 6 * np.sqrt(3) * f * Q / (g["a"] * g["b"] * c * np.pi ** 1.5) * np.exp(-3 * r2)
    keep = r2 <= R2_CUT
    q, ids = q[keep], active[keep]
    q *= Q / (q * mesh.vol[ids]).sum()
    order = np.argsort(ids)
    return ids[order] + 1, q[order]


# ------------------------------------------------------------------ writer
def fmt_ids(ids, per_line=16):
    ids = list(ids)
    return "\n".join(", ".join(str(v) for v in ids[i:i + per_line]) for i in range(0, len(ids), per_line))


def write_deck(path: Path, refine=1, n_layers=12, q_scale=1.0, h=10.0, h_bottom=10.0, emis=0.2,
               final_cooling=3600.0, hfl=True, title_note="", n_seg=N_SEG, goldak_a=None,
               k_liq_factor=1.0, layer_h=LAYER_H, rows=2):
    m = Mesh(refine, n_layers, n_seg, layer_h, rows)
    Q = ETA * U_ARC * I_ARC * q_scale
    g = dict(GOLDAK)
    if goldak_a is not None:
        g["a"] = goldak_a
    P = "WAAM-1."
    out = []
    w = out.append
    w("*Heading")
    w("** WAAM thermal model: ER5356 wall on Al 5083 substrate (Sundaram et al., Materials)")
    w(f"** I={I_ARC} A, U={U_ARC} V (assumed), eta={ETA} (assumed), Q={Q:.1f} W (q_scale={q_scale})")
    w(f"** travel {TRAVEL * 60000:.1f} mm/min, {n_layers} layers x {layer_h * 1000:.2f} mm, "
      f"interpass {INTERPASS:.0f} s, strategy=unidirectional")
    w("** Goldak a={:.1f} b={:.1f} cf={:.1f} cr={:.1f} mm, ff={:g} fr={:g}".format(
        g["a"] * 1e3, g["b"] * 1e3, g["cf"] * 1e3, g["cr"] * 1e3, g["ff"], g["fr"]))
    if n_seg != N_SEG or k_liq_factor != 1.0:
        w(f"** wall segments={n_seg} (length {n_seg * SEG_L * 1e3:.0f} mm), liquid k factor={k_liq_factor:g}")
    if rows != 2:
        w(f"** element rows per layer = {rows * refine}")
    w(f"** mesh refine={refine}, h={h}, h_bottom={h_bottom}, emissivity={emis} {title_note}")
    w("** Units: m, s, kg, J, W, degC.  Generated by scripts/generate_waam_inp.py")
    w("*Preprint, echo=NO, model=NO, history=NO, contact=NO")
    w("**")
    w("*Part, name=WAAM")
    w("*Node")
    for i, (a, b, c) in enumerate(m.coords, 1):
        w(f"{i}, {a:.8e}, {b:.8e}, {c:.8e}")
    w("*Element, type=DC3D8")
    for i, c in enumerate(m.conn, 1):
        w(f"{i}, " + ", ".join(str(v) for v in c))
    w("*Nset, nset=ALLN, generate")
    w(f"1, {len(m.coords)}, 1")
    for k, v in m.nsets.items():
        w(f"*Nset, nset={k}")
        w(fmt_ids(v) + ",")
    w("*Nset, nset=PROBES")
    w(fmt_ids([v[0] for v in m.nsets.values()]))
    for k, v in m.elsets.items():
        if v:
            w(f"*Elset, elset={k}")
            w(fmt_ids(v))
    w("*Solid Section, elset=SUB_ALL, material=AL5083")
    w(",")
    w("*Solid Section, elset=WALL_ALL, material=AL5356")
    w(",")
    w("*End Part")
    w("**")
    w("*Assembly, name=Assembly")
    w("*Instance, name=WAAM-1, part=WAAM")
    w("*End Instance")
    w("*End Assembly")
    for name, d in [("AL5356", MAT["AL5356"]), ("AL5083", MAT["AL5083"])]:
        w("**")
        w(f"*Material, name={name}")
        w("*Conductivity")
        t_liq = d["L"][2]
        out.extend(f" {a * k_liq_factor if b >= t_liq else a:g}, {b}" for a, b in d["k"])
        w("*Density")
        w(f" {d['rho']},")
        w("*Specific Heat")
        out.extend(f" {a}, {b}" for a, b in d["cp"])
        w("*Latent Heat")
        w(" {}, {}, {}".format(*d["L"]))
    w("**")
    w("*Physical Constants, absolute zero=-273.15, stefan boltzmann=5.67e-08")
    w("*Initial Conditions, type=TEMPERATURE")
    w(f"{P}ALLN, {T0:g}")

    sub_faces = [("SUB_X0", 6), ("SUB_X1", 4), ("SUB_Y0", 3), ("SUB_Y1", 5), ("SUB_TOPFREE", 2)]
    el_out = "HFL," if hfl else None

    def bc_block(faces):
        w("*Film, op=NEW")
        for s, f in faces:
            w(f"{P}{s}, F{f}, {T_AMB:g}, {h:g}")
        w(f"{P}SUB_BOT, F1, {T_AMB:g}, {h_bottom:g}")
        w("*Radiate, op=NEW")
        for s, f in faces:
            w(f"{P}{s}, R{f}, {T_AMB:g}, {emis:g}")

    def outputs(freq):
        w(f"*Output, field, frequency={freq}")
        w("*Node Output")
        w("NT,")
        if el_out:
            w("*Element Output, directions=YES")
            w(el_out)
        w("*Output, history, frequency=1")
        w(f"*Node Output, nset={P}PROBES")
        w("NT,")
        w("*End Step")

    ns = n_seg
    C = f"C{ns:02d}"

    def done_layers(upto):
        f = []
        for k in range(1, upto + 1):
            f += [(f"W_Y0_L{k:02d}_{C}", 3), (f"W_Y1_L{k:02d}_{C}", 5), (f"W_X0_L{k:02d}", 6), (f"W_X1_L{k:02d}", 4)]
        return f

    w("** " + "=" * 64)
    w("*Step, name=Remove_wall, nlgeom=NO, inc=100")
    w("*Heat Transfer, end=PERIOD, deltmx=50.")
    w("0.001, 0.001, 1e-08, 0.001")
    w("*Model Change, type=ELEMENT, remove")
    w(f"{P}WALL_ALL,")
    bc_block(sub_faces + [("SUB_FP", 2)])
    outputs(1)

    ix = {}
    active = list(m.elsets["SUB_ALL"])
    dt = SEG_L / TRAVEL
    for l in range(1, n_layers + 1):
        zc = SUB_T + l * layer_h
        for s in range(1, ns + 1):
            active += m.elsets[f"W_L{l:02d}_S{s:02d}"]
            xc = WALL_X0 + (s - 0.5) * SEG_L
            ids, q = goldak_flux(m, np.array(active) - 1, xc, 0.0375, zc, Q, g)
            w("** " + "-" * 64)
            w(f"*Step, name=L{l:02d}_S{s:02d}, nlgeom=NO, inc=1000")
            w("*Heat Transfer, end=PERIOD, deltmx=150.")
            w(f"{dt / 20:.5g}, {dt:.5g}, 1e-08, {dt / 5:.5g}")
            w("*Model Change, type=ELEMENT, add")
            w(f"{P}W_L{l:02d}_S{s:02d},")
            w("*Dflux, op=NEW")
            for e, v in zip(ids, q):
                w(f"{P}{e}, BF, {v:.6e}")
            cur = [(f"W_Y0_L{l:02d}_C{s:02d}", 3), (f"W_Y1_L{l:02d}_C{s:02d}", 5),
                   (f"W_TOP_L{l:02d}_C{s:02d}", 2), (f"W_X0_L{l:02d}", 6)]
            if s == ns:
                cur.append((f"W_X1_L{l:02d}", 4))
            else:
                cur.append((f"REMTOP_L{l:02d}_C{s:02d}", 2))
            bc_block(sub_faces + done_layers(l - 1) + cur)
            outputs(99999)
        last = l == n_layers
        w("** " + "-" * 64)
        if not last:
            w(f"*Step, name=Interpass_L{l:02d}, nlgeom=NO, inc=2000")
            w("*Heat Transfer, end=PERIOD, deltmx=50.")
            w(f"0.01, {INTERPASS:g}, 1e-08, {INTERPASS / 20:g}")
        else:
            w("*Step, name=Final_cooling, nlgeom=NO, inc=2000")
            w("*Heat Transfer, end=PERIOD, deltmx=50.")
            w(f"0.01, {final_cooling:g}, 1e-08, {final_cooling / 20:g}")
        w("*Dflux, op=NEW")
        bc_block(sub_faces + done_layers(l - 1) + [(f"W_Y0_L{l:02d}_{C}", 3), (f"W_Y1_L{l:02d}_{C}", 5),
                                                    (f"W_TOP_L{l:02d}_{C}", 2), (f"W_X0_L{l:02d}", 6),
                                                    (f"W_X1_L{l:02d}", 4)])
        outputs(5 if last else 99999)
    Path(path).write_text("\n".join(out) + "\n")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--q-scale", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--h-bottom", type=float, default=10.0)
    ap.add_argument("--emissivity", type=float, default=0.2)
    ap.add_argument("--final-cooling", type=float, default=3600.0)
    ap.add_argument("--no-hfl", action="store_true", help="omit element heat-flux output (smaller ODB)")
    ap.add_argument("--segments", type=int, default=N_SEG, help="5 mm segments per layer (wall length)")
    ap.add_argument("--goldak-a", type=float, default=None, help="Goldak depth semi-axis a (mm)")
    ap.add_argument("--k-liq-factor", type=float, default=1.0, help="multiplier on liquid conductivity")
    ap.add_argument("--layer-height", type=float, default=LAYER_H * 1e3, help="layer height (mm)")
    ap.add_argument("--rows-per-layer", type=int, default=2, help="element rows per layer (before --refine)")
    a = ap.parse_args()
    m = write_deck(a.out, a.refine, a.layers, a.q_scale, a.h, a.h_bottom, a.emissivity,
                   a.final_cooling, not a.no_hfl, n_seg=a.segments,
                   goldak_a=None if a.goldak_a is None else a.goldak_a * 1e-3, k_liq_factor=a.k_liq_factor,
                   layer_h=a.layer_height * 1e-3, rows=a.rows_per_layer)
    print(f"{a.out}: {len(m.coords)} nodes, {len(m.conn)} elements, probes {m.nsets}")


if __name__ == "__main__":
    main()
