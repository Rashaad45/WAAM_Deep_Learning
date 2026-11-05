"""
draw_architecture.py
====================
Draws the architecture figure of the paper: from the milled wall to the two outputs
of the ST-GAT, in five columns (a) - (e).

Nothing is trained here and no result file is read. The numbers printed on the
figure (zones, windows, feature sizes, heads, alpha, edge weights) are taken from
config.py and graph_data.py, so the picture always matches the code.

How to run (from the project folder)
    python src/draw_architecture.py

Files written (open the .png in Word, upload the .pdf to the journal)
    outputs/figures/fig_architecture.png   300 dpi
    outputs/figures/fig_architecture.pdf   vector, sharp at any size

To change a text or a colour, edit the matching block in draw_figure() below.
Every box is one call of box(x, y, width, height, text, colour).
"""
import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from config import CONFIG, OUTPUT_DIR
from graph_data import TrialGraphBuilder

# Set this to True once measured zone-level Ra is in data/ra_measurements.csv.
# While it is False, the Ra output is drawn dashed with a footnote, because the
# model is then trained on the material-mean Ra of ref. [1], not on measured Ra.
RA_MEASURED = False

# One colour per column: (light background, dark edge / title colour).
COLOURS = {
    "a": ("#eef4fb", "#2471a3"),            # sensing        - blue
    "b": ("#eef7f0", "#1e8449"),            # node inputs    - green
    "c": ("#fdf3e8", "#d35400"),            # graph          - orange
    "d": ("#f3effa", "#6c3483"),            # ST block       - purple
    "e": ("#fbeeee", "#c0392b"),            # outputs        - red
}
KINEMATIC_GREY = "#7f8c8d"                  # same colours as Figure 5 in make_report.py
THERMAL_ORANGE = "#e67e22"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})


# ===========================================================================
# Small drawing helpers (all positions are in "figure units": 0-120 wide, 0-55 high)
# ===========================================================================
def panel(ax, x, width, key, title):
    """Light rounded background of one column, with its letter and title on top."""
    light, dark = COLOURS[key]
    ax.add_patch(FancyBboxPatch((x, 2), width, 51, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=light, edgecolor="none"))
    ax.text(x + 1, 50.8, f"({key})  {title}", fontsize=8.5, fontweight="bold", color=dark,
            ha="left", va="center")


def box(ax, x, y, width, height, text, key, fontsize=7, dashed=False):
    """White rounded box with centred text. (x, y) is the lower-left corner."""
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor="white", edgecolor=COLOURS[key][1], linewidth=1.1,
                                linestyle="--" if dashed else "-"))
    ax.text(x + width / 2, y + height / 2, text, fontsize=fontsize, ha="center", va="center",
            linespacing=1.35)


def arrow(ax, start, end, colour="#333333"):
    """Straight arrow from start (x, y) to end (x, y)."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.2, mutation_scale=9,
                                shrinkA=0, shrinkB=0))


def wiggle(ax, x0, x1, y, amplitude, seed):
    """A short made-up signal trace, only to show 'a sensor signal' in panel (a)."""
    t = np.linspace(0, 1, 500)
    rng = np.random.default_rng(seed)
    s = np.sin(2 * np.pi * 28 * t) + 0.5 * np.sin(2 * np.pi * 61 * t) + 0.3 * rng.normal(size=t.size)
    ax.plot(x0 + t * (x1 - x0), y + amplitude * s / np.abs(s).max(), lw=0.6, color="#34495e")


# ===========================================================================
# The figure
# ===========================================================================
def draw_figure():
    cfg = CONFIG
    n_zones = cfg.process.zone_count
    n_windows = cfg.windows_per_zone
    samples = int(cfg.signal.window_ms / 1000 * cfg.signal.force_fs_hz / cfg.signal.encoder_decimation)
    hidden = cfg.train.hidden_dim
    heads = cfg.train.attention_heads
    alpha = cfg.graph.thermal_diffusivity_m2_s
    adjacency = TrialGraphBuilder(cfg).build_adjacency().numpy()     # [2, N, N]: kinematic, thermal

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set(xlim=(0, 120), ylim=(0, 55), aspect="equal")
    ax.axis("off")

    # ---------------------------------------------------------------- (a) sensing
    panel(ax, 1, 22, "a", "Milling and sensing")
    ax.add_patch(Circle((6.5, 46.5), 1.3, facecolor="#bdc3c7", edgecolor="#555555", lw=0.8))   # the tool
    arrow(ax, (9, 46.5), (15, 46.5))
    ax.text(12, 47.8, f"feed {cfg.process.feed_rate_mm_min:.0f} mm/min", fontsize=6.5, ha="center")
    for i in range(n_zones):                                          # the wall, split into zones
        ax.add_patch(Rectangle((3 + i * 3, 40.5), 3, 3.5, facecolor="#d6e4f0" if i % 2 else "#ebf2f9",
                               edgecolor=COLOURS["a"][1], lw=0.8))
        ax.text(4.5 + i * 3, 42.25, f"Z{i + 1}", fontsize=6.5, ha="center", va="center")
    ax.text(12, 38.6, f"{cfg.process.wall_path_length_mm:.0f} mm thin wall, "
                      f"{n_zones} zones × {cfg.process.zone_length_mm:.0f} mm", fontsize=6.5, ha="center")

    ax.text(12, 35.3, f"Force, 3 axes, {cfg.signal.force_fs_hz / 1000:.0f} kHz", fontsize=6.5, ha="center")
    wiggle(ax, 3, 21, 31.8, 1.6, seed=1)
    ax.add_patch(Rectangle((8, 29.8), 4, 4, facecolor=COLOURS["a"][1], alpha=0.18, edgecolor="none"))
    ax.add_patch(Rectangle((10, 29.8), 4, 4, facecolor="none", edgecolor=COLOURS["a"][1], lw=0.8, ls="--"))
    ax.text(12, 27.6, f"{cfg.signal.window_ms:.0f} ms windows, {cfg.signal.overlap_fraction:.0%} overlap",
            fontsize=6.5, ha="center", style="italic")

    ax.text(12, 24.6, f"Acceleration, 3 axes, {cfg.signal.acceleration_fs_hz / 1000:.0f} kHz",
            fontsize=6.5, ha="center")
    wiggle(ax, 3, 21, 21.2, 1.6, seed=2)

    ax.text(12, 11.5, f"T = {n_windows} windows per zone\n\nWAAM Al 5356 and\nwrought AA 5052 walls",
            fontsize=7, ha="center", va="center", linespacing=1.3)

    # ---------------------------------------------------------------- (b) node inputs
    panel(ax, 25, 24, "b", "Node inputs")
    box(ax, 26, 38, 15.5, 8.5, "Hand-crafted features\n10 × 6 sensor channels\n+ 4 kinematic = 64", "b")
    box(ax, 26, 27.5, 15.5, 8.5, f"Raw force window\n3 × {samples} samples\n1D-CNN encoder → 16", "b")
    box(ax, 26, 17, 15.5, 8.5, f"Raw acceleration window\n3 × {samples} samples\n1D-CNN encoder → 16", "b")
    box(ax, 43.3, 17, 4.8, 29.5, "", "b")
    ax.text(45.7, 31.75, f"Concatenate (96) → Linear → {hidden}", fontsize=7, rotation=90,
            ha="center", va="center")
    for y in (42.25, 31.75, 21.25):                                    # each input goes into the concat box
        arrow(ax, (41.5, y), (43.3, y))
    ax.text(37, 9.5, f"One graph per trial:\n{n_zones} zones × {n_windows} windows × {hidden}",
            fontsize=7, ha="center", va="center", linespacing=1.3)

    # ---------------------------------------------------------------- (c) graph
    panel(ax, 51, 24, "c", "Physics-informed zone graph")
    ax.text(63, 45.3, "Kinematic edge: w = 1 for neighbouring zones", fontsize=6.5, ha="center")
    ax.text(63, 41.6, r"Thermal edge: $w_{ij}=\exp\left(-\dfrac{d_{ij}^{\,2}}{4\,\alpha\,\Delta t_{ij}}\right)$",
            fontsize=7, ha="center", va="center")
    ax.text(63, 37.6, r"$\Delta t_{ij}=d_{ij}/v_f$,   $\alpha=k/\rho c=$" + f"{alpha * 1e5:.1f}"
            + r"$\times10^{-5}$ m$^2$/s", fontsize=6.5, ha="center", va="center")

    node_x = [54 + i * 3.6 for i in range(n_zones)]
    node_y = 23
    for i in range(n_zones):                                           # thermal edges as arcs above
        for j in range(i + 1, n_zones):
            weight = adjacency[1, i, j]
            if weight > 0:
                ax.annotate("", xy=(node_x[j], node_y), xytext=(node_x[i], node_y),
                            arrowprops=dict(arrowstyle="-", color=THERMAL_ORANGE, lw=0.3 + 1.2 * weight,
                                            alpha=0.85, connectionstyle="arc3,rad=-0.45",
                                            shrinkA=4, shrinkB=4))
    for i in range(n_zones - 1):                                       # kinematic edges as straight lines
        if adjacency[0, i, i + 1] > 0:
            ax.plot([node_x[i], node_x[i + 1]], [node_y, node_y], color=KINEMATIC_GREY, lw=2.2, zorder=2)
    for i, x in enumerate(node_x):                                     # the zone nodes
        ax.add_patch(Circle((x, node_y), 1.2, facecolor="white", edgecolor="#2c3e50", lw=1, zorder=3))
        ax.text(x, node_y, f"Z{i + 1}", fontsize=6, ha="center", va="center", zorder=4)

    ax.plot([54, 57.5], [14.5, 14.5], color=KINEMATIC_GREY, lw=2.2)
    ax.text(58.5, 14.5, "kinematic (tool path)", fontsize=6.5, va="center")
    ax.plot([54, 57.5], [11.5, 11.5], color=THERMAL_ORANGE, lw=1.4)
    ax.text(58.5, 11.5, "thermal (heat kernel)", fontsize=6.5, va="center")
    ax.text(63, 6.5, "nodes = zones; line width = edge weight", fontsize=6.2, ha="center",
            style="italic", color="#555555")

    # ---------------------------------------------------------------- (d) spatio-temporal block
    panel(ax, 77, 23, "d", "Spatio-temporal block")
    ax.add_patch(FancyBboxPatch((78, 16), 21, 31.5, boxstyle="round,pad=0,rounding_size=1",
                                facecolor="none", edgecolor=COLOURS["d"][1], lw=0.9, ls=(0, (4, 3))))
    ax.text(98.3, 46.3, "× 2", fontsize=8, fontweight="bold", color=COLOURS["d"][1], ha="right", va="center")
    box(ax, 79, 17.5, 19, 9.5, f"Edge-aware graph attention\n{heads} heads, scores use $w_{{ij}}$\n"
                               "(ST-GCN: graph convolution)", "d")
    box(ax, 79, 29.8, 19, 4.4, "Residual + LayerNorm", "d")
    box(ax, 79, 37, 19, 8, f"Gated temporal convolution\nkernel {cfg.train.temporal_kernel}, "
                           f"along the {n_windows} windows", "d")
    arrow(ax, (88.5, 27), (88.5, 29.8))
    arrow(ax, (88.5, 34.2), (88.5, 37))
    ax.text(88.5, 11.2, r"$e_{ij}=\mathrm{LeakyReLU}(a_l^{\top}Wh_i+a_r^{\top}Wh_j+b^{\top}w_{ij})$",
            fontsize=6, ha="center", va="center")
    ax.text(88.5, 7.6, r"$\alpha_{ij}=\mathrm{softmax}_j\,(e_{ij})$  (space)", fontsize=6.5,
            ha="center", va="center")
    ax.text(88.5, 4.6, "gated conv over windows (time)", fontsize=6.5, ha="center", va="center")

    # ---------------------------------------------------------------- (e) outputs
    panel(ax, 102, 17, "e", "Outputs per zone")
    box(ax, 103, 37, 15, 8, f"Mean over windows\nzone embedding ({hidden})", "e")
    box(ax, 103, 22, 7.2, 9, "Chatter\nprobability\n(sigmoid)", "e", fontsize=6.5)
    box(ax, 110.8, 22, 7.2, 9, "Ra (µm)" if RA_MEASURED else "Ra (µm)*", "e", fontsize=6.5,
        dashed=not RA_MEASURED)
    arrow(ax, (106.6, 37), (106.6, 31))
    arrow(ax, (114.4, 37), (114.4, 31))
    box(ax, 103, 7, 15, 9, r"$\mathcal{L}=\mathrm{BCE}_w+\lambda\,\mathrm{MSE}$" + "\n"
                           f"class-weighted BCE, λ = {cfg.train.regression_weight:g}", "e", fontsize=6.5)
    if not RA_MEASURED:
        ax.text(110.5, 19.3, "* trained on material-mean Ra", fontsize=5.5, ha="center", va="center",
                style="italic", color="#555555")

    # ---------------------------------------------------------------- arrows between the columns
    arrow(ax, (22.3, 21.25), (26, 21.25))            # (a) -> (b)
    arrow(ax, (48.1, 23), (52.6, 23))                # (b) -> (c)
    arrow(ax, (73.4, 23), (79, 23))                  # (c) -> (d)
    arrow(ax, (98, 41), (103, 41))                   # (d) -> (e)
    return fig


def main() -> None:
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = draw_figure()
    for suffix, dpi in (("png", 300), ("pdf", None)):
        path = figure_dir / f"fig_architecture.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
