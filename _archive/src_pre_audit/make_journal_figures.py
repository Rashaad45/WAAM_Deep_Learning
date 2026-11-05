"""
make_journal_figures.py
=======================
Draws the journal figures of the methods and the results, one PNG + one PDF each,
into outputs/figures/. (The architecture figure has its own file: draw_architecture.py.)

    Methods (need only Step 1, "prepare"):
      fig_labelling_rule     force PSD of a stable and an unstable zone + the 20 dB chatter rule
      fig_chatter_map        stable / unstable label of every zone of every trial
      fig_edge_weights       kinematic and thermal edge-weight matrices of the zone graph
    Results (need Step 2, "train" and "ablations"):
      fig_roc_pr             ROC and precision-recall curves of every classifier
      fig_model_comparison   F1 and ROC-AUC with 95 % CI for every model
      fig_ablation           F1 and ROC-AUC of every ST-GAT ablation
      fig_attention          GAT attention between zones, per material
      fig_attribution        share of force / acceleration / kinematic inputs (integrated gradients)

Every figure is one function below. If its input file does not exist yet, the
figure is skipped with a message, and the others are still drawn.

How to run (from the project folder)
    python src/make_journal_figures.py                       all figures (~1 min, reads 2 raw files)
    python src/make_journal_figures.py --only fig_ablation   one figure
    python src/make_journal_figures.py --skip-raw            all except fig_labelling_rule (no raw files)
"""
import argparse

import matplotlib
matplotlib.use("Agg")                       # draw to files only, no window
import matplotlib.pyplot as plt             # noqa: E402  (must come after matplotlib.use)
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
from matplotlib.colors import ListedColormap                                        # noqa: E402
from scipy import signal                                                            # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve   # noqa: E402

from config import CONFIG, CACHE_DIR, DATA_DIR, OUTPUT_DIR, PROJECT_DIR             # noqa: E402
from graph_data import TrialGraphBuilder                                            # noqa: E402

FIGURE_DIR = OUTPUT_DIR / "figures"

# Same colours as make_report.py and draw_architecture.py, so all figures match.
MATERIAL_COLOURS = {"WAAM": "#c0392b", "WROUGHT": "#2471a3"}
MATERIAL_NAMES = {"WAAM": "WAAM", "WROUGHT": "Wrought"}
KINEMATIC_GREY = "#7f8c8d"
THERMAL_ORANGE = "#e67e22"

# Model name in outputs/*.csv -> (name on the figure, colour). Our models are drawn thicker.
MODELS = {
    "Logistic regression": ("Logistic regression", "#95a5a6"),
    "Random forest": ("Random forest", "#7f8c8d"),
    "SVM (RBF)": ("SVM (RBF)", "#bdc3c7"),
    "cnn1d": ("1D-CNN", "#a569bd"),
    "bilstm": ("BiLSTM", "#5dade2"),
    "fusion_mlp": ("Fusion MLP (no graph)", "#27ae60"),
    "gcn": ("ST-GCN (ours)", "#e67e22"),
    "gat": ("ST-GAT (ours)", "#c0392b"),
}
OUR_MODELS = ("gcn", "gat")

# Ablation name in outputs/ablation.csv -> name on the figure.
ABLATIONS = {
    "full": "Full ST-GAT",
    "without_force_features": "− force inputs",
    "without_acceleration_features": "− acceleration inputs",
    "without_thermal_edges": "− thermal edges",
    "without_kinematic_edges": "− kinematic edges",
    "no_spatial_edges": "− all graph edges",
    "without_temporal_module": "− temporal module",
    "material_flag": "+ material flag",
    "classification_only": "chatter loss only",
}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})


# ===========================================================================
# Small helpers
# ===========================================================================
def save(fig, name: str) -> None:
    """Write <name>.png (300 dpi) and <name>.pdf (vector) into outputs/figures/."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def read_output(file_name: str) -> pd.DataFrame | None:
    """Read outputs/<file_name>, or return None (and say so) when it does not exist yet."""
    path = OUTPUT_DIR / file_name
    if not path.exists():
        print(f"  skipped: {path.name} not found - run the training step first")
        return None
    return pd.read_csv(path)


def zone_predictions() -> pd.DataFrame | None:
    """One row per (model, zone) of the main run: seeds averaged, full variant only."""
    pred = read_output("predictions_oof.csv")
    if pred is None:
        return None
    pred = pred[pred.variant == "full"]
    return pred.groupby(["model", "trial_id", "material_type", "zone_id"], as_index=False).agg(
        y_class=("y_class", "first"), probability=("probability", "mean"))


def metric_with_ci(ax, labels, values, low, high, colours, xlabel):
    """Horizontal dot plot with 95 % CI whiskers (one row per label, first label on top)."""
    y = np.arange(len(labels))[::-1]
    for yi, v, lo, hi, c in zip(y, values, low, high, colours):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.5)
        ax.plot(v, yi, "o", color=c, markersize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#eeeeee")


# ===========================================================================
# Methods figures
# ===========================================================================
# Zones shown in fig_labelling_rule: (trial, zone index 0-5, panel title).
# trial_06 (wrought) and trial_16 (WAAM) were cut with the SAME speed and depth (1400 rpm, 0.5 mm).
LABEL_EXAMPLES = [("trial_06", 3, "Wrought: below threshold"),
                  ("trial_16", 1, "WAAM: peak on a spindle harmonic"),
                  ("trial_16", 5, "WAAM: unstable (chatter)")]


def fig_labelling_rule():
    """Force PSD of the zones in LABEL_EXAMPLES, with the chatter rule drawn on top."""
    from prepare_dataset import RawFileReader              # only needed here (reads the raw CSV)
    from signal_features import ZoneChatterLabeler

    manifest = pd.read_csv(DATA_DIR / "trial_manifest.csv").set_index("trial_id")
    summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv").set_index("trial_id")
    fs = CONFIG.signal.force_fs_hz
    labeler = ZoneChatterLabeler(CONFIG.signal, CONFIG.labels)
    low, high = CONFIG.signal.chatter_band_hz

    forces = {}                                             # read each raw file only once
    fig, axes = plt.subplots(1, len(LABEL_EXAMPLES), figsize=(7.4, 2.8), sharey=True)
    for ax, (trial, zone, title) in zip(axes, LABEL_EXAMPLES):
        if trial not in forces:
            print(f"  reading the raw force file of {trial} ...")
            forces[trial] = RawFileReader.read_force(PROJECT_DIR / manifest.loc[trial, "force_csv"])
        force = forces[trial]
        rpm = float(summary.loc[trial, "spindle_speed_rpm"])
        zone_length_s = (summary.loc[trial, "cut_end_s"] - summary.loc[trial, "cut_start_s"]) / CONFIG.process.zone_count
        start = np.load(CACHE_DIR / f"{trial}.npz")["zone_start_s"][zone]
        zone_force = force[int(start * fs): int((start + zone_length_s) * fs)].T     # [3, samples]
        label = labeler.label_zone(zone_force, fs, rpm)

        # The same PSD as the labelling rule: long Welch segment, three axes added together.
        centred = zone_force - zone_force.mean(axis=1, keepdims=True)
        freq, psd = signal.welch(centred, fs=fs, nperseg=min(CONFIG.labels.label_nperseg, centred.shape[1]), axis=1)
        psd_db = 10 * np.log10(psd.sum(axis=0) + 1e-30)
        in_band = (freq >= low) & (freq <= high)
        median_db = np.median(psd_db[in_band])
        shown = (freq >= 1000) & (freq <= 7000)

        ax.axvspan(low, high, color="#f6ddcc", alpha=0.6, lw=0, label="chatter band")
        ax.plot(freq[shown], psd_db[shown], color="#34495e", lw=0.5)
        ax.hlines(median_db, low, high, colors="#555555", linestyles=":", lw=1, label="band median")
        ax.hlines(median_db + CONFIG.labels.peak_prominence_db, low, high, colors=MATERIAL_COLOURS["WAAM"],
                  linestyles="--", lw=1, label=f"median + {CONFIG.labels.peak_prominence_db:.0f} dB")
        peak_db = median_db + label["peak_prominence_db"]
        ax.plot(label["peak_hz"], peak_db, "v", color=MATERIAL_COLOURS["WAAM"], markersize=6)
        verdict = "unstable" if label["chatter_label"] else "stable"
        harmonic = "harmonic" if label["peak_is_spindle_harmonic"] else "non-harmonic"
        ax.set_title(f"{title}\n{trial.replace('trial_', 'T')}, zone {zone + 1}", fontsize=8)
        ax.text(0.03, 0.04, f"peak {label['peak_hz']:.0f} Hz, {label['peak_prominence_db']:.1f} dB\n"
                            f"{harmonic} → {verdict}", transform=ax.transAxes, fontsize=6.5)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Force PSD, 3 axes summed (dB)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_labelling_rule")


def fig_chatter_map():
    """Stable / unstable label of every zone; the number in each cell is the peak prominence (dB)."""
    labels = pd.read_csv(DATA_DIR / "zone_labels.csv")
    # Wrought trials first, then WAAM, each sorted by trial number.
    order = (labels.drop_duplicates("trial_id").sort_values(["material_type", "trial_id"], ascending=[False, True]))
    trials = order.trial_id.tolist()
    grid = labels.pivot(index="trial_id", columns="zone_id", values="chatter_label").loc[trials]
    prominence = labels.pivot(index="trial_id", columns="zone_id", values="label_peak_prominence_db").loc[trials]

    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    ax.imshow(grid.to_numpy(), cmap=ListedColormap(["#eaf2f8", MATERIAL_COLOURS["WAAM"]]), aspect="auto")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            unstable = grid.iat[r, c] == 1
            ax.text(c, r, f"{prominence.iat[r, c]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if unstable else "#555555", fontweight="bold" if unstable else "normal")
    n_wrought = int((order.material_type == "WROUGHT").sum())
    ax.axhline(n_wrought - 0.5, color="black", lw=1.2)              # line between the two materials
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"Z{z + 1}" for z in grid.columns])
    ax.set_yticks(range(len(trials)))
    ax.set_xlabel("Zone along the wall (tool path →)")
    ax.set_yticklabels([f"{t.replace('trial_', 'T')} ({'WAAM' if m == 'WAAM' else 'wrought'})"
                        for t, m in zip(trials, order.material_type)])
    ax.set_title(f"{int(grid.to_numpy().sum())} of {grid.size} zones unstable "
                 f"(red, ≥ {CONFIG.labels.peak_prominence_db:.0f} dB)\nnumber = peak prominence (dB)", fontsize=7.5)
    ax.spines[:].set_visible(False)
    save(fig, "fig_chatter_map")


def fig_edge_weights():
    """The two 6 x 6 edge-weight matrices used by every trial graph."""
    adjacency = TrialGraphBuilder(CONFIG).build_adjacency().numpy()
    n = adjacency.shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9))
    titles = ("Kinematic edges (tool path)", "Thermal edges (heat kernel)")
    for ax, matrix, title, cmap in zip(axes, adjacency, titles, ("Greys", "Oranges")):
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if matrix[i, j] > 0.6 else "black")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"Z{i + 1}" for i in range(n)])
        ax.set_yticklabels([f"Z{i + 1}" for i in range(n)])
        ax.set_title(title)
        ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.8, label="edge weight $w_{ij}$")
    alpha = CONFIG.graph.thermal_diffusivity_m2_s
    fig.text(0.5, -0.02, f"Thermal: $w_{{ij}}=\\exp(-d_{{ij}}\\,v_f/4\\alpha)$, α = {alpha:.2e} m²/s, "
                         f"v_f = {CONFIG.process.feed_rate_mm_min:.0f} mm/min", ha="center", fontsize=7)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)       # colorbar + tight_layout do not mix: save directly
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"fig_edge_weights.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_edge_weights.png / .pdf")


# ===========================================================================
# Results figures
# ===========================================================================
def fig_roc_pr():
    """ROC (left) and precision-recall (right) curves of every classifier, out-of-fold."""
    zones = zone_predictions()
    if zones is None:
        return
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    prevalence = None
    for model, (name, colour) in MODELS.items():
        sub = zones[(zones.model == model) & zones.probability.notna()]
        if sub.empty or sub.y_class.nunique() < 2:          # a curve needs both classes
            continue
        y, p = sub.y_class.astype(int), sub.probability
        width = 2.0 if model in OUR_MODELS else 1.0
        fpr, tpr, _ = roc_curve(y, p)
        auc = np.trapezoid(tpr, fpr)
        ax_roc.plot(fpr, tpr, color=colour, lw=width)
        precision, recall, _ = precision_recall_curve(y, p)
        ax_pr.step(recall, precision, where="post", color=colour, lw=width,
                   label=f"{name}\nAUC {auc:.2f} · AP {average_precision_score(y, p):.2f}")
        prevalence = y.mean()
    ax_roc.plot([0, 1], [0, 1], "k:", lw=1)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    if prevalence is not None:
        ax_pr.axhline(prevalence, color="k", ls=":", lw=1)
        ax_pr.text(0.98, prevalence + 0.02, f"chance = {prevalence:.2f}", ha="right", fontsize=6.5)
    ax_pr.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.05), title="Precision-recall")
    ax_pr.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.02), fontsize=6, labelspacing=0.6)
    save(fig, "fig_roc_pr")


def fig_model_comparison():
    """F1 and ROC-AUC of every chatter classifier with 95 % trial-bootstrap CI."""
    table = read_output("metrics_main.csv")
    if table is None:
        return
    table = table[table.model.isin(MODELS) & table.f1.notna()].set_index("model")
    models = [m for m in MODELS if m in table.index]
    labels = [MODELS[m][0] for m in models]
    colours = [MODELS[m][1] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, metric, name in zip(axes, ("f1", "roc_auc"), ("F1 score (unstable class)", "ROC-AUC")):
        t = table.loc[models]
        metric_with_ci(ax, labels, t[metric], t[f"{metric}_ci_low"], t[f"{metric}_ci_high"], colours, name)
        ax.set_xlim(0, 1)
    axes[1].axvline(0.5, color="k", ls=":", lw=1)              # chance level of the AUC
    save(fig, "fig_model_comparison")


def fig_ablation():
    """F1 and ROC-AUC of each ST-GAT ablation; the dashed line is the full model."""
    table = read_output("ablation.csv")
    if table is None:
        return
    table = table[table.variant.isin(ABLATIONS) & table.f1.notna()].set_index("variant")
    variants = [v for v in ABLATIONS if v in table.index]
    labels = [ABLATIONS[v] for v in variants]
    colours = [MATERIAL_COLOURS["WAAM"] if v == "full" else "#566573" for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, metric, name in zip(axes, ("f1", "roc_auc"), ("F1 score (unstable class)", "ROC-AUC")):
        t = table.loc[variants]
        metric_with_ci(ax, labels, t[metric], t[f"{metric}_ci_low"], t[f"{metric}_ci_high"], colours, name)
        if "full" in table.index:
            ax.axvline(table.loc["full", metric], color=MATERIAL_COLOURS["WAAM"], ls="--", lw=1)
        ax.set_xlim(0, 1)
    save(fig, "fig_ablation")


def fig_attention():
    """Mean GAT attention between zones, per material (needs attention_per_trial.csv)."""
    n = CONFIG.process.zone_count
    uniform = 1 / n                                           # what attention looks like if it learns nothing
    per_trial = read_output("attention_per_trial.csv") if (OUTPUT_DIR / "attention_per_trial.csv").exists() else None

    if per_trial is None:
        # Older training run: only the mean per edge type exists -> bar chart.
        means = read_output("attention.csv")
        if means is None:
            return
        print("  note: attention_per_trial.csv not found, drawing bars; rerun 'train' for the 6 x 6 matrices")
        means = means.set_index("material_type")
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        x = np.arange(len(means))
        ax.bar(x - 0.2, means.kinematic_attention, 0.4, color=KINEMATIC_GREY, label="kinematic edges")
        ax.bar(x + 0.2, means.thermal_attention, 0.4, color=THERMAL_ORANGE, label="thermal edges")
        ax.axhline(uniform, color="k", ls=":", lw=1)
        ax.text(len(means) - 0.5, uniform * 1.03, f"uniform = 1/{n}", ha="right", fontsize=6.5)
        ax.set(xticks=x, xticklabels=[MATERIAL_NAMES[m] for m in means.index], ylabel="Mean attention coefficient")
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        save(fig, "fig_attention")
        return

    columns = [f"a_{i}_{j}" for i in range(n) for j in range(n)]
    materials = sorted(per_trial.material_type.unique(), reverse=True)    # WROUGHT, WAAM
    fig, axes = plt.subplots(1, len(materials), figsize=(3.0 * len(materials) + 0.6, 2.9))
    axes = np.atleast_1d(axes)
    for ax, material in zip(axes, materials):
        matrix = per_trial[per_trial.material_type == material][columns].mean().to_numpy().reshape(n, n)
        image = ax.imshow(matrix, cmap="Purples", vmin=0, vmax=max(0.4, matrix.max()))
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                        color="white" if matrix[i, j] > 0.3 else "black")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"Z{i + 1}" for i in range(n)])
        ax.set_yticklabels([f"Z{i + 1}" for i in range(n)])
        ax.set(xlabel="attends to zone j", ylabel="zone i", title=MATERIAL_NAMES[material])
        ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.8, label=f"attention $\\alpha_{{ij}}$ (uniform = {uniform:.2f})")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)       # colorbar + tight_layout do not mix: save directly
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"fig_attention.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_attention.png / .pdf")


def fig_attribution():
    """Integrated-gradient share of every input group in the Ra prediction, per material."""
    share = read_output("attribution.csv")
    if share is None:
        return
    table = share.pivot(index="material_type", columns="modality", values="share_percent")
    groups = [("Force", "#2471a3"), ("Acceleration", "#27ae60"), ("Kinematic / geometric", KINEMATIC_GREY)]
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for name, colour in groups:
        if name not in table:
            continue
        values = table[name].to_numpy()
        ax.barh(y, values, left=left, color=colour, label=name, height=0.55)
        for yi, l, v in zip(y, left, values):
            ax.text(l + v / 2, yi, f"{v:.0f} %", ha="center", va="center", fontsize=7, color="white")
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([MATERIAL_NAMES[m] for m in table.index])
    ax.set(xlim=(0, 100), xlabel="Share of attribution (%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, "fig_attribution")


# ===========================================================================
# Run
# ===========================================================================
FIGURES = {f.__name__: f for f in (fig_labelling_rule, fig_chatter_map, fig_edge_weights, fig_roc_pr,
                                   fig_model_comparison, fig_ablation, fig_attention, fig_attribution)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), help="draw only these figures")
    parser.add_argument("--skip-raw", action="store_true", help="skip fig_labelling_rule (reads raw CSV files)")
    args = parser.parse_args()

    names = args.only or list(FIGURES)
    if args.skip_raw and "fig_labelling_rule" in names:
        names.remove("fig_labelling_rule")
    for name in names:
        print(name)
        FIGURES[name]()
    print(f"\nFigures are in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
