"""
make_figures.py - every figure of the paper, drawn ONLY from results/*.csv.

    python make_figures.py        -> paper/figures/fig*.png (300 dpi) and .pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
import os
RES = Path(os.environ.get("WAAM_RESULTS_DIR", ROOT / "results"))
OUT = Path(os.environ.get("WAAM_FIGURES_DIR", ROOT / "paper" / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill, light mode): slot 1 blue, slot 2 orange; inks; recessive grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, NULLBAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "svm": "SVM (RBF)", "cnn1d": "1D-CNN",
         "bilstm": "BiLSTM", "fusion_mlp": "Fusion MLP", "small_gnn": "Small GNN", "gcn": "ST-GCN", "gat": "ST-GAT"}
ORDER = list(NAMES)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def fig1_mapping():
    d = pd.read_csv(RES / "trial_mapping.csv").dropna(subset=["measured_rpm"])
    x = d.trial_id.str[-2:].astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].scatter(x, d.nominal_rpm_manifest, s=36, marker="s", facecolors="none", edgecolors=INK2,
                  linewidths=1.2, label="Provisional manifest (nominal)", zorder=3)
    ax[0].scatter(x, d.measured_rpm, s=22, color=BLUE, label="Measured from force spectrum", zorder=4)
    ax[0].set(xlabel="Trial number", ylabel="Spindle speed (rpm)", xticks=x)
    ax[0].tick_params(axis="x", labelsize=6.5, labelrotation=90)
    ax[0].legend(loc="center left", bbox_to_anchor=(0, 0.5))
    for b, g in d.groupby("speed_block"):
        ax[1].plot(g.trial_id.str[-2:].astype(int), g.mean_force_n, color=BLUE if g.measured_rpm.iloc[0] < 1250
                   else ORANGE, lw=2, marker="o", ms=4)
    ax[1].plot([], [], color=BLUE, lw=2, marker="o", ms=4, label="~1031 rpm blocks")
    ax[1].plot([], [], color=ORANGE, lw=2, marker="o", ms=4, label="~1461 rpm blocks")
    ax[1].set(xlabel="Trial number", ylabel="Mean resultant force (N)", xticks=x)
    ax[1].tick_params(axis="x", labelsize=6.5, labelrotation=90)
    ax[1].legend(loc="upper left")
    for a, lab in zip(ax, "ab"):
        a.text(-0.14, 1.02, f"({lab})", transform=a.transAxes, fontweight="bold", color=INK)
    save(fig, "fig1_mapping")


def fig2_prominence():
    d = pd.read_csv(RES / "zone_label_details.csv")
    thr = pd.read_csv(RES / "label_counts.csv")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    on = d.on_harmonic.astype(bool)
    ax.scatter(x[~on], d.prominence_db[~on], s=18, color=BLUE, label="Peak not on a spindle harmonic", zorder=3)
    ax.scatter(x[on], d.prominence_db[on], s=22, marker="x", color=ORANGE, label="Peak on a spindle harmonic (±2 Hz)",
               zorder=3)
    for t in thr.threshold_db:
        n = int(thr.loc[thr.threshold_db == t, "n_positive"].iloc[0])
        ax.axhline(t, color=INK2, lw=0.8, ls="--")
        ax.text(1.005, t, f"{t:.0f} dB: {n} unstable", transform=ax.get_yaxis_transform(), ha="left",
                va="center", color=INK2, fontsize=7)
    starts = d.groupby("trial_id").cumcount() == 0
    ax.set_xticks(x[starts]); ax.set_xticklabels(d.trial_id[starts].str[-2:])
    ax.set(xlabel="Trial (3 zones each)", ylabel="Peak prominence over band median (dB)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "fig2_prominence")


def fig3_benchmark():
    m = pd.read_csv(RES / "metrics.csv")
    main = m[(m.experiment == "main_15db") & (m.variant == "full")].set_index("model").reindex(ORDER)
    al = m[(m.experiment == "alignment_15db") & (m.variant == "full")].set_index("model").reindex(ORDER)
    y = np.arange(len(ORDER))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    lo, hi = main.null_auc_q025.mean(), main.null_auc_q975.mean()
    ax.axvspan(lo, hi, color=NULLBAND, alpha=0.6, lw=0, label="95% label-permutation null")
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.errorbar(main.auc, y + 0.12, xerr=[main.auc - main.auc_ci_low, main.auc_ci_high - main.auc], fmt="o",
                color=BLUE, ms=5, lw=1.4, capsize=0, label="All 16 trials (trial-bootstrap 95% CI)")
    ax.errorbar(al.auc, y - 0.12, xerr=[al.auc - al.auc_ci_low, al.auc_ci_high - al.auc], fmt="s", color=ORANGE,
                ms=4, lw=1.2, capsize=0, label="Without trials 04 and 06 (weak alignment)")
    ax.set_yticks(y); ax.set_yticklabels([NAMES[k] for k in ORDER])
    ax.set(xlabel="ROC-AUC (15 dB labels, LOTO, 5 seeds averaged)", xlim=(0, 1))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    save(fig, "fig3_benchmark")


def fig4_paired():
    t = pd.read_csv(RES / "paired_tests.csv")
    abl = t[t.experiment_b == "ablation_15db"].copy()
    abl["label"] = abl.model_a.map(NAMES) + ": full − " + abl.variant_b.str.replace("_", " ")
    vs = t[(t.experiment_a == "main_15db") & (t.model_b == "logreg") & (t.experiment_b == "main_15db")].copy()
    vs["label"] = vs.model_a.map(NAMES) + " − Logistic regression"
    gr = t[(t.model_b == "fusion_mlp")].copy()
    gr["label"] = gr.model_a.map(NAMES) + " − Fusion MLP"
    d = pd.concat([vs, gr, abl])
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.22 * len(d) + 0.8))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.errorbar(d.delta_auc, y, xerr=[d.delta_auc - d.delta_ci_low, d.delta_ci_high - d.delta_auc], fmt="o",
                color=BLUE, ms=4, lw=1.2, capsize=0)
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.text(1.02, yy, f"p = {r.perm_p:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(d.label)
    ax.set(xlabel="ΔAUC (trial-bootstrap 95% CI; p: paired trial-level permutation)")
    save(fig, "fig4_paired")


def fig5_loss():
    c = pd.read_csv(RES / "training_loss_curves.csv")
    c = c[(c.experiment == "main_15db") & (c.variant == "full")]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    cols = {"cnn1d": "#6250d6", "bilstm": AQUA, "fusion_mlp": "#eda100", "small_gnn": "#e87ba4",
            "gcn": ORANGE, "gat": BLUE}
    for k in ["cnn1d", "bilstm", "fusion_mlp", "small_gnn", "gcn", "gat"]:
        g = c[c.model == k]
        ax.plot(g.epoch, g.train_loss, color=cols[k], lw=2, label=NAMES[k])
    ax.set(xlabel="Epoch", ylabel="Training loss (weighted BCE)", yscale="log")
    ax.legend(ncol=2)
    save(fig, "fig5_training_loss")


if __name__ == "__main__":
    for f in (fig1_mapping, fig2_prominence, fig3_benchmark, fig4_paired, fig5_loss):
        f()
        print("wrote", f.__name__)
