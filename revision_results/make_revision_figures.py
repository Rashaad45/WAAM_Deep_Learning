"""
make_revision_figures.py - every revision figure, drawn from revision_results/*.csv and the out-of-fold
prediction CSVs only.   python revision_results/make_revision_figures.py

Writes revision_results/figures/fig_*.png (300 dpi) and .pdf, and re-exports the five manuscript figures
(make_figures.py, unchanged) into revision_results/figures/paper_figures/.
"""
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
import rev_common as C  # noqa: E402

REV, OUT = C.REV, C.FIG
# Reference palette (dataviz skill, light mode), fixed entity -> slot mapping used in every figure.
COL = {"logreg": "#2a78d6", "rf": "#eb6834", "bilstm": "#1baf7a", "gat": "#eda100", "svm": "#e87ba4",
       "fusion_mlp": "#6250d6", "cnn1d": "#008300", "small_gnn": "#e34948", "gcn": "#52514e"}
MRK = {"logreg": "o", "rf": "s", "bilstm": "^", "gat": "D", "svm": "v", "fusion_mlp": "P", "cnn1d": "X",
       "small_gnn": "h", "gcn": "<"}
INK, INK2, GRID, NULLBAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
                     "lines.linewidth": 1.6})
N = C.NAMES
FOUR = ["logreg", "rf", "bilstm", "gat"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", name)


def ztab(m, cfg="F0"):
    z = C.zone_table(m, cfg)
    return z if z is not None else C.original_zone_table(m)


def panel_tag(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", color=INK, ha="right")


# 1 -------------------------------------------------------------------------------------------
def fig_confusion():
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.4))
    for ax, m, t in zip(axs.flat, FOUR, "abcd"):
        z = ztab(m)
        y, yh = z.y.to_numpy(int), (z.probability.to_numpy() >= 0.5).astype(int)
        cm = np.array([[np.sum((y == a) & (yh == b)) for b in (0, 1)] for a in (0, 1)])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=48 / 2)
        for a in (0, 1):
            for b in (0, 1):
                ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=10,
                        color="white" if cm[a, b] > 14 else INK)
        ax.set_xticks([0, 1], ["Stable", "Unstable"]); ax.set_yticks([0, 1], ["Stable", "Unstable"])
        ax.grid(False)
        ax.set_title(N[m], fontsize=8, color=INK)
        if t in "cd":
            ax.set_xlabel("Predicted (p ≥ 0.5)")
        if t in "ac":
            ax.set_ylabel("Label (τ = 15 dB)")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig_confusion")


# 2 -------------------------------------------------------------------------------------------
def curves(y, p):
    from sklearn.metrics import roc_curve, precision_recall_curve
    f, t, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    return f, t, rc[::-1], pr[::-1]


def fig_roc_pr():
    grid = np.linspace(0, 1, 101)
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.7), sharex=True, sharey="row")
    rows = []
    for j, m in enumerate(FOUR):
        z = ztab(m).sort_values(["trial_id", "zone_id"])
        y, p, tr = z.y.to_numpy(int), z.probability.to_numpy(), z.trial_id.to_numpy()
        f, t, rc, pr = curves(y, p)
        tb, pb = [], []
        for i in C.EV.bootstrap_indices(tr, C.rng0()):
            if 0 < y[i].sum() < len(i):
                ff, tt, rr, pp = curves(y[i], p[i])
                tb.append(np.interp(grid, ff, tt)); pb.append(np.interp(grid, rr, np.maximum.accumulate(pp[::-1])[::-1]))
        tb, pb = np.array(tb), np.array(pb)
        c = COL[m]
        ax = axs[0, j]
        ax.fill_between(grid, np.quantile(tb, .025, 0), np.quantile(tb, .975, 0), color=c, alpha=0.18, lw=0)
        ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=0.8)
        ax.plot(f, t, color=c, drawstyle="steps-post")
        ax.set_title(f"{N[m]}\nAUC {C.auc(y, p):.2f}", fontsize=8)
        ax = axs[1, j]
        ax.fill_between(grid, np.quantile(pb, .025, 0), np.quantile(pb, .975, 0), color=c, alpha=0.18, lw=0)
        ax.axhline(y.mean(), ls=":", color=INK2, lw=0.8)
        ax.plot(rc, pr, color=c, drawstyle="steps-post")
        from sklearn.metrics import average_precision_score
        ax.text(0.04, 0.06, f"AP {average_precision_score(y, p):.2f}", transform=ax.transAxes, color=INK2)
        rows.append(dict(model=m, auc=C.auc(y, p), ap=average_precision_score(y, p), boot_draws_used=len(tb)))
    axs[0, 0].set_ylabel("True-positive rate"); axs[1, 0].set_ylabel("Precision")
    for ax in axs[0]:
        ax.set_xlabel("False-positive rate")
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axs[1]:
        ax.set_xlabel("Recall")
    axs[1, 1].text(0.98, 0.49, "prevalence 0.52", ha="right", va="top", color=INK2, fontsize=6.5,
                   transform=axs[1, 1].get_yaxis_transform())
    for ax in axs.flat:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, "fig_roc_pr")
    pd.DataFrame(rows).to_csv(REV / "fig_roc_pr_values.csv", index=False)


# 3 -------------------------------------------------------------------------------------------
def fig_importance():
    imp = pd.read_csv(REV / "importance.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for r, m in enumerate(("logreg", "rf")):
        for cidx, level in enumerate(("individual", "group")):
            ax = axs[r, cidx]
            d = imp[(imp.model == m) & (imp.level == level)]
            a = d[d.scheme == "across_oof"].set_index("feature_or_group").delta_auc_mean
            w = d[d.scheme == "within_fold"].set_index("feature_or_group").delta_auc_mean
            top = a.sort_values(ascending=False).index[:15 if level == "individual" else len(a)]
            yy = np.arange(len(top))[::-1]
            ax.axvline(0, color=INK2, lw=0.8)
            ax.hlines(yy, 0, a[top], color=COL[m], lw=1.2, alpha=0.5)
            ax.plot(a[top], yy, MRK[m], color=COL[m], ms=5)
            ax.plot(w[top], yy, "|", color=INK, ms=8, mew=1.4)
            ax.set_yticks(yy, [s.replace("_", " ") for s in top], fontsize=6.5)
            ax.set_xlabel("ΔAUC when permuted (pooled OOF, 5 seeds, 20 repeats)")
            ax.set_title(f"{N[m]}: {'top 15 features' if level == 'individual' else 'feature family × modality'}",
                         fontsize=8)
    h = [plt.Line2D([], [], marker="o", color=INK2, ls="none", ms=5, label="permuted across all held-out zones"),
         plt.Line2D([], [], marker="|", color=INK, ls="none", ms=8, mew=1.4, label="permuted within held-out trial")]
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_importance")


def fig_shap():
    try:
        import shap
    except ImportError:
        print("shap not installed; fig_shap skipped"); return
    s = pd.read_csv(REV / "shap_rf_oof.csv")
    x = pd.read_csv(REV / "zone_mean_features.csv")
    names = C.FEATURE_NAMES
    plt.figure()
    shap.summary_plot(s[names].to_numpy(), x[names].to_numpy(), feature_names=[n.replace("_", " ") for n in names],
                      max_display=15, show=False, plot_size=(5.0, 4.6))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("TreeSHAP value (out-of-fold RF, 5 seeds averaged)", fontsize=8)
    save(fig, "fig_shap")


# 4 -------------------------------------------------------------------------------------------
def fig_ablation():
    a = pd.read_csv(REV / "ablation_features.csv")
    cfgs = list(C.CONFIGS)
    short = {"F0": "F0 all", "F1": "F1 −CI", "F2": "F2 −force chatter band", "F3": "F3 statistical only",
             "F4": "F4 spectral only", "F5": "F5 spectral −CI", "F6": "F6 force only", "F7": "F7 accel. only",
             "F8": "F8 accel. −CI"}
    deep = [m for m in ("bilstm", "fusion_mlp", "gat") if (a.model == m).any()]
    dcfgs = [c for c in cfgs if c in set(a[a.model.isin(deep)].config)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.7), gridspec_kw=dict(width_ratios=[len(cfgs), max(len(dcfgs), 1) + 1.2]))
    null = a[a.model.isin(["logreg", "rf", "svm"])]
    for ax, models, cf, title in ((axs[0], ("logreg", "rf", "svm"), cfgs, "Classical models"),
                                  (axs[1], deep, dcfgs, "Deep models (raw-window encoders unchanged)")):
        y = np.arange(len(cf))[::-1]
        ax.axvspan(null.zone_perm_null_q025.mean(), null.zone_perm_null_q975.mean(), color=NULLBAND, alpha=0.6, lw=0,
                   label="95% zone-label permutation null")
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        for k, m in enumerate(models):
            d = a[a.model == m].set_index("config").reindex(cf)
            off = (k - (len(models) - 1) / 2) * 0.25
            ax.errorbar(d.auc, y + off, xerr=[d.auc - d.auc_ci_low, d.auc_ci_high - d.auc], fmt=MRK[m], color=COL[m],
                        ms=4, lw=1.1, capsize=0, label=N[m])
        ax.set_yticks(y, [short[c] for c in cf])
        ax.set(xlabel="Pooled ROC-AUC (trial-bootstrap 95% CI)", xlim=(0, 1))
        ax.set_xticks([0, 0.5, 1] if ax is axs[1] else [0, .2, .4, .6, .8, 1])
        ax.set_title(title, fontsize=8)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[1:] if ax is axs[1] else h, l[1:] if ax is axs[1] else l, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_ablation")


# 5 -------------------------------------------------------------------------------------------
def fig_calibration():
    b = pd.read_csv(REV / "calibration_bins.csv")
    c = pd.read_csv(REV / "calibration.csv")
    keys = list(c[["model", "config"]].itertuples(index=False, name=None))
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.0, 2.2 * nr), sharex=True, sharey=True)
    for ax, (m, cfg) in zip(axs.flat, keys):
        d = b[(b.model == m) & (b.config == cfg)]
        r = c[(c.model == m) & (c.config == cfg)].iloc[0]
        ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=0.8)
        ax.vlines(d.mean_predicted, d.boot_obs_q025, d.boot_obs_q975, color=COL[m], lw=1.0, alpha=0.6)
        ax.plot(d.mean_predicted, d.observed_frequency, "-" + MRK[m], color=COL[m], ms=4)
        ax.set_title(f"{N[m]} ({cfg})\nBrier {r.brier:.3f} · ECE {r.ece_5_equal_count_bins:.3f}\n"
                     f"slope {r.calibration_slope:.2f}", fontsize=6.8)
        ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.set_xticks([0, 0.5, 1])
    for ax in axs.flat[len(keys):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axs[:, 0]:
        ax.set_ylabel("Observed frequency")
    fig.text(0.5, -0.01, "5 equal-count bins (n = 9–10 zones); bars: trial-bootstrap 95% range of observed frequency",
             ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig_calibration")


# 6 -------------------------------------------------------------------------------------------
def fig_levels():
    d = pd.read_csv(REV / "discrimination_levels.csv")
    d["label"] = d.model.map(N) + np.where(d.config == "F1", " (F1, −CI)", "")
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    spec = [("pooled_auc", "o", "#2a78d6", "Pooled (48 zones)"),
            ("within_trial_auc_mean", "s", "#eb6834", "Within-trial mean (8 mixed trials)"),
            ("trial_level_auc_any_unstable", "D", "#1baf7a", "Trial level (≥1 unstable zone)")]
    for k, (col, mk, cc, lab) in enumerate(spec):
        ax.plot(d[col], y + (1 - k) * 0.22, mk, color=cc, ms=4.5, label=lab, ls="none")
    ax.set_yticks(y, d.label)
    ax.set(xlabel="ROC-AUC", xlim=(0, 1.02))
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), fontsize=6.5)
    save(fig, "fig_levels")


def paper_figures():
    out = OUT / "paper_figures"
    out.mkdir(exist_ok=True)
    env = dict(os.environ, WAAM_FIGURES_DIR=str(out))
    subprocess.run([sys.executable, str(C.ROOT / "make_figures.py")], cwd=C.ROOT, env=env, check=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in (fig_confusion, fig_roc_pr, fig_ablation, fig_calibration, fig_levels, fig_importance, fig_shap,
              paper_figures):
        if only and f.__name__ not in only:
            continue
        try:
            f()
        except Exception as e:
            print(f"FAILED {f.__name__}: {e!r}")
