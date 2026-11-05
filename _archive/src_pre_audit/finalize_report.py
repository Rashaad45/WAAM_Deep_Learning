"""
finalize_report.py   -   STEP 4 of the pipeline: the submission version
=========================================================================
Turns the auto-filled draft into the version to submit.

  1. Regenerates every figure from the CURRENT results in outputs/
     (make_report.py, make_journal_figures.py, draw_architecture.py) and draws a
     journal-style parity plot. Older figures were drawn from an earlier training run.
  2. Checks that the results still support the written narrative. If a rerun changed
     the direction of a finding, it stops and tells you which sentence to revisit.
  3. Edits the document:
       * inserts the 9 figures with numbered captions at the right places, replacing the
         old Figure 2, which showed a thermal 2D-CNN branch that was never used;
       * rewrites the abstract, the Results text, the RQ answers and the Conclusions so that
         they match the numbers. The old abstract claimed a gain for the graph model even
         though the GAT was significantly worse than the fusion MLP;
       * removes the "Note to authors" boxes, the promised Supplementary Tables S1/S2 that were
         never produced, and the claim of a hyperparameter search that never happened;
       * fills author details from report/author_info.json (created on the first run).
  4. Writes report/WAAM_STGNN_Materials_FINAL.docx and report/final_checklist.txt, and
     prints READY or NOT READY.

Run from the project folder:
    python src/finalize_report.py              full run (about 2 min)
    python src/finalize_report.py --skip-regen reuse the figures already in outputs/figures
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import docx
import matplotlib
import numpy as np
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm
from docx.text.paragraph import Paragraph

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REP = ROOT / "report"
FILLED = REP / "WAAM_STGNN_Materials_filled.docx"
FINAL = REP / "WAAM_STGNN_Materials_FINAL.docx"
AUTHOR_FILE = REP / "author_info.json"
CHECKLIST = REP / "final_checklist.txt"

sys.path.insert(0, str(SRC))
from config import CACHE_DIR, CONFIG  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

AUTHOR_TEMPLATE = {
    "_help": "Fill in what you have, save, and rerun: python src/finalize_report.py --skip-regen. "
             "Leave a value empty to keep the placeholder (the checklist will list it).",
    "title": "",
    "authors_line": "",
    "affiliations": "",
    "correspondence": "",
    "accelerometer": "",
    "ir_camera": "",
    "repository": "",
    "author_contributions": "",
    "funding": "",
    "acknowledgments": "",
    "ref1_article_number": "",
}


# ===========================================================================
# 1. Regenerate figures
# ===========================================================================
def run_step(script: str, label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"\nFAILED at: {label}. Fix the error above, then rerun.")


def journal_style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})


def draw_parity(R) -> None:
    """Journal-style parity plot of the ST-GAT (replaces fig3_parity_gat.png)."""
    journal_style()
    z = R.gat_zones
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    rng = np.random.default_rng(0)
    for material, colour, name in (("WROUGHT", "#2471a3", "Wrought"), ("WAAM", "#c0392b", "WAAM")):
        s = z[z.material_type == material]
        ax.scatter(s.y + rng.normal(0, 0.006, len(s)), s.p, s=14, alpha=0.75, color=colour,
                   edgecolors="none", label=f"{name} (n = {len(s)} zones)")
    lo = min(z.y.min(), z.p.min()) - 0.05
    hi = max(z.y.max(), z.p.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="perfect prediction")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Target Ra (µm, specimen mean of [1])",
           ylabel="Predicted Ra (µm)")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.97, 0.04, f"RMSE = {R.v('gat', 'rmse_um'):.3f} µm", transform=ax.transAxes, ha="right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"fig_parity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_parity.png / .pdf")


# ===========================================================================
# 2. Results
# ===========================================================================
class Results:
    def __init__(self):
        self.m = pd.read_csv(OUT / "metrics_main.csv").set_index("model")
        self.a = pd.read_csv(OUT / "ablation.csv").set_index("variant")
        self.t = pd.read_csv(OUT / "stat_tests.csv").set_index("test")
        self.att = pd.read_csv(OUT / "attention.csv").set_index("material_type")
        self.attr = pd.read_csv(OUT / "attribution.csv")
        self.lat = json.loads((OUT / "latency.json").read_text())
        self.labels = pd.read_csv(ROOT / "data" / "zone_labels.csv")
        self.summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv").set_index("trial_id")
        pred = pd.read_csv(OUT / "predictions_oof.csv")
        self.gat_zones = (pred[(pred.variant == "full") & (pred.model == "gat")]
                          .groupby(["trial_id", "material_type", "zone_id"], as_index=False)
                          .agg(y=("y_reg", "first"), p=("ra_pred", "mean")))
        self.n_trials = self.labels.trial_id.nunique()
        self.n_zones = len(self.labels)
        self.n_unstable = int(self.labels.chatter_label.sum())

    def v(self, model, metric):
        return float(self.m.loc[model, metric])

    def av(self, variant, metric):
        return float(self.a.loc[variant, metric])

    @staticmethod
    def _ci(table, key, metric, d):
        f = f"{{:.{d}f}}"
        val, lo, hi = table.loc[key, metric], table.loc[key, f"{metric}_ci_low"], table.loc[key, f"{metric}_ci_high"]
        return f"{f.format(val)} ({f.format(lo)}–{f.format(hi)})" if pd.notna(lo) else f.format(val)

    def ci(self, model, metric, d=2):
        return self._ci(self.m, model, metric, d)

    def p(self, test):
        return float(self.t.loc[test, "p_value"])

    def d_auc(self, variant):
        return self.av(variant, "roc_auc") - self.av("full", "roc_auc")

    def d_f1(self, variant):
        return self.av(variant, "f1") - self.av("full", "f1")

    def d_rmse_pct(self, variant):
        full = self.av("full", "rmse_um")
        return (self.av(variant, "rmse_um") - full) / full * 100


def narrative_checks(R):
    """Every finding the rewritten text relies on. All must hold, or the text is wrong."""
    aucs = R.m.roc_auc.dropna()
    regs = R.m.rmse_um.dropna().drop("Predict training mean (reference)", errors="ignore")
    kin = R.attr[R.attr.modality.str.startswith("Kinematic")].share_percent
    modal = ["without_force_features", "without_acceleration_features"]
    return [
        ("ST-GAT AUC is below the fusion MLP AUC", R.v("gat", "roc_auc") < R.v("fusion_mlp", "roc_auc")),
        ("DeLong test GAT vs fusion MLP is significant (p < 0.05)", R.p("DeLong AUC") < 0.05),
        ("Logistic regression has the highest AUC", aucs.idxmax() == "Logistic regression"),
        ("ST-GAT has the lowest Ra RMSE", regs.idxmin() == "gat"),
        ("Removing all spatial edges does not lower AUC", R.d_auc("no_spatial_edges") >= -0.02),
        ("Removing force lowers AUC the most of the modality ablations",
         min(modal, key=lambda v: R.av(v, "roc_auc")) == "without_force_features" and R.d_auc(modal[0]) < 0),
        ("Removing acceleration does not lower AUC", R.d_auc("without_acceleration_features") >= -0.02),
        ("Kinematic/geometric attribution share is below 10%", bool((kin < 10).all())),
        ("GAT attention is near uniform (0.15–0.19)",
         bool(((R.att[["kinematic_attention", "thermal_attention"]] > 0.15)
               & (R.att[["kinematic_attention", "thermal_attention"]] < 0.19)).all().all())),
    ]


# ===========================================================================
# 3. Word helpers
# ===========================================================================
def text_of(p):
    return "".join(r.text for r in p.runs)


def has_math(p):
    return p._p.find(f".//{M_NS}oMath") is not None


def find(doc, needle, startswith=False):
    for p in doc.paragraphs:
        t = p.text.strip()
        if (t.startswith(needle) if startswith else needle in t):
            return p
    raise KeyError(f"Paragraph not found: {needle!r}")


def find_opt(doc, needle, startswith=False):
    try:
        return find(doc, needle, startswith)
    except KeyError:
        return None


def set_text(p, text, keep_label=False):
    """Replace a paragraph's text, keeping the first run's formatting (and a bold label run)."""
    if has_math(p):
        raise RuntimeError(f"Refusing to rewrite a paragraph with equations: {p.text[:60]!r}")
    runs = p.runs
    first = 1 if keep_label and len(runs) > 1 else 0
    runs[first].text = (" " if keep_label else "") + text
    for r in runs[first + 1:]:
        r._r.getparent().remove(r._r)


def replace_in(p, old, new):
    """Replace text even when Word split it over several runs. Returns True if found."""
    runs = p.runs
    full = "".join(r.text for r in runs)
    i = full.find(old)
    if i < 0:
        return False
    j, pos, first = i + len(old), 0, True
    for r in runs:
        s, e = pos, pos + len(r.text)
        pos = e
        if e <= i or s >= j:
            continue
        a, b, t = max(i, s) - s, min(j, e) - s, r.text
        r.text = t[:a] + (new if first else "") + t[b:]
        first = False
    return True


def replace_doc(doc, old, new):
    n = 0
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs += cell.paragraphs
    for p in paragraphs:
        while old in text_of(p) and replace_in(p, old, new):
            n += 1
    return n


def delete(p):
    p._p.getparent().remove(p._p)


def insert_after(p, style=None):
    el = OxmlElement("w:p")
    p._p.addnext(el)
    new = Paragraph(el, p._parent)
    if style:
        new.style = style
    return new


def add_figure(anchor, image, width_cm, number, caption):
    if not image.exists():
        raise FileNotFoundError(image)
    pic = insert_after(anchor, "PasteMarker")
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.keep_with_next = True
    pic.paragraph_format.first_line_indent = Cm(0)
    pic.add_run().add_picture(str(image), width=Cm(width_cm))
    cap = insert_after(pic, "Body Text")
    cap.add_run(f"Figure {number}.").bold = True
    cap.add_run(" " + caption)
    return cap


def append(p, text):
    p.add_run(text)


# ===========================================================================
# 4. Edits
# ===========================================================================
def edit_text(doc, R):
    gat, fus, lr = "gat", "fusion_mlp", "Logistic regression"
    p_delong, p_wil = R.p("DeLong AUC"), R.p("Wilcoxon fold RMSE")
    mean_rmse = R.v("Predict training mean (reference)", "rmse_um")
    majority = (1 - R.n_unstable / R.n_zones) * 100
    ns = "no_spatial_edges"
    params = f"{R.lat['parameters']:,}"

    # --- Abstract -----------------------------------------------------------
    p = find(doc, "Under trial-grouped cross-validation")
    head = p.text.split("Under trial-grouped cross-validation")[0]
    set_text(p, head + (
        f"Under leave-one-trial-out cross-validation, the GAT detected instability with an area under the ROC "
        f"curve of {R.v(gat, 'roc_auc'):.2f}, significantly below a non-graph multimodal fusion network "
        f"({R.v(fus, 'roc_auc'):.2f}; DeLong p = {p_delong:.3f}), and a logistic regression on hand-crafted "
        f"spectral features was the best classifier ({R.v(lr, 'roc_auc'):.2f}). Removing all spatial edges did "
        f"not reduce performance (ΔAUC = {R.d_auc(ns):+.2f}), and the learned attention was close to uniform. "
        f"The GAT gave the lowest Ra error (RMSE {R.v(gat, 'rmse_um'):.3f} µm against {mean_rmse:.3f} µm for "
        "the training mean), but zone-level Ra had not been measured, so the targets were specimen means and "
        "this result measures material recognition rather than zone-level roughness prediction. The infrared "
        "records could not be calibrated and were not used as input. On "
        f"{R.n_zones} zones with {R.n_unstable} unstable, the graph structure did not improve on simpler models. "
        "The benchmark, the physically defined zone graph and the reproducible pipeline provide a baseline for "
        "larger studies with measured zone-level roughness and calibrated thermography."))
    replace_doc(doc, "; infrared thermography", "")

    # --- Notes to authors -----------------------------------------------------
    for p in [p for p in doc.paragraphs if p.text.strip().startswith("Note to authors")]:
        delete(p)

    # --- 2.9: protocol that was never run, hyperparameter search that never happened ---
    p = find(doc, "Data splitting.", startswith=True)
    replace_in(p, "hyperparameters are selected by an inner trial-grouped split on the remaining",
               "early stopping uses an inner trial-grouped split of the remaining")
    replace_in(p, " (one WAAM and one wrought trial held out for early stopping). A second, harder protocol, "
                  "leave-one-condition-out, holds out both materials at one (speed, depth) combination to test "
                  "generalisation to unseen cutting conditions.",
               " (one WAAM and one wrought trial held out); hyperparameters were fixed in advance "
               "(Section 2.11) and were not tuned on the test folds.")

    # --- 3.2 ------------------------------------------------------------------------
    set_text(find(doc, "The highest ROC-AUC was obtained by"),
             f"The highest ROC-AUC was obtained by logistic regression on the hand-crafted features, "
             f"{R.ci(lr, 'roc_auc')}, followed by the random forest ({R.ci('Random forest', 'roc_auc')}). Among "
             f"the deep models, the multimodal fusion MLP without graph structure performed best "
             f"({R.ci(fus, 'roc_auc')}); the ST-GAT reached {R.ci(gat, 'roc_auc')} and the ST-GCN "
             f"{R.ci('gcn', 'roc_auc')}. The ST-GAT AUC was significantly lower than that of the fusion MLP "
             f"(difference {R.v(gat, 'roc_auc') - R.v(fus, 'roc_auc'):+.2f}, DeLong p = {p_delong:.3f}). Recall, "
             f"the share of unstable zones detected, was {R.ci(gat, 'recall')} for the ST-GAT and "
             f"{R.ci(fus, 'recall')} for the fusion MLP. With only {R.n_unstable} unstable zones in five trials "
             "all intervals are wide, and accuracy mainly reflects the stable majority: a classifier that always "
             f"predicts “stable” would reach {majority:.1f}%. Figure 6 shows the out-of-fold ROC and "
             "precision–recall curves.")

    # --- 3.3 ------------------------------------------------------------------------
    z = R.gat_zones
    mid = z.groupby("material_type").y.first().mean()
    wrong = z[(z.material_type == "WAAM") & (z.p < mid)]
    if len(wrong):
        trials = ", ".join(sorted(wrong.trial_id.unique()))
        outlier = (f" The largest errors come from {len(wrong)} WAAM zones ({trials}) whose predictions fall "
                   "near the wrought value.")
    else:
        outlier = ""
    set_text(find(doc, "Predicting the training-set mean gives an RMSE"),
             f"Predicting the training-set mean gives an RMSE of {mean_rmse:.3f} µm. The learned regressors "
             f"roughly halved this error: the ST-GAT reached the lowest RMSE, {R.ci(gat, 'rmse_um', 3)} µm, "
             f"against {R.ci('Ridge regression', 'rmse_um', 3)} µm for ridge regression and "
             f"{R.ci(fus, 'rmse_um', 3)} µm for the fusion MLP. The confidence intervals overlap, and the "
             f"fold-level difference from the fusion MLP was not significant (Wilcoxon p = {p_wil:.2f}). Because "
             "every zone of a specimen carries the same imputed Ra, these errors measure how reliably each model "
             "recognises the material from the signals; they say nothing about roughness variation along the "
             "wall, which requires measured zone-level Ra. Figure 7 shows the ST-GAT predictions, which form two "
             "clusters at the two material means." + outlier)

    # --- 3.4 ------------------------------------------------------------------------
    set_text(find(doc, "row is the cleanest test of RQ1"),
             "The “no spatial edges” row is the cleanest test of RQ1: it keeps every input and parameter of the "
             f"model except message passing between zones. Removing all spatial edges changed F1 by "
             f"{R.d_f1(ns):+.2f}, ROC-AUC by {R.d_auc(ns):+.2f} and Ra RMSE by {R.d_rmse_pct(ns):+.1f}%. Removing "
             "either edge type alone or the temporal module left performance unchanged or slightly better, and "
             "adding the material flag gave no gain (Figure 8). Of the input modalities, removing the force "
             f"features lowered the ROC-AUC the most, to {R.av('without_force_features', 'roc_auc'):.2f}, whereas "
             f"removing the acceleration features did not ({R.av('without_acceleration_features', 'roc_auc'):.2f}).")

    # --- 3.5 ------------------------------------------------------------------------
    a = R.att
    set_text(find(doc, "mean attention was"),
             "Integrated gradients split the Ra attribution almost evenly between force and acceleration in both "
             "materials, with less than 10% assigned to the kinematic and geometric inputs (Table 5). The GAT "
             f"attention was close to uniform: the mean coefficient was {a.loc['WAAM', 'kinematic_attention']:.3f} "
             f"on kinematic and {a.loc['WAAM', 'thermal_attention']:.3f} on thermal edges for WAAM trials, and "
             f"{a.loc['WROUGHT', 'kinematic_attention']:.3f} and {a.loc['WROUGHT', 'thermal_attention']:.3f} for "
             "wrought trials, against 1/6 ≈ 0.167 for uniform attention (Figure 9). The network therefore did not "
             "learn to favour particular neighbours, which agrees with the ablation result. Thermal images were "
             "not available as input, so the expectation from [1] that thermal information matters more for WAAM "
             "than for wrought material could not be tested.")

    # --- 4.1: say what the ablations showed --------------------------------------------
    p = find(doc, "Porosity enters through all three routes", startswith=True)
    insert_after(p, "Body Text").add_run(
        "The ablations (Table 4, Figure 8) did not show that these mechanisms helped the network on the present "
        "data. Three reasons are likely. First, the heat-diffusion edges carried no measured temperature, because "
        "the thermal frames could not be calibrated, so the mechanism that most motivated spatial coupling was "
        "missing from the inputs. Second, all unstable zones lie in five WAAM trials, so a classifier can rely on "
        "trial- and material-level spectral signatures that need no information from neighbouring zones. Third, "
        f"with {R.n_trials} trials and {R.n_unstable} unstable zones, a network with {params} parameters overfits "
        "more easily than a linear model on a few spectral features, which matches the ranking in Table 2.")

    # --- 4.2 ------------------------------------------------------------------------------
    set_text(find(doc, "RQ1.", startswith=True),
             "Only partly. The graph carried enough information for the ST-GAT to recognise the material-level Ra "
             f"(RMSE {R.v(gat, 'rmse_um'):.3f} µm against {mean_rmse:.3f} µm for the mean predictor), but "
             f"removing message passing between zones changed F1 by {R.d_f1(ns):+.2f}, ROC-AUC by "
             f"{R.d_auc(ns):+.2f} and Ra RMSE by {R.d_rmse_pct(ns):+.1f}%, and the learned attention was close to "
             "uniform. On this dataset, the graph structure itself added no predictive information.",
             keep_label=True)
    set_text(find(doc, "RQ2.", startswith=True),
             f"No. The ST-GAT reached an AUC of {R.ci(gat, 'roc_auc')} and an Ra RMSE of "
             f"{R.ci(gat, 'rmse_um', 3)} µm, against {R.ci(fus, 'roc_auc')} and {R.ci(fus, 'rmse_um', 3)} µm for "
             f"the multimodal fusion MLP; its AUC was significantly lower (DeLong p = {p_delong:.3f}). The best "
             f"instability detector was logistic regression on hand-crafted spectral features "
             f"({R.ci(lr, 'roc_auc')}). For Ra, the learned models differed by less than their confidence "
             f"intervals. With {R.n_zones} zones and {R.n_unstable} positive labels, low-capacity models on "
             "informative features generalised better than the deep sequence and graph models.",
             keep_label=True)

    p = find(doc, "The dataset contains 20 trials at two spindle speeds and one feed.")
    replace_in(p, "The dataset contains 20 trials at two spindle speeds and one feed.",
               f"The dataset contains 20 trials, {R.n_trials} of them usable, at two spindle speeds and one feed.")

    # --- 5 ----------------------------------------------------------------------------------
    set_text(find(doc, "Sensing in production", startswith=True),
             "Dynamometers are rarely installed on production machine tools. In the ablation without force "
             f"features (Table 4) the ROC-AUC of the ST-GAT fell from {R.av('full', 'roc_auc'):.2f} to "
             f"{R.av('without_force_features', 'roc_auc'):.2f}, so on this dataset the accelerometer alone did not "
             "carry the instability information. Spindle-motor current is a low-cost substitute for the "
             "dynamometer that was not recorded here.", keep_label=True)

    # --- 6 ----------------------------------------------------------------------------------
    set_text(find(doc, "This work proposed a spatio-temporal graph formulation"),
             "This work proposed a spatio-temporal graph formulation for monitoring the milling of WAAM Al 5356 "
             "thin walls, in which cutting zones are nodes, tool-path order and heat diffusion define the edges, "
             f"and force and vibration features describe each node. On {R.n_trials} trials evaluated with "
             f"leave-one-trial-out cross-validation, the ST-GAT detected instability with an AUC of "
             f"{R.v(gat, 'roc_auc'):.2f}, significantly below a non-graph multimodal fusion network "
             f"({R.v(fus, 'roc_auc'):.2f}; DeLong p = {p_delong:.3f}) and well below a logistic regression on "
             f"hand-crafted spectral features ({R.v(lr, 'roc_auc'):.2f}). Removing all spatial edges did not "
             "reduce performance, and the learned attention was close to uniform. The ST-GAT gave the lowest Ra "
             f"error (RMSE {R.v(gat, 'rmse_um'):.3f} µm), but the Ra targets were specimen means, so this measures "
             "material recognition. On a dataset of this size the graph structure did not add predictive value. A "
             "fair test of it needs measured zone-level Ra, calibrated thermal frames that give the heat-diffusion "
             "edges real content, and more unstable cuts spread over more trials.")
    set_text(find(doc, "A comparison against classical", startswith=True),
             "A leakage-free benchmark against classical, Euclidean deep and non-graph multimodal baselines, which "
             "showed that on this dataset the graph did not improve on simpler models and that a linear model on "
             "spectral features was the strongest instability detector.")
    set_text(find(doc, "A quantification of the relative importance", startswith=True),
             "A reproducible pipeline (stream alignment, a labelling rule fixed before training, graph construction "
             "and trial-grouped evaluation) that can be rerun once zone-level Ra and calibrated thermal data are "
             "available.")
    p = find_opt(doc, "(Contribution 4", startswith=True)
    if p is not None:
        delete(p)

    # --- Back matter --------------------------------------------------------------------------
    for needle in ("Table S1:", "Supplementary Materials"):
        p = find_opt(doc, needle, startswith=True)
        if p is not None:
            delete(p)
    replace_doc(doc, "(force, acceleration, infrared records and roughness measurements)",
                "(force and acceleration records, raw infrared files and zone labels)")


def edit_figures(doc, R):
    lc, sig = CONFIG.labels, CONFIG.signal
    s = R.summary
    same = (s.loc["trial_06", "spindle_speed_rpm"] == s.loc["trial_16", "spindle_speed_rpm"]
            and s.loc["trial_06", "depth_of_cut_mm"] == s.loc["trial_16", "depth_of_cut_mm"])
    cond = (f" Both walls were cut at {s.loc['trial_16', 'spindle_speed_rpm']:.0f} rpm and "
            f"{s.loc['trial_16', 'depth_of_cut_mm']:.2f} mm depth of cut." if same else "")
    band = f"{sig.chatter_band_hz[0] / 1000:g}–{sig.chatter_band_hz[1] / 1000:g} kHz"

    # Figure 1: keep the image, note the thermal branch.
    replace_in(find(doc, "Three sensing streams were recorded"), "Three sensing streams were recorded",
               "Three sensing streams were recorded (Figure 1)")
    append(find(doc, "Figure 1.", startswith=True),
           " Thermal sensing (orange) was recorded, but thermal features were not used as model input in the "
           "reported runs (Section 2.2).")

    # Figure 2: labelling rule (2.3).
    p = find(doc, "Each zone in each trial was labelled stable or unstable")
    append(p, " Figure 2 illustrates the rule on a wrought and a WAAM wall cut under the same conditions.")
    add_figure(p, FIG / "fig_labelling_rule.png", 16.5, 2,
               f"Instability labelling rule applied to the full-zone force spectrum (three axes summed). Left: a "
               f"wrought zone whose band peak stays below the threshold (stable). Centre: a WAAM zone whose dominant "
               f"peak lies on a spindle-rotation harmonic and is therefore not counted as chatter. Right: a WAAM zone "
               f"with a non-harmonic peak above the threshold (unstable).{cond} Shaded: chatter band ({band}); "
               f"dotted: band median; dashed: median + {lc.peak_prominence_db:g} dB.")

    # Figure 3: edge weights (2.5).
    p = find(doc, "The two edge types are removed in turn in the ablation study.")
    append(p, " Figure 3 shows the two weight matrices, which are the same for every trial.")
    add_figure(p, FIG / "fig_edge_weights.png", 14.0, 3,
               "Edge weights of the six-zone graph. Left: kinematic edges join zones that are consecutive along the "
               "tool path. Right: thermal edges from the heat kernel of Equation (10), computed from the zone "
               "spacing, the feed rate and the thermal diffusivity of the alloy.")

    # Figure 4: architecture (2.7) replaces the old Figure 2.
    old_cap = find(doc, "Figure 2. Architecture", startswith=True)
    old_pic = Paragraph(old_cap._p.getprevious(), old_cap._parent)
    if "<w:drawing" not in old_pic._p.xml:
        raise RuntimeError("Expected the old Figure 2 image just above its caption.")
    anchor = Paragraph(old_pic._p.getprevious(), old_cap._parent)
    delete(old_pic)
    delete(old_cap)
    add_figure(anchor, FIG / "fig_architecture.png", 17.4, 4,
               "Spatio-temporal graph network as implemented. (a) Milling and sensing: six 5 mm zones and 100 ms "
               "windows. (b) Node inputs: hand-crafted features and 1D-CNN encodings of the raw force and "
               "acceleration windows. (c) Zone graph with kinematic and heat-kernel edges. (d) Two spatio-temporal "
               "blocks of edge-aware graph attention (graph convolution for the ST-GCN) and gated temporal "
               "convolution. (e) Output heads and multi-task loss. The thermal image encoder described in Section "
               "2.7 was disabled because no calibrated frames were available.")
    replace_in(find(doc, "Before the graph layers, each modality"), "Before the graph layers, each modality",
               "Before the graph layers (Figure 4), each modality")

    # Figure 5: chatter map (3.1).
    p = find(doc, "The dataset is strongly imbalanced")
    append(p, " Figure 5 shows the label of every zone.")
    add_figure(p, FIG / "fig_chatter_map.png", 8.5, 5,
               f"Instability label of every zone of the {R.n_trials} usable trials (red: unstable). The number in "
               "each cell is the prominence (dB) of the dominant peak in the chatter band above the band median; "
               "the horizontal line separates the wrought trials (top) from the WAAM trials.")

    # Figure 6: ROC / PR (3.2).
    add_figure(find(doc, "The highest ROC-AUC was obtained by"), FIG / "fig_roc_pr.png", 16.5, 6,
               "Out-of-fold ROC (left) and precision–recall (right) curves for instability detection under "
               "leave-one-trial-out cross-validation, with probabilities averaged over five seeds. AP: average "
               "precision; dotted lines: chance level.")

    # Figure 7: parity (3.3).
    add_figure(find(doc, "Predicting the training-set mean gives an RMSE"), FIG / "fig_parity.png", 8.0, 7,
               "Predicted versus target Ra for the ST-GAT (out-of-fold, averaged over five seeds). The targets are "
               "the specimen-mean Ra of [1], so each material forms one column; points are jittered horizontally "
               "for visibility.")

    # Figure 8: ablation (3.4).
    add_figure(find(doc, "row is the cleanest test of RQ1"), FIG / "fig_ablation.png", 16.0, 8,
               "F1 score and ROC-AUC of the ST-GAT ablations with 95% trial-bootstrap confidence intervals. The "
               "dashed line marks the full model.")

    # Figure 9: attention (3.5).
    add_figure(find(doc, "Integrated gradients split the Ra attribution"), FIG / "fig_attention.png", 14.0, 9,
               "Mean GAT attention coefficients between zones, averaged over the trials of each material. Uniform "
               "attention over six zones would give 1/6 ≈ 0.17 in every cell.")


def apply_author_info(doc, info):
    filled = []
    if info.get("title"):
        set_text(doc.paragraphs[0], info["title"])
        filled.append("title")
    if info.get("authors_line"):
        p = doc.paragraphs[1]
        if p.text.strip():
            p = insert_after(doc.paragraphs[0], "First Paragraph")
        p.add_run(info["authors_line"])
        last = p
        for key in ("affiliations", "correspondence"):
            if info.get(key):
                last = insert_after(last, "Body Text")
                last.add_run(info[key])
        filled.append("authors")
    if info.get("accelerometer"):
        replace_doc(doc, "(model and mounting position not recorded in the data archive – [author to supply])",
                    f"({info['accelerometer']})")
    if info.get("ir_camera"):
        replace_doc(doc, "exact model, frame rate and emissivity setting not recorded in the archive – "
                         "[author to confirm]", info["ir_camera"])
    if info.get("repository"):
        replace_doc(doc, "[repository – author to supply]", info["repository"])
        replace_doc(doc, "[repository, DOI – author to supply]", info["repository"])
    if info.get("author_contributions"):
        text = info["author_contributions"].strip()
        if "All authors have read" not in text:
            text += " All authors have read and agreed to the published version of the manuscript."
        set_text(find(doc, "Author Contributions:", startswith=True), text, keep_label=True)
    if info.get("funding"):
        set_text(find(doc, "Funding:", startswith=True), info["funding"], keep_label=True)
    if info.get("acknowledgments"):
        replace_in(find(doc, "Acknowledgments:", startswith=True), "[author to complete]",
                   info["acknowledgments"].strip())
    if info.get("ref1_article_number"):
        replace_doc(doc, "[article no. – check]", info["ref1_article_number"])
    return filled


# ===========================================================================
# 5. Final check
# ===========================================================================
def final_check(doc, info):
    problems = []
    body = [p.text for p in doc.paragraphs]
    tables = [c.text for t in doc.tables for r in t.rows for c in r.cells]
    open_ = {}
    for t in body + tables:
        for b in re.findall(r"\[[^\]]*\]", t):
            if not re.fullmatch(r"\[\d[\d,–\- ]*\]", b):
                where = t.split(":")[0][:40] if ":" in t[:45] else t[:40]
                open_.setdefault((b, where), 0)
                open_[(b, where)] += 1
    for (b, where), n in open_.items():
        problems.append(f"placeholder {b}{f' x{n}' if n > 1 else ''}  in '{where}...'")
    for bad in ("Note to authors", "Table S", "Figure S", "pending", "outputs/", "for this draft", "make_report"):
        for t in body:
            if bad in t:
                problems.append(f"leftover text '{bad}': {t[:80]}...")
    captions = [t for t in body if re.match(r"Figure \d+\.", t)]
    nums = [int(re.match(r"Figure (\d+)\.", t).group(1)) for t in captions]
    if nums != list(range(1, len(nums) + 1)):
        problems.append(f"figure captions out of order: {nums}")
    for n in nums:
        if not any(re.search(rf"Figures? {n}\b", t) and not t.startswith(f"Figure {n}.") for t in body):
            problems.append(f"Figure {n} is never referred to in the text")
    drawings = sum(p._p.xml.count("<w:drawing") for p in doc.paragraphs)
    if drawings != len(nums):
        problems.append(f"{drawings} images but {len(nums)} figure captions")
    if not info.get("authors_line"):
        problems.append("author names / affiliations missing under the title (authors_line in author_info.json)")
    if not info.get("title"):
        problems.append("title still says 'Force, Vibration and Thermal Signals' although thermal data were not "
                        "used; consider setting 'title' in author_info.json")
    return problems, len(nums)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-regen", action="store_true", help="do not regenerate figures / refill the draft")
    ap.add_argument("--force-narrative", action="store_true",
                    help="write the text even if a narrative check fails (not recommended)")
    args = ap.parse_args()

    if not args.skip_regen:
        run_step("make_report.py", "1/3  Refill the draft from outputs/ (make_report.py)")
        run_step("make_journal_figures.py", "2/3  Journal figures from the current run")
        run_step("draw_architecture.py", "3/3  Architecture figure")

    R = Results()
    print("\n=== Parity plot ===")
    draw_parity(R)

    print("\n=== Checking that the results support the text ===")
    failed = []
    for name, ok in narrative_checks(R):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        if not ok:
            failed.append(name)
    if failed and not args.force_narrative:
        sys.exit("\nThe results changed since the text was written. The sentences that rely on the failed checks "
                 "above are in edit_text() in src/finalize_report.py. Revise them, then rerun.")

    if not AUTHOR_FILE.exists():
        AUTHOR_FILE.write_text(json.dumps(AUTHOR_TEMPLATE, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nCreated {AUTHOR_FILE.relative_to(ROOT)}: fill it in and rerun with --skip-regen.")
    try:
        info = json.loads(AUTHOR_FILE.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        sys.exit(f"\n{AUTHOR_FILE.name} is not valid JSON ({e}). Check quotes and commas, then rerun.")

    print("\n=== Editing the manuscript ===")
    doc = docx.Document(FILLED)
    edit_text(doc, R)
    edit_figures(doc, R)
    filled = apply_author_info(doc, info)
    doc.save(FINAL)

    problems, n_fig = final_check(docx.Document(FINAL), info)
    lines = [f"Final manuscript: {FINAL}", f"Figures embedded: {n_fig}",
             f"Author details applied: {', '.join(filled) or 'none'}", ""]
    if problems:
        lines += ["NOT READY. Still to do:"] + [f"  - {x}" for x in problems]
    else:
        lines += ["READY TO SUBMIT. Open the file in Word once, check page breaks, and save it as PDF if needed."]
    CHECKLIST.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n(checklist saved to {CHECKLIST.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
