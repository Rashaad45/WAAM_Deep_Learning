"""Step 3 of the pipeline: fill the manuscript draft from the dataset and the training outputs.

Reads   report/WAAM_STGNN_Materials_draft.docx  (never modified)
        data/processed/dataset_summary.csv, data/zone_labels.csv
        outputs/*.csv, outputs/*.json
Writes  report/WAAM_STGNN_Materials_filled.docx  the draft with every fillable blank completed
        report/fill_log.csv                      every change: location, old text, new text, source
        outputs/figures/*.png                    Figures 3-5 and the ROC curves

Rules followed when filling:
  * numbers in Results come only from outputs/ (this run), never typed by hand;
  * method details come from config.py, the raw files, or the earlier paper [1];
  * literature values are used only where the draft itself asks for them (alloy data);
  * where a value does not exist (e.g. zone-level Ra, inter-annotator kappa, author
    details) the blank is replaced by an explicit statement, never by an invented number.

Usage:  python src/make_report.py
"""
import json
import re
from pathlib import Path

import docx
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import CONFIG, CACHE_DIR, DATA_DIR, OUTPUT_DIR, REPORT_DIR  # noqa: E402

DRAFT = REPORT_DIR / "WAAM_STGNN_Materials_draft.docx"
FILLED = REPORT_DIR / "WAAM_STGNN_Materials_filled.docx"
M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
PENDING = "[pending: run the ablation stage]"
NA_THERMAL = "n/a (no calibrated IR)"

MODEL_ROWS = {"Logistic regression": "Logistic regression", "Random forest": "Random forest",
              "SVM (RBF)": "SVM (RBF)", "1D-CNN": "cnn1d", "BiLSTM": "bilstm",
              "Multimodal fusion MLP (no graph)": "fusion_mlp", "ST-GCN (this work)": "gcn",
              "ST-GAT (this work)": "gat", "Ridge regression": "Ridge regression",
              "Random forest regressor": "Random forest regressor", "SVR (RBF)": "SVR (RBF)",
              "Predict training mean (reference)": "Predict training mean (reference)"}


# --------------------------------------------------------------------------
# Loading results
# --------------------------------------------------------------------------
def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


class Results:
    """Everything the report needs, loaded once."""

    def __init__(self):
        self.summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv")
        self.labels = pd.read_csv(DATA_DIR / "zone_labels.csv")
        self.included = self.summary[self.summary.included]
        self.main = read_csv(OUTPUT_DIR / "metrics_main.csv")
        if self.main is None:
            raise SystemExit("outputs/metrics_main.csv not found - run train_evaluate.py --stage main first.")
        self.main = self.main.set_index("model")
        self.by_material = read_csv(OUTPUT_DIR / "metrics_by_material.csv")
        self.ablation = read_csv(OUTPUT_DIR / "ablation.csv")
        if self.ablation is not None:
            self.ablation = self.ablation.set_index("variant")
        self.attribution = read_csv(OUTPUT_DIR / "attribution.csv")
        self.attention = read_csv(OUTPUT_DIR / "attention.csv")
        self.tests = read_csv(OUTPUT_DIR / "stat_tests.csv")
        self.loco = read_csv(OUTPUT_DIR / "loco" / "metrics_main.csv")
        self.latency = json.loads((OUTPUT_DIR / "latency.json").read_text())
        self.info = json.loads((OUTPUT_DIR / "run_info.json").read_text())
        self.pred = pd.read_csv(OUTPUT_DIR / "predictions_oof.csv")

    # -- convenience ---------------------------------------------------------
    @property
    def n_trials(self) -> int:
        return len(self.included)

    def trials_of(self, material: str) -> int:
        return int((self.included.material_type == material).sum())

    def value(self, model: str, metric: str) -> float:
        return float(self.main.loc[model, metric]) if model in self.main.index else np.nan

    def ci_text(self, model: str, metric: str, digits: int = 2, table=None) -> str:
        table = self.main if table is None else table
        if model not in table.index or metric not in table.columns or pd.isna(table.loc[model, metric]):
            return "n/a"
        v, lo, hi = (table.loc[model, metric], table.loc[model, f"{metric}_ci_low"], table.loc[model, f"{metric}_ci_high"])
        f = f"{{:.{digits}f}}"
        return f"{f.format(v)} ({f.format(lo)}–{f.format(hi)})" if pd.notna(lo) else f.format(v)

    def test(self, name: str) -> pd.Series | None:
        if self.tests is None or self.tests.empty:
            return None
        rows = self.tests[self.tests.test == name]
        return rows.iloc[0] if len(rows) else None

    def delta_rmse(self, variant: str) -> float:
        if self.ablation is None or variant not in self.ablation.index or "full" not in self.ablation.index:
            return np.nan
        full = self.ablation.loc["full", "rmse_um"]
        return float((self.ablation.loc[variant, "rmse_um"] - full) / full * 100)

    def delta_f1(self, variant: str) -> float:
        if self.ablation is None or variant not in self.ablation.index or "full" not in self.ablation.index:
            return np.nan
        return float(self.ablation.loc[variant, "f1"] - self.ablation.loc["full", "f1"])

    def best_baseline(self) -> str:
        """Strongest non-graph multimodal baseline = best AUC among non-graph learned models."""
        candidates = ["fusion_mlp", "bilstm", "cnn1d", "Logistic regression", "Random forest", "SVM (RBF)"]
        aucs = {m: self.value(m, "roc_auc") for m in candidates if m in self.main.index}
        aucs = {m: a for m, a in aucs.items() if np.isfinite(a)}
        return max(aucs, key=aucs.get) if aucs else "fusion_mlp"


def pretty(model: str) -> str:
    names = {"cnn1d": "1D-CNN", "bilstm": "BiLSTM", "fusion_mlp": "multimodal fusion MLP",
             "gcn": "ST-GCN", "gat": "ST-GAT"}
    return names.get(model, model)


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+.{digits}f}"


# --------------------------------------------------------------------------
# Low-level docx editing (keeps equations and run formatting intact)
# --------------------------------------------------------------------------
class Filler:
    def __init__(self, document: docx.Document):
        self.doc = document
        self.log = []

    def _paragraphs(self):
        yield from self.doc.paragraphs
        for table in self.doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    yield from cell.paragraphs

    def replace(self, old: str, new: str, source: str, count: int = 1) -> None:
        """Replace `old` inside a single run (first `count` occurrences in document order)."""
        done = 0
        for p in self._paragraphs():
            for run in p.runs:
                while old in run.text and done < count:
                    run.text = run.text.replace(old, new, 1)
                    done += 1
                    self.log.append(dict(location=p.text[:60], old=old, new=new, source=source))
            if done >= count:
                return
        if done == 0:
            self.log.append(dict(location="NOT FOUND", old=old, new=new, source=source))

    def set_paragraph(self, contains: str, text: str, source: str, keep_label: bool = False) -> None:
        """Replace the whole text of the paragraph containing `contains` (removes equations in it)."""
        for p in self.doc.paragraphs:
            if contains in p.text:
                old = p.text
                runs = p.runs
                start = 1 if keep_label and len(runs) > 1 else 0
                runs[start].text = (" " if keep_label else "") + text
                for r in runs[start + 1:]:
                    r._r.getparent().remove(r._r)
                for m in p._p.findall(f".//{M_NS}oMath"):
                    m.getparent().remove(m)
                self.log.append(dict(location=old[:60], old=old, new=text, source=source))
                return
        self.log.append(dict(location="NOT FOUND", old=contains, new=text, source=source))

    def set_math(self, old: str, new: str, source: str) -> None:
        """Replace text inside an equation (m:t element)."""
        for p in self.doc.paragraphs:
            for t in p._p.iter(f"{M_NS}t"):
                if t.text and old in t.text:
                    t.text = t.text.replace(old, new)
                    self.log.append(dict(location=p.text[:60], old=old, new=new, source=source))
                    return
        self.log.append(dict(location="NOT FOUND (math)", old=old, new=new, source=source))

    def set_cell(self, table, row: int, col: int, text: str, source: str) -> None:
        cell = table.rows[row].cells[col]
        para = cell.paragraphs[0]
        old = cell.text
        if para.runs:
            para.runs[0].text = text
            for r in para.runs[1:]:
                r._r.getparent().remove(r._r)
        else:
            para.add_run(text)
        self.log.append(dict(location=f"table '{table.rows[0].cells[0].text}' r{row}c{col}", old=old, new=text,
                             source=source))


def find_table(doc: docx.Document, first_header: str, second_header: str = ""):
    for t in doc.tables:
        cells = [c.text.strip() for c in t.rows[0].cells]
        if cells[0] == first_header and (not second_header or second_header in cells):
            return t
    raise KeyError(first_header)


# --------------------------------------------------------------------------
# Section fills
# --------------------------------------------------------------------------
def fill_methods(f: Filler, r: Results) -> None:
    sig, cfg = CONFIG.signal, CONFIG
    s = r.included
    weak = s[s.accel_alignment_r < cfg.cut.min_alignment_r]
    excluded = r.summary[~r.summary.included]
    src_cfg, src_data, src_paper = "src/config.py", "raw files in WAAM/ (measured)", "ref. [1]"

    # 2.2 Machining and sensing ------------------------------------------------
    f.replace("giving 10 trials per material and 20 in total [1].",
              f"giving 10 trials per material and 20 in total [1]. Synchronised force and acceleration records "
              f"exist for trials 4–20 of the archive; {r.n_trials} of them ({r.trials_of('WAAM')} WAAM, "
              f"{r.trials_of('WROUGHT')} wrought) were usable (trials 1–3 have no force record; "
              + "; ".join(f"{t.trial_id}: {t.exclusion_reason}" for t in excluded.itertuples()) + ").",
              src_data)
    f.set_paragraph("Triaxial accelerometer [model]",
                    "Triaxial accelerometer (model and mounting position not recorded in the data archive – "
                    "[author to supply]), acquired with the KiNOVA data-acquisition software and sampled at "
                    f"{sig.acceleration_fs_hz / 1000:g} kHz (measured from the time column of every file). "
                    "Force and acceleration were therefore recorded at the same rate and no resampling was needed "
                    f"for synchronisation; for the CNN encoders both streams were decimated by {sig.encoder_decimation} "
                    f"to {sig.force_fs_hz / sig.encoder_decimation / 1000:g} kHz with an anti-aliasing polyphase filter "
                    "(Nyquist 6.25 kHz, above both structural modes). All three axes are exported in g by the "
                    "acquisition software, which was taken as the common unit reference.",
                    src_data, keep_label=True)
    f.set_paragraph("Infrared camera [model]",
                    "Infrared records were taken with a Sonel thermal imager (radiometric .irgd files, 384 × 288 "
                    "pixels; exact model, frame rate and emissivity setting not recorded in the archive – "
                    "[author to confirm]). The .irgd files could not be exported to calibrated temperature frames "
                    "without the vendor software, so thermal images were NOT used as model input in the runs "
                    "reported here; the thermal branch of the network is disabled and only the heat-diffusion "
                    "edges (which depend on geometry and alloy properties, not on images) are retained.",
                    src_data, keep_label=True)
    f.set_paragraph("Streams were aligned in time using",
                    "The force and acceleration PCs were not clock-synchronised, so streams were aligned by matching "
                    "the RMS envelope of the acceleration (10 ms bins) to a step template that marks the cut in the "
                    "force record; only lags for which the vibration record contains the whole cut were searched. "
                    f"The correlation at the selected lag was r = {s.accel_alignment_r.min():.2f}–"
                    f"{s.accel_alignment_r.max():.2f} (median {s.accel_alignment_r.median():.2f}) and the lags were "
                    f"{s.accel_lag_s.min():+.2f} to {s.accel_lag_s.max():+.2f} s. The residual alignment uncertainty "
                    "is about ±10 ms (one envelope bin)"
                    + (f"; for {', '.join(weak.trial_id)} the accelerometer barely responded to the cut and the "
                       "alignment is less certain." if len(weak) else "."),
                    src_data)

    # 2.3 Labels -----------------------------------------------------------------
    lab = r.labels
    n_unst, n_tot = int(lab.chatter_label.sum()), len(lab)
    by_mat = lab.groupby("material_type").chatter_label.agg(["sum", "count"])
    lc = CONFIG.labels
    f.set_paragraph("Each zone in each trial was labelled stable or unstable",
                    "Each zone in each trial was labelled stable or unstable. Because the model receives spectral "
                    "features, a label defined from the same short-window spectra would make high accuracy partly "
                    "circular. Labels were therefore computed from the full-duration force record of each zone "
                    f"(≈{CONFIG.zone_duration_s:.2f} s, Welch PSD with {lc.label_nperseg} points, ≈0.76 Hz resolution, "
                    f"against ≈24 Hz in the {sig.window_ms:g} ms model windows) with a rule fixed before model training: "
                    "a zone is unstable when the dominant peak of the summed three-axis force PSD in the "
                    f"{sig.chatter_band_hz[0] / 1000:g}–{sig.chatter_band_hz[1] / 1000:g} kHz band does not lie within "
                    f"±{sig.harmonic_tolerance_hz:g} Hz of a spindle-rotation harmonic and rises at least "
                    f"{lc.peak_prominence_db:g} dB above the band median. Visual inspection of the machined surface was "
                    "not available for this dataset (no per-zone surface images in the archive), so the labels rest on "
                    "this single criterion and inter-annotator agreement does not apply. The rule labelled "
                    f"{n_unst} of {n_tot} zones unstable ({n_unst / n_tot * 100:.1f}%): "
                    + ", ".join(f"{int(v['sum'])}/{int(v['count'])} for {m.lower()}" for m, v in by_mat.iterrows()) + ".",
                    "data/zone_labels.csv", keep_label=True)
    f.set_paragraph("Ra was measured with [instrument]",
                    "Zone-level Ra measurements are not available in the data archive. The earlier study [1] "
                    "reports Ra as specimen means over three positions (left, centre, right): 1.22 µm for WAAM and "
                    "0.83 µm for wrought material. In the runs reported here every zone was assigned the mean of its "
                    "specimen (IMPUTED), so the Ra head can only learn the material difference and Ra errors must not "
                    "be read as zone-level prediction errors. Measured zone values (instrument, ISO 21920-2 cut-off "
                    "and evaluation length, positions per zone) can be entered in data/ra_measurements.csv, after "
                    "which the pipeline uses them automatically. Measurement repeatability σ_meas is likewise not "
                    "available.", src_paper, keep_label=True)

    # 2.4 Features -------------------------------------------------------------------
    f.replace("(here [2.5–5.5] kHz)", f"(here {sig.chatter_band_hz[0] / 1000:g}–{sig.chatter_band_hz[1] / 1000:g} kHz)", src_cfg)
    hop_bin = sig.force_fs_hz / sig.welch_segment / 2
    f.replace(" is [X] Hz.", f" is {hop_bin:.1f} Hz (half a Welch bin of the {sig.welch_segment}-point segments, "
                             "around each tooth-passing harmonic).", src_cfg)
    w = sig.window_ms
    f.replace("Window length was [X] ms with [50]% overlap, which spans [X] tooth periods",
              f"Window length was {w:g} ms with {sig.overlap_fraction * 100:.0f}% overlap "
              f"({CONFIG.windows_per_zone} windows per zone), which spans {w / 15:.1f} tooth periods", src_cfg)
    f.replace("and [X] at 1400 rpm", f"and {w / (60 / 1400 / 4 * 1000):.1f} at 1400 rpm", src_cfg)
    f.replace("dividing the [30] mm tool path", "dividing the 30 mm tool path", src_paper)
    f.replace("segments of [5] mm.", "segments of 5 mm (in practice the detected cut of each force record, "
                                     f"{r.included.cut_duration_s.min():.1f}–{r.included.cut_duration_s.max():.1f} s long, "
                                     "was divided into six equal parts).", src_data)
    f.replace(" [values to be checked against the alloy datasheet; the WAAM deposit may differ]",
              " (datasheet values for wrought Al 5356; the WAAM deposit may differ)", "alloy datasheet")
    f.replace("and a small 2D-CNN over the thermal region of interest.",
              "and a small 2D-CNN over the thermal region of interest (disabled in the runs reported here, since no "
              "calibrated thermal frames were available).", src_data)
    f.replace("[Report which variant was retained and the selected λ.]",
              f"The fixed-weight loss with λ = {CONFIG.train.regression_weight:g} was retained; the uncertainty "
              "weighting was not used in the reported runs.", src_cfg)

    # 2.9 Evaluation protocol ----------------------------------------------------------
    n = r.n_trials
    f.replace("cross-validation (20 folds)", f"cross-validation ({n} folds)", src_data)
    f.replace("on the remaining 19 trials", f"on the remaining {n - 1} trials (one WAAM and one wrought trial held "
                                            "out for early stopping)", src_data)
    f.replace(" [Optionally: train on wrought, test on WAAM, to probe transfer across materials.]",
              " Transfer from wrought to WAAM material was not evaluated.", "this run")
    f.replace(". Metrics are given as mean ± standard deviation over folds with 95% bootstrap confidence intervals.",
              ". Because each test fold contains only six zones, metrics are computed on the pooled out-of-fold "
              "predictions and given with 95% confidence intervals from 2000 bootstrap resamples of trials.", "this run")
    f.replace("paired differences in fold-level F1 and RMSE with the ", "paired differences in fold-level RMSE with the ",
              "this run")
    f.replace("With 20 trials the statistical power", f"With {n} trials the statistical power", src_data)
    f.replace(", and GNNExplainer [36] is used to identify the subgraphs most responsible for high-roughness predictions.",
              ". (GNNExplainer [36] was not applied in the present runs.)", "this run")

    # 2.11 Implementation ----------------------------------------------------------------
    info, t = r.info, CONFIG.train
    f.set_paragraph("Models were implemented in",
                    f"Models were implemented in PyTorch {info['versions']['torch'].split('+')[0]} (graph convolution "
                    "and attention layers written directly in PyTorch; PyTorch Geometric was not required for "
                    f"six-node graphs) and trained with Adam (learning rate {t.learning_rate:g}, weight decay "
                    f"{t.weight_decay:g}), batch size 1 graph, for up to {info['max_epochs']} epochs with early "
                    f"stopping on the inner validation loss (patience {info['patience']}). Hidden dimension "
                    f"{t.hidden_dim}, {t.attention_heads} attention heads, temporal kernel K_t = {t.temporal_kernel}, "
                    f"dropout {t.dropout:g}. Random seeds were fixed and each configuration was run with "
                    f"{len(info['seeds'])} seed(s) ({', '.join(map(str, info['seeds']))}). Training and inference "
                    f"latency were measured on the CPU of a laptop (AMD Ryzen 5 5500U, 6 cores, no GPU) (Section 5). "
                    "Code and processed data are available at [repository – author to supply].",
                    "outputs/run_info.json")


def fill_results(f: Filler, r: Results) -> None:
    lab = r.labels
    src = "outputs/ (this run)"

    # ---- Table 1 ---------------------------------------------------------------
    t1 = find_table(f.doc, "Quantity")
    T = int(r.included.windows_per_zone.iloc[0])
    counts = {m: r.trials_of(m) for m in ("WAAM", "WROUGHT")}
    for col, mat in ((1, "WAAM"), (2, "WROUGHT")):
        sub = lab[lab.material_type == mat]
        f.set_cell(t1, 1, col, f"10 ({counts[mat]} usable)", "data/processed/dataset_summary.csv")
        f.set_cell(t1, 2, col, "6", "src/config.py")
        f.set_cell(t1, 3, col, str(T), "src/config.py")
        f.set_cell(t1, 4, col, str(counts[mat] * T), src)
        f.set_cell(t1, 5, col, f"{int(sub.chatter_label.sum())} / {int((sub.chatter_label == 0).sum())}", "data/zone_labels.csv")
        f.set_cell(t1, 6, col, f"0 (mean of [1] imputed for {len(sub)})", "data/zone_labels.csv")
        ra = CONFIG.labels.ra_material_mean_um[mat]
        f.set_cell(t1, 7, col, f"{ra:.2f} (imputed, single value)", "ref. [1]")
        f.set_cell(t1, 8, col, "not available", "data archive")
    f.set_cell(t1, 1, 3, f"20 ({r.n_trials} usable)", "data/processed/dataset_summary.csv")
    f.set_cell(t1, 4, 3, str(r.n_trials * T), src)
    f.set_cell(t1, 5, 3, f"{int(lab.chatter_label.sum())} / {int((lab.chatter_label == 0).sum())}", "data/zone_labels.csv")
    f.set_cell(t1, 6, 3, f"0 / {len(lab)}", "data/zone_labels.csv")

    # Class balance paragraph (3.1)
    per_trial = lab.groupby(["trial_id", "material_type"]).chatter_label.sum().reset_index()
    summ = r.included.set_index("trial_id")
    unstable_trials = per_trial[per_trial.chatter_label > 0]
    desc = "; ".join(f"{t.trial_id} ({int(summ.loc[t.trial_id, 'spindle_speed_rpm'])} rpm, "
                     f"{summ.loc[t.trial_id, 'depth_of_cut_mm']:.2f} mm): {int(t.chatter_label)}/6"
                     for t in unstable_trials.itertuples())
    wrought_unst = int(lab[lab.material_type == "WROUGHT"].chatter_label.sum())
    f.set_paragraph("[Describe the class balance",
                    f"The dataset is strongly imbalanced: {int(lab.chatter_label.sum())} of {len(lab)} zones "
                    f"({lab.chatter_label.mean() * 100:.1f}%) were labelled unstable, in {len(unstable_trials)} trials "
                    f"({desc}). "
                    + ("No wrought trial was labelled unstable, so every unstable zone is a WAAM zone; a classifier "
                       "can therefore partly separate the classes through material-related signal differences, and "
                       "the classification results below must be read with this confound in mind. "
                       if wrought_unst == 0 else "")
                    + "Ra targets are the material means of [1] (Table 1), so Ra results measure how well each model "
                      "recognises the material from the signals, not zone-level roughness.", "data/zone_labels.csv")

    # ---- Table 2 ---------------------------------------------------------------
    t2 = find_table(f.doc, "Model", "ROC-AUC")
    for i, row in enumerate(t2.rows[1:], start=1):
        key = MODEL_ROWS.get(row.cells[0].text.strip())
        f.set_cell(t2, i, 2, r.ci_text(key, "accuracy", 1), src)
        for col, metric in ((3, "precision"), (4, "recall"), (5, "f1"), (6, "roc_auc")):
            f.set_cell(t2, i, col, r.ci_text(key, metric, 2), src)
    f.replace("(mean ± SD over 20 folds)", f"(pooled out-of-fold predictions of {r.n_trials} folds; 95% trial-bootstrap "
                                           "confidence interval in parentheses)", src)

    # Text 3.2
    gat, fus = "gat", "fusion_mlp"
    dl = r.test("DeLong AUC")
    margin = r.value(gat, "roc_auc") - r.value(fus, "roc_auc")
    with_auc = [m for m in r.main.index if pd.notna(r.main.loc[m, "roc_auc"])]
    best = max(with_auc, key=lambda m: r.main.loc[m, "roc_auc"]) if with_auc else gat
    loco_txt = ""
    if r.loco is not None:
        lo = r.loco.set_index("model")
        if gat in lo.index and fus in lo.index:
            loco_txt = (f" Under leave-one-condition-out validation (Table S1) the AUC was "
                        f"{lo.loc[gat, 'roc_auc']:.2f} for the ST-GAT and {lo.loc[fus, 'roc_auc']:.2f} for the fusion MLP.")
    else:
        loco_txt = " The leave-one-condition-out protocol (Table S1) was not run for this draft."
    f.set_paragraph("[Text: state the best model",
                    f"The highest ROC-AUC was obtained by the {pretty(best)} ({r.ci_text(best, 'roc_auc')}). "
                    f"The ST-GAT reached an AUC of {r.ci_text(gat, 'roc_auc')} against {r.ci_text(fus, 'roc_auc')} for the "
                    f"multimodal fusion MLP without graph structure (difference {margin:+.2f}"
                    + (f", DeLong p = {dl.p_value:.3f}" if dl is not None and pd.notna(dl.p_value) else "") + ")."
                    + loco_txt +
                    f" Recall, the share of unstable zones that were detected, was {r.ci_text(gat, 'recall')} for the "
                    f"ST-GAT and {r.ci_text(fus, 'recall')} for the fusion MLP; with only "
                    f"{int(lab.chatter_label.sum())} unstable zones these intervals are wide, and accuracy is dominated "
                    "by the stable majority class.", src)

    # ---- Table 3 ---------------------------------------------------------------
    t3 = find_table(f.doc, "Model", "RMSE / σ_meas")
    for i, row in enumerate(t3.rows[1:], start=1):
        key = MODEL_ROWS.get(row.cells[0].text.strip())
        f.set_cell(t3, i, 1, r.ci_text(key, "rmse_um", 3), src)
        f.set_cell(t3, i, 2, r.ci_text(key, "mae_um", 3), src)
        if row.cells[3].text.strip() != "0":
            f.set_cell(t3, i, 3, r.ci_text(key, "r2", 2), src)
        f.set_cell(t3, i, 4, "n/a (σ_meas unavailable)", "data archive")
    f.replace(" Zone-level Ra prediction under leave-one-trial-out cross-validation.",
              " Zone-level Ra prediction under leave-one-trial-out cross-validation (pooled out-of-fold; "
              "Ra targets are material means of [1], see Section 2.3).", src)
    mean_rmse = r.value("Predict training mean (reference)", "rmse_um")
    with_rmse = [m for m in r.main.index if pd.notna(r.main.loc[m, "rmse_um"])
                 and m != "Predict training mean (reference)"]
    best_ra = min(with_rmse, key=lambda m: r.main.loc[m, "rmse_um"]) if with_rmse else gat
    f.set_paragraph("[Text: include the trivial mean predictor",
                    f"Predicting the training-set mean gives an RMSE of {mean_rmse:.3f} µm, the size of the error "
                    f"when the material is ignored. The lowest RMSE, {r.ci_text(best_ra, 'rmse_um', 3)} µm, was obtained "
                    f"by the {pretty(best_ra)}; the ST-GAT reached {r.ci_text(gat, 'rmse_um', 3)} µm and the fusion MLP "
                    f"{r.ci_text(fus, 'rmse_um', 3)} µm. Because every zone of a specimen carries the same imputed Ra, "
                    "these errors measure material recognition; they do not show whether errors concentrate at the "
                    "end zones, which requires measured zone-level Ra. Figure 3 shows the parity plot coloured by "
                    "material and Figure 4 the predicted Ra along the wall for two WAAM trials "
                    "(outputs/figures).", src)

    # ---- Table 4 ---------------------------------------------------------------
    t4 = find_table(f.doc, "Configuration")
    variant_rows = {"Full model": "full", "Without thermal features": None,
                    "Without force features": "without_force_features",
                    "Without acceleration features": "without_acceleration_features",
                    "Without thermal edges (kinematic only)": "without_thermal_edges",
                    "Without kinematic edges (thermal only)": "without_kinematic_edges",
                    "No spatial edges (self-loops only)": "no_spatial_edges",
                    "Without temporal module": "without_temporal_module",
                    "With material flag as feature": "material_flag",
                    "Single-task (instability only / Ra only)": "single"}
    abl = r.ablation
    for i, row in enumerate(t4.rows[1:], start=1):
        variant = variant_rows.get(row.cells[0].text.strip(), "?")
        if variant is None:
            for col in (1, 2, 3, 4):
                f.set_cell(t4, i, col, NA_THERMAL, "data archive")
            continue
        if abl is None:
            for col in (1, 2, 3, 4):
                f.set_cell(t4, i, col, PENDING if not (variant == "full" and col in (2, 4)) else "—", "")
            continue
        if variant == "single":
            f1 = abl.loc["classification_only", "f1"] if "classification_only" in abl.index else np.nan
            rmse = abl.loc["roughness_only", "rmse_um"] if "roughness_only" in abl.index else np.nan
            full = abl.loc["full"]
            f.set_cell(t4, i, 1, f"{f1:.2f}", src)
            f.set_cell(t4, i, 2, f"{f1 - full.f1:+.2f}", src)
            f.set_cell(t4, i, 3, f"{rmse:.3f}", src)
            f.set_cell(t4, i, 4, pct((rmse - full.rmse_um) / full.rmse_um * 100), src)
            continue
        if variant not in abl.index:
            for col in (1, 2, 3, 4):
                f.set_cell(t4, i, col, PENDING, "")
            continue
        f.set_cell(t4, i, 1, f"{abl.loc[variant, 'f1']:.2f}", src)
        f.set_cell(t4, i, 3, f"{abl.loc[variant, 'rmse_um']:.3f}", src)
        if variant != "full":
            f.set_cell(t4, i, 2, f"{r.delta_f1(variant):+.2f}", src)
            f.set_cell(t4, i, 4, pct(r.delta_rmse(variant)), src)
    f.set_paragraph("[The “no spatial edges” row is the cleanest test of RQ1",
                    "The “no spatial edges” row is the cleanest test of RQ1: it keeps every input and parameter of the "
                    "model except message passing between zones. "
                    + (f"Removing all spatial edges changed F1 by {r.delta_f1('no_spatial_edges'):+.2f} and Ra RMSE by "
                       f"{pct(r.delta_rmse('no_spatial_edges'))}%." if abl is not None else
                       "[Ablation stage not yet run - re-run make_report.py after it finishes.]"), src)

    # ---- Table 5 and attention --------------------------------------------------
    t5 = find_table(f.doc, "Modality")
    att = r.attribution
    ablation_for = {"Force": "without_force_features", "Acceleration": "without_acceleration_features"}
    for i, row in enumerate(t5.rows[1:], start=1):
        modality = row.cells[0].text.strip()
        if modality == "Thermal":
            for col in (1, 2, 3):
                f.set_cell(t5, i, col, NA_THERMAL, "data archive")
            continue
        for col, mat in ((1, "WAAM"), (2, "WROUGHT")):
            share = np.nan
            if att is not None:
                m = att[(att.material_type == mat) & (att.modality == modality)]
                share = float(m.share_percent.iloc[0]) if len(m) else np.nan
            f.set_cell(t5, i, col, f"{share:.1f}" if np.isfinite(share) else "n/a", src)
        if modality in ablation_for:
            f.set_cell(t5, i, 3, pct(r.delta_rmse(ablation_for[modality])) if abl is not None else PENDING, src)
        else:
            f.set_cell(t5, i, 3, "— (not ablated)", src)
    att_txt = ""
    if r.attention is not None:
        a = r.attention.set_index("material_type")
        att_txt = " ".join(f"For {m.lower()} trials the mean attention was {a.loc[m, 'kinematic_attention']:.3f} on "
                           f"kinematic and {a.loc[m, 'thermal_attention']:.3f} on thermal edges." for m in a.index)
    f.set_paragraph("[Report mean attention on thermal versus kinematic edges",
                    att_txt + " Thermal images were not available as input, so the expectation from [1] that thermal "
                    "information matters more for WAAM than for wrought material could not be tested with thermal "
                    "features; only the heat-diffusion edges, which carry no measured temperature, were evaluated. "
                    "Figure 5 shows the attention coefficients per material (outputs/figures).", src)


def fill_discussion(f: Filler, r: Results) -> None:
    src = "outputs/ (this run)"
    abl, gat, fus = r.ablation, "gat", "fusion_mlp"
    # RQ answers ------------------------------------------------------------------
    if abl is not None and "no_spatial_edges" in abl.index:
        d_f1, d_rmse = r.delta_f1("no_spatial_edges"), r.delta_rmse("no_spatial_edges")
        # Changes smaller than 0.05 in F1 or 5 % in RMSE are treated as no clear difference.
        helped = d_f1 < -0.05 or d_rmse > 5
        hurt = d_f1 > 0.05 or d_rmse < -5
        verdict = ("Message passing between zones therefore improved the predictions."
                   if helped and not hurt else
                   "Removing the edges improved the predictions, so message passing did not help on this dataset."
                   if hurt and not helped else
                   "The effect of message passing was mixed or small relative to the fold-to-fold variation.")
        rq1 = (f"Removing message passing between zones (self-loops only) changed F1 by {d_f1:+.2f} and Ra RMSE by "
               f"{pct(d_rmse)}% relative to the full ST-GAT. The graph representation preserved the predictive "
               f"information. {verdict}")
    else:
        rq1 = "[Ablation stage not yet run - re-run make_report.py after it finishes.]"
    f.set_paragraph("[Answer from Table 4, in particular", rq1, src, keep_label=True)

    ci = lambda m, k: (r.main.loc[m, f"{k}_ci_low"], r.main.loc[m, f"{k}_ci_high"])
    lo_g, hi_g = ci(gat, "roc_auc")
    lo_f, hi_f = ci(fus, "roc_auc")
    overlap = not (lo_g > hi_f or lo_f > hi_g)
    f.set_paragraph("[Answer from Tables 2 and 3.",
                    f"The ST-GAT reached an AUC of {r.ci_text(gat, 'roc_auc')} and an Ra RMSE of "
                    f"{r.ci_text(gat, 'rmse_um', 3)} µm, against {r.ci_text(fus, 'roc_auc')} and "
                    f"{r.ci_text(fus, 'rmse_um', 3)} µm for the multimodal fusion MLP. "
                    + ("The confidence intervals overlap, so with this dataset the improvement over single-modality "
                       "baselines cannot be attributed to the graph rather than to the multimodal inputs."
                       if overlap else "The confidence intervals do not overlap, which favours the graph model."),
                    src, keep_label=True)
    f.set_paragraph("[Answer from Table 5 and the edge-attention analysis",
                    "Thermal features could not be evaluated because no calibrated infrared frames were available. "
                    "Among the measured modalities, integrated gradients attributed "
                    + (", ".join(f"{row.share_percent:.0f}% to {row.modality.lower()}" for row in
                                 r.attribution[r.attribution.material_type == 'WAAM'].itertuples())
                       if r.attribution is not None else "n/a")
                    + " of the Ra prediction for WAAM trials, and "
                    + (", ".join(f"{row.share_percent:.0f}% to {row.modality.lower()}" for row in
                                 r.attribution[r.attribution.material_type == 'WROUGHT'].itertuples())
                       if r.attribution is not None else "n/a")
                    + " for wrought trials. Because the Ra targets are material means, these shares describe which "
                      "signals identify the material, and the relation to the finding of [1] (lower vibration "
                      "amplitude but higher roughness for WAAM) remains open until zone-level Ra is measured.",
                    src, keep_label=True)
    f.set_paragraph("Instability labels combine a spectral criterion and visual inspection.",
                    "Instability labels rest on a single spectral criterion applied to the full zone record; visual "
                    "inspection of the surface was not available. Any rule that depends on spectra shares information "
                    "with the model inputs; computing it at a much finer frequency resolution than the model windows "
                    "(Section 2.3) reduces but does not eliminate this dependence. All unstable zones are WAAM zones, "
                    "so instability and material are partly confounded. Ra targets are specimen means from [1], not "
                    "zone measurements.", "this run")
    f.replace("[describe how emissivity was handled]",
              "thermal images were not used as model input in the present runs, so emissivity does not affect the "
              "reported results; it must be documented if calibrated frames are added", "this run")

    # 5. Latency ---------------------------------------------------------------------
    lat = r.latency
    f.set_paragraph("For online use, inference must complete within one window hop.",
                    f"For online use, inference must complete within one window hop. With {CONFIG.signal.window_ms:g} ms "
                    f"windows and {CONFIG.signal.overlap_fraction * 100:.0f}% overlap, the budget is "
                    f"{lat['budget_ms_one_hop']:g} ms. The ST-GAT has {lat['parameters']:,} parameters and required "
                    f"{lat['ms_per_graph_one_window']:.1f} ms per graph (six zones, one window) on a laptop CPU "
                    f"(AMD Ryzen 5 5500U, {lat['threads']} threads), "
                    f"{'within' if lat['within_budget'] else 'outside'} that budget; a whole trial "
                    f"({lat['windows_per_graph']} windows) took {lat['ms_per_graph_all_windows']:.0f} ms. Because the "
                    "graph is small (N = 6 nodes), the graph layers are cheap; the modality encoders dominate the "
                    "computation.", "outputs/latency.json", keep_label=True)

    # 6. Conclusions -------------------------------------------------------------------
    f.set_paragraph("This work proposed a spatio-temporal graph formulation",
                    "This work proposed a spatio-temporal graph formulation for monitoring the milling of WAAM Al 5356 "
                    "thin walls, in which cutting zones are nodes, tool-path order and heat diffusion define the edges, "
                    "and force, vibration and thermal measurements form the node features. "
                    f"On {r.n_trials} trials with force and acceleration input, the ST-GAT detected instability with an "
                    f"AUC of {r.ci_text(gat, 'roc_auc')} against {r.ci_text(fus, 'roc_auc')} for a non-graph multimodal "
                    f"baseline, and recognised the material-level Ra with an RMSE of {r.value(gat, 'rmse_um'):.3f} µm. "
                    "Thermal information and zone-level Ra could not be evaluated with the available records, so RQ3 "
                    "remains open.", src)
    f.replace("[Add or remove items so that each corresponds to a result actually shown.]",
              "(Contribution 4, on the importance of thermal information, is not supported by the present runs and "
              "should be removed unless calibrated infrared data are added.)", "this run")
    f.replace(" [Add as needed.]", "", "this run")

    # Abstract ------------------------------------------------------------------------
    base = r.best_baseline()
    dl = r.test("DeLong AUC")
    f.replace("Cutting forces (25 kHz), triaxial acceleration and infrared thermography from [20] milling trials",
              f"Cutting forces and triaxial acceleration (both 25 kHz) from {r.n_trials} milling trials", "this run")
    f.replace("the GAT model reached [XX.X]% accuracy", f"the GAT model reached {r.value(gat, 'accuracy'):.1f}% accuracy", src)
    f.replace("ROC curve of [0.XX] for", f"ROC curve of {r.value(gat, 'roc_auc'):.2f} for", src)
    f.replace("error of [0.0XX] µm, compared with", f"error of {r.value(gat, 'rmse_um'):.3f} µm, compared with", src)
    f.replace("compared with [XX.X]%, [0.XX] and [0.0XX] µm for the strongest non-graph multimodal baseline.",
              f"compared with {r.value(base, 'accuracy'):.1f}%, {r.value(base, 'roc_auc'):.2f} and "
              f"{r.value(base if pd.notna(r.value(base, 'rmse_um')) else fus, 'rmse_um'):.3f} µm for the strongest "
              f"non-graph multimodal baseline ({pretty(base)}).", src)
    if abl is not None:
        vib = f"removing acceleration features changed Ra error by {pct(r.delta_rmse('without_acceleration_features'))}% " \
              f"and removing force features by {pct(r.delta_rmse('without_force_features'))}%"
    else:
        vib = "the vibratory ablation is reported in Table 4"
    f.replace("Removing thermal information raised Ra error by [XX]%, against [XX]% for vibratory information.",
              f"Thermal information could not be evaluated because no calibrated infrared frames were available; {vib}.",
              src)
    significant = dl is not None and pd.notna(dl.p_value) and dl.p_value < 0.05
    f.replace("[One sentence on the practical implication, written after the results are known.]",
              ("Zone-level graph modelling gave a measurable gain for instability monitoring on thin WAAM walls."
               if significant else
               "With the present data the graph structure did not significantly improve on a non-graph multimodal "
               "model, so larger datasets with measured zone-level labels are needed before the approach is deployed."),
              src)

    # Back matter ------------------------------------------------------------------------
    f.replace("[repository, DOI]", "[repository, DOI – author to supply]", "author")
    f.replace("[ ]", "[author to complete]", "author details are not in the data", count=100)
    f.replace("[tool name, version]", "Claude (Anthropic, Opus 5.5)", "this session")
    f.replace("[describe, e.g., drafting the initial structure of the text]",
              "writing the analysis code and drafting results text from the computed outputs", "this session")
    f.replace("[year]", "2025", "ref. [1] header (Materials 2025, 18)")
    f.replace("[vol]", "18", "ref. [1] header (Materials 2025, 18)")
    f.replace("[article no.]", "[article no. – check]", "ref. [1]")
    f.replace(" Every number in this section must come from your runs. Table structures and the order "
              "of reporting are set up so that the text can be completed directly. Delete this note before submission.",
              f" Every number in this section was generated by src/make_report.py from the run in "
              f"outputs/ ({r.n_trials} trials, seeds {r.info['seeds']}, Ra labels "
              f"{'IMPUTED from material means' if r.info['ra_labels_imputed'] else 'measured'}, no thermal input). "
              "Check report/fill_log.csv and delete this note before submission.", src)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(r: Results) -> None:
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colours = {"WAAM": "#c0392b", "WROUGHT": "#2471a3"}
    deep = r.pred[r.pred.variant == "full"]
    zone = (deep.groupby(["model", "trial_id", "material_type", "zone_id"], as_index=False)
            .agg(y_class=("y_class", "first"), y_reg=("y_reg", "first"),
                 probability=("probability", "mean"), ra_pred=("ra_pred", "mean")))

    # Figure 3: parity plot (ST-GAT)
    g = zone[zone.model == "gat"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for mat, sub in g.groupby("material_type"):
        ax.scatter(sub.y_reg + np.random.default_rng(0).normal(0, 0.005, len(sub)), sub.ra_pred, s=18, alpha=0.7,
                   color=colours.get(mat), label=mat.capitalize())
    lim = [min(g.y_reg.min(), g.ra_pred.min()) - 0.05, max(g.y_reg.max(), g.ra_pred.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="Target Ra (µm, material mean of [1])", ylabel="Predicted Ra (µm)", xlim=lim, ylim=lim,
           title="Figure 3 – ST-GAT parity (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_parity_gat.png", dpi=200)
    plt.close(fig)

    # Figure 4: Ra along the wall for two WAAM trials
    waam = sorted(g[g.material_type == "WAAM"].trial_id.unique())[:2] or sorted(g.trial_id.unique())[:2]
    fig, axes = plt.subplots(1, len(waam), figsize=(8, 3.2), sharey=True)
    for ax, tid in zip(np.atleast_1d(axes), waam):
        sub = g[g.trial_id == tid].sort_values("zone_id")
        pos = (sub.zone_id + 0.5) * CONFIG.process.zone_length_mm
        ax.plot(pos, sub.y_reg, "k--o", label="Target (imputed)")
        ax.plot(pos, sub.ra_pred, "-o", color=colours["WAAM"], label="ST-GAT")
        ax.set(title=tid, xlabel="Position along wall (mm)")
    np.atleast_1d(axes)[0].set_ylabel("Ra (µm)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Figure 4 – Ra along the wall")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_ra_along_wall.png", dpi=200)
    plt.close(fig)

    # Figure 5: attention on edge types
    if r.attention is not None:
        a = r.attention.set_index("material_type")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        x = np.arange(len(a))
        ax.bar(x - 0.2, a.kinematic_attention, 0.4, label="Kinematic edges", color="#7f8c8d")
        ax.bar(x + 0.2, a.thermal_attention, 0.4, label="Thermal (heat-kernel) edges", color="#e67e22")
        ax.set(xticks=x, xticklabels=[m.capitalize() for m in a.index], ylabel="Mean attention coefficient",
               title="Figure 5 – GAT attention by edge type")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig5_attention.png", dpi=200)
        plt.close(fig)

    # ROC curves of the deep models and the classical classifiers
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for model in ["Logistic regression", "Random forest", "SVM (RBF)", "cnn1d", "bilstm", "fusion_mlp", "gcn", "gat"]:
        sub = zone[(zone.model == model) & zone.probability.notna()]
        if sub.empty or sub.y_class.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub.y_class.astype(int), sub.probability)
        ax.plot(fpr, tpr, lw=2 if model == "gat" else 1, label=f"{pretty(model)} ({r.value(model, 'roc_auc'):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC – instability (out-of-fold)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse
    global OUTPUT_DIR, FILLED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR, help="folder written by train_evaluate.py")
    parser.add_argument("--out", type=Path, default=FILLED, help="filled .docx to write")
    args = parser.parse_args()
    OUTPUT_DIR, FILLED = args.results, args.out
    if not DRAFT.exists():
        raise SystemExit(f"Draft not found: {DRAFT}")
    results = Results()
    filler = Filler(docx.Document(DRAFT))
    fill_methods(filler, results)
    fill_results(filler, results)
    fill_discussion(filler, results)
    filler.doc.save(FILLED)
    log = pd.DataFrame(filler.log)
    log.to_csv(REPORT_DIR / "fill_log.csv", index=False)
    make_figures(results)

    remaining = []
    for p in filler.doc.paragraphs:
        remaining += re.findall(r"\[[^\]]*\]", p.text)
    for t in filler.doc.tables:
        for row in t.rows:
            for c in row.cells:
                remaining += re.findall(r"\[[^\]]*\]", c.text)
    remaining = [x for x in remaining if not re.fullmatch(r"\[\d[\d,–\- ]*\]", x)]   # drop citations like [1,2]
    missing = log[log.location.str.startswith("NOT FOUND")]
    print(f"Filled report written to {FILLED}")
    print(f"{len(log) - len(missing)} changes logged in {REPORT_DIR / 'fill_log.csv'}; figures in {OUTPUT_DIR / 'figures'}")
    if len(missing):
        print(f"WARNING: {len(missing)} placeholders were not found:\n{missing[['old']].to_string(index=False)}")
    print("Brackets still open (need the author or the ablation stage):")
    for x in sorted(set(remaining)):
        print("  ", x)


if __name__ == "__main__":
    main()
