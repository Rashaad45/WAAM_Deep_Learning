"""
make_report.py   -   STEP 3 of the pipeline
============================================
Fills the manuscript draft with the numbers of the dataset and of the training run,
and draws the figures.

Reads   report/WAAM_STGNN_Materials_draft.docx   (never modified)
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

How the file is organised
  ResultsLoader      reads every input table once
  DocxEditor         low-level "find this text in the Word file and replace it"
  MethodsSection     fills Section 2
  ResultsSection     fills Section 3 and Tables 1-5
  DiscussionSection  fills Sections 4-6, the abstract and the back matter
  FigureMaker        draws the PNG figures
  ReportBuilder      runs all of the above in order

How to run (from the project folder)
  python run_pipeline.py report
or directly
  python src/make_report.py
"""
import argparse
import json
import re
from pathlib import Path

import docx                              # python-docx: read and write Word files
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                    # draw into files only (no window pops up)
import matplotlib.pyplot as plt          # noqa: E402  (must come after matplotlib.use)
from sklearn.metrics import roc_curve    # noqa: E402

from config import CONFIG, CACHE_DIR, DATA_DIR, OUTPUT_DIR, REPORT_DIR   # noqa: E402

DRAFT = REPORT_DIR / "WAAM_STGNN_Materials_draft.docx"
FILLED = REPORT_DIR / "WAAM_STGNN_Materials_filled.docx"

# XML namespace of Word equations (needed to find text inside formulas).
M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

# Text used when a result does not exist (yet).
PENDING = "[pending: run the ablation stage]"
NA_THERMAL = "n/a (no calibrated IR)"

# Row label in the Word tables  ->  model name used in outputs/metrics_main.csv
MODEL_ROWS = {"Logistic regression": "Logistic regression", "Random forest": "Random forest",
              "SVM (RBF)": "SVM (RBF)", "1D-CNN": "cnn1d", "BiLSTM": "bilstm",
              "Multimodal fusion MLP (no graph)": "fusion_mlp", "ST-GCN (this work)": "gcn",
              "ST-GAT (this work)": "gat", "Ridge regression": "Ridge regression",
              "Random forest regressor": "Random forest regressor", "SVR (RBF)": "SVR (RBF)",
              "Predict training mean (reference)": "Predict training mean (reference)"}


# ===========================================================================
# Small text helpers
# ===========================================================================
def pretty(model: str) -> str:
    """Short code name -> name used in the text, e.g. "gat" -> "ST-GAT"."""
    names = {"cnn1d": "1D-CNN", "bilstm": "BiLSTM", "fusion_mlp": "multimodal fusion MLP",
             "gcn": "ST-GCN", "gat": "ST-GAT"}
    return names.get(model, model)


def pct(x: float, digits: int = 1) -> str:
    """Number with a sign, e.g. 3.21 -> "+3.2"; NaN -> "n/a"."""
    return "n/a" if not np.isfinite(x) else f"{x:+.{digits}f}"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """pd.read_csv, but returns None instead of failing when the file is missing."""
    return pd.read_csv(path) if path.exists() else None


# ===========================================================================
# 1. Loading all results
# ===========================================================================
class ResultsLoader:
    """Everything the report needs, loaded once from data/ and outputs/."""

    def __init__(self, results_dir: Path):
        # --- dataset --------------------------------------------------------
        self.summary = pd.read_csv(CACHE_DIR / "dataset_summary.csv")
        self.labels = pd.read_csv(DATA_DIR / "zone_labels.csv")
        self.included = self.summary[self.summary.included]     # only usable trials

        # --- main results (required) --------------------------------------
        self.main = read_csv_if_exists(results_dir / "metrics_main.csv")
        if self.main is None:
            raise SystemExit("outputs/metrics_main.csv not found - run train_evaluate.py --stage main first.")
        self.main = self.main.set_index("model")                # look up rows by model name

        # --- optional results ---------------------------------------------
        self.by_material = read_csv_if_exists(results_dir / "metrics_by_material.csv")
        self.ablation = read_csv_if_exists(results_dir / "ablation.csv")
        if self.ablation is not None:
            self.ablation = self.ablation.set_index("variant")
        self.attribution = read_csv_if_exists(results_dir / "attribution.csv")
        self.attention = read_csv_if_exists(results_dir / "attention.csv")
        self.tests = read_csv_if_exists(results_dir / "stat_tests.csv")
        self.loco = read_csv_if_exists(results_dir / "loco" / "metrics_main.csv")

        # --- run details ----------------------------------------------------
        self.latency = json.loads((results_dir / "latency.json").read_text())
        self.info = json.loads((results_dir / "run_info.json").read_text())
        self.pred = pd.read_csv(results_dir / "predictions_oof.csv")

    # -----------------------------------------------------------------------
    # Convenience look-ups used by the section writers
    # -----------------------------------------------------------------------
    @property
    def n_trials(self) -> int:
        return len(self.included)

    def trials_of(self, material: str) -> int:
        """Number of usable trials of one material ("WAAM" or "WROUGHT")."""
        return int((self.included.material_type == material).sum())

    def value(self, model: str, metric: str) -> float:
        """One number from metrics_main.csv, or NaN if the model is missing."""
        return float(self.main.loc[model, metric]) if model in self.main.index else np.nan

    def ci_text(self, model: str, metric: str, digits: int = 2, table=None) -> str:
        """Value with its confidence interval, e.g. "0.91 (0.85–0.97)"."""
        table = self.main if table is None else table
        if model not in table.index or metric not in table.columns or pd.isna(table.loc[model, metric]):
            return "n/a"
        value = table.loc[model, metric]
        low = table.loc[model, f"{metric}_ci_low"]
        high = table.loc[model, f"{metric}_ci_high"]
        fmt = f"{{:.{digits}f}}"                         # e.g. "{:.2f}"
        if pd.notna(low):
            return f"{fmt.format(value)} ({fmt.format(low)}–{fmt.format(high)})"
        return fmt.format(value)

    def test(self, name: str) -> pd.Series | None:
        """One row of stat_tests.csv (e.g. "DeLong AUC"), or None."""
        if self.tests is None or self.tests.empty:
            return None
        rows = self.tests[self.tests.test == name]
        return rows.iloc[0] if len(rows) else None

    def has_ablation(self, variant: str) -> bool:
        return self.ablation is not None and variant in self.ablation.index and "full" in self.ablation.index

    def delta_rmse(self, variant: str) -> float:
        """Change of Ra RMSE in % when going from the full model to an ablation."""
        if not self.has_ablation(variant):
            return np.nan
        full = self.ablation.loc["full", "rmse_um"]
        return float((self.ablation.loc[variant, "rmse_um"] - full) / full * 100)

    def delta_f1(self, variant: str) -> float:
        """Change of F1 (absolute) when going from the full model to an ablation."""
        if not self.has_ablation(variant):
            return np.nan
        return float(self.ablation.loc[variant, "f1"] - self.ablation.loc["full", "f1"])

    def best_baseline(self) -> str:
        """Strongest non-graph model = best AUC among the non-graph learned models."""
        candidates = ["fusion_mlp", "bilstm", "cnn1d", "Logistic regression", "Random forest", "SVM (RBF)"]
        aucs = {m: self.value(m, "roc_auc") for m in candidates if m in self.main.index}
        aucs = {m: a for m, a in aucs.items() if np.isfinite(a)}
        return max(aucs, key=aucs.get) if aucs else "fusion_mlp"


# ===========================================================================
# 2. Editing the Word file
# ===========================================================================
class DocxEditor:
    """Finds text in the Word document and replaces it, keeping formatting and equations.

    Word stores a paragraph as several "runs" (pieces of text with the same
    formatting). We change the text of runs instead of the whole paragraph so
    that bold, italic and equations stay intact. Every change is recorded in
    self.log, which becomes report/fill_log.csv.
    """

    def __init__(self, document: docx.Document):
        self.doc = document
        self.log = []

    def _all_paragraphs(self):
        """Every paragraph of the body AND of every table cell."""
        yield from self.doc.paragraphs
        for table in self.doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    yield from cell.paragraphs

    def _record(self, location: str, old: str, new: str, source: str) -> None:
        self.log.append(dict(location=location, old=old, new=new, source=source))

    def replace(self, old: str, new: str, source: str, count: int = 1) -> None:
        """Replace `old` inside a single run (the first `count` times it appears)."""
        done = 0
        for paragraph in self._all_paragraphs():
            for run in paragraph.runs:
                while old in run.text and done < count:
                    run.text = run.text.replace(old, new, 1)
                    done += 1
                    self._record(paragraph.text[:60], old, new, source)
            if done >= count:
                return
        if done == 0:
            self._record("NOT FOUND", old, new, source)

    def set_paragraph(self, contains: str, text: str, source: str, keep_label: bool = False) -> None:
        """Replace the whole text of the first paragraph that contains `contains`.

        keep_label=True keeps the first run (e.g. a bold "RQ1." label) and replaces the rest.
        Equations inside the paragraph are removed.
        """
        for paragraph in self.doc.paragraphs:
            if contains not in paragraph.text:
                continue
            old = paragraph.text
            runs = paragraph.runs
            first = 1 if keep_label and len(runs) > 1 else 0
            runs[first].text = (" " if keep_label else "") + text
            for run in runs[first + 1:]:                          # delete the other runs
                run._r.getparent().remove(run._r)
            for equation in paragraph._p.findall(f".//{M_NS}oMath"):
                equation.getparent().remove(equation)
            self._record(old[:60], old, text, source)
            return
        self._record("NOT FOUND", contains, text, source)

    def set_math(self, old: str, new: str, source: str) -> None:
        """Replace text inside an equation."""
        for paragraph in self.doc.paragraphs:
            for t in paragraph._p.iter(f"{M_NS}t"):
                if t.text and old in t.text:
                    t.text = t.text.replace(old, new)
                    self._record(paragraph.text[:60], old, new, source)
                    return
        self._record("NOT FOUND (math)", old, new, source)

    def set_cell(self, table, row: int, col: int, text: str, source: str) -> None:
        """Write `text` into one table cell."""
        cell = table.rows[row].cells[col]
        paragraph = cell.paragraphs[0]
        old = cell.text
        if paragraph.runs:
            paragraph.runs[0].text = text
            for run in paragraph.runs[1:]:
                run._r.getparent().remove(run._r)
        else:
            paragraph.add_run(text)
        self._record(f"table '{table.rows[0].cells[0].text}' r{row}c{col}", old, text, source)

    def find_table(self, first_header: str, second_header: str = ""):
        """The table whose first header cell is `first_header` (and that has `second_header`)."""
        for table in self.doc.tables:
            headers = [c.text.strip() for c in table.rows[0].cells]
            if headers[0] == first_header and (not second_header or second_header in headers):
                return table
        raise KeyError(first_header)


# ===========================================================================
# 3. Section 2 - Materials and methods
# ===========================================================================
class MethodsSection:
    SRC_CFG = "src/config.py"
    SRC_DATA = "raw files in WAAM/ (measured)"
    SRC_PAPER = "ref. [1]"

    def __init__(self, editor: DocxEditor, results: ResultsLoader):
        self.ed = editor
        self.r = results

    def fill(self) -> None:
        self.machining_and_sensing()
        self.labels()
        self.features()
        self.evaluation_protocol()
        self.implementation()

    # --- 2.2 Machining and sensing ---------------------------------------
    def machining_and_sensing(self) -> None:
        ed, r, sig = self.ed, self.r, CONFIG.signal
        s = r.included
        weak = s[s.accel_alignment_r < CONFIG.cut.min_alignment_r]
        excluded = r.summary[~r.summary.included]

        exclusion_list = "; ".join(f"{t.trial_id}: {t.exclusion_reason}" for t in excluded.itertuples())
        ed.replace("giving 10 trials per material and 20 in total [1].",
                   f"giving 10 trials per material and 20 in total [1]. Synchronised force and acceleration records "
                   f"exist for trials 4–20 of the archive; {r.n_trials} of them ({r.trials_of('WAAM')} WAAM, "
                   f"{r.trials_of('WROUGHT')} wrought) were usable (trials 1–3 have no force record; "
                   + exclusion_list + ").",
                   self.SRC_DATA)

        ed.set_paragraph("Triaxial accelerometer [model]",
                         "Triaxial accelerometer (model and mounting position not recorded in the data archive – "
                         "[author to supply]), acquired with the KiNOVA data-acquisition software and sampled at "
                         f"{sig.acceleration_fs_hz / 1000:g} kHz (measured from the time column of every file). "
                         "Force and acceleration were therefore recorded at the same rate and no resampling was needed "
                         f"for synchronisation; for the CNN encoders both streams were decimated by "
                         f"{sig.encoder_decimation} "
                         f"to {sig.force_fs_hz / sig.encoder_decimation / 1000:g} kHz with an anti-aliasing polyphase "
                         "filter "
                         "(Nyquist 6.25 kHz, above both structural modes). All three axes are exported in g by the "
                         "acquisition software, which was taken as the common unit reference.",
                         self.SRC_DATA, keep_label=True)

        ed.set_paragraph("Infrared camera [model]",
                         "Infrared records were taken with a Sonel thermal imager (radiometric .irgd files, 384 × 288 "
                         "pixels; exact model, frame rate and emissivity setting not recorded in the archive – "
                         "[author to confirm]). The .irgd files could not be exported to calibrated temperature frames "
                         "without the vendor software, so thermal images were NOT used as model input in the runs "
                         "reported here; the thermal branch of the network is disabled and only the heat-diffusion "
                         "edges (which depend on geometry and alloy properties, not on images) are retained.",
                         self.SRC_DATA, keep_label=True)

        weak_note = (f"; for {', '.join(weak.trial_id)} the accelerometer barely responded to the cut and the "
                     "alignment is less certain." if len(weak) else ".")
        ed.set_paragraph("Streams were aligned in time using",
                         "The force and acceleration PCs were not clock-synchronised, so streams were aligned by "
                         "matching the RMS envelope of the acceleration (10 ms bins) to a step template that marks the "
                         "cut in the force record; only lags for which the vibration record contains the whole cut were "
                         "searched. "
                         f"The correlation at the selected lag was r = {s.accel_alignment_r.min():.2f}–"
                         f"{s.accel_alignment_r.max():.2f} (median {s.accel_alignment_r.median():.2f}) and the lags "
                         f"were {s.accel_lag_s.min():+.2f} to {s.accel_lag_s.max():+.2f} s. The residual alignment "
                         "uncertainty is about ±10 ms (one envelope bin)" + weak_note,
                         self.SRC_DATA)

    # --- 2.3 Labels ----------------------------------------------------------
    def labels(self) -> None:
        ed, sig, lc = self.ed, CONFIG.signal, CONFIG.labels
        lab = self.r.labels
        n_unstable, n_total = int(lab.chatter_label.sum()), len(lab)
        by_material = lab.groupby("material_type").chatter_label.agg(["sum", "count"])
        per_material = ", ".join(f"{int(v['sum'])}/{int(v['count'])} for {m.lower()}"
                                 for m, v in by_material.iterrows())

        ed.set_paragraph("Each zone in each trial was labelled stable or unstable",
                         "Each zone in each trial was labelled stable or unstable. Because the model receives spectral "
                         "features, a label defined from the same short-window spectra would make high accuracy partly "
                         "circular. Labels were therefore computed from the full-duration force record of each zone "
                         f"(≈{CONFIG.zone_duration_s:.2f} s, Welch PSD with {lc.label_nperseg} points, ≈0.76 Hz "
                         f"resolution, against ≈24 Hz in the {sig.window_ms:g} ms model windows) with a rule fixed "
                         "before model training: "
                         "a zone is unstable when the dominant peak of the summed three-axis force PSD in the "
                         f"{sig.chatter_band_hz[0] / 1000:g}–{sig.chatter_band_hz[1] / 1000:g} kHz band does not lie "
                         f"within ±{sig.harmonic_tolerance_hz:g} Hz of a spindle-rotation harmonic and rises at least "
                         f"{lc.peak_prominence_db:g} dB above the band median. Visual inspection of the machined "
                         "surface was not available for this dataset (no per-zone surface images in the archive), so "
                         "the labels rest on this single criterion and inter-annotator agreement does not apply. The "
                         f"rule labelled {n_unstable} of {n_total} zones unstable "
                         f"({n_unstable / n_total * 100:.1f}%): " + per_material + ".",
                         "data/zone_labels.csv", keep_label=True)

        ed.set_paragraph("Ra was measured with [instrument]",
                         "Zone-level Ra measurements are not available in the data archive. The earlier study [1] "
                         "reports Ra as specimen means over three positions (left, centre, right): 1.22 µm for WAAM "
                         "and 0.83 µm for wrought material. In the runs reported here every zone was assigned the mean "
                         "of its specimen (IMPUTED), so the Ra head can only learn the material difference and Ra "
                         "errors must not be read as zone-level prediction errors. Measured zone values (instrument, "
                         "ISO 21920-2 cut-off and evaluation length, positions per zone) can be entered in "
                         "data/ra_measurements.csv, after which the pipeline uses them automatically. Measurement "
                         "repeatability σ_meas is likewise not available.",
                         self.SRC_PAPER, keep_label=True)

    # --- 2.4 Features ---------------------------------------------------------
    def features(self) -> None:
        ed, r, sig = self.ed, self.r, CONFIG.signal
        band = f"{sig.chatter_band_hz[0] / 1000:g}–{sig.chatter_band_hz[1] / 1000:g} kHz"
        ed.replace("(here [2.5–5.5] kHz)", f"(here {band})", self.SRC_CFG)

        half_bin = sig.force_fs_hz / sig.welch_segment / 2
        ed.replace(" is [X] Hz.", f" is {half_bin:.1f} Hz (half a Welch bin of the {sig.welch_segment}-point "
                                  "segments, around each tooth-passing harmonic).", self.SRC_CFG)

        w = sig.window_ms
        ed.replace("Window length was [X] ms with [50]% overlap, which spans [X] tooth periods",
                   f"Window length was {w:g} ms with {sig.overlap_fraction * 100:.0f}% overlap "
                   f"({CONFIG.windows_per_zone} windows per zone), which spans {w / 15:.1f} tooth periods",
                   self.SRC_CFG)
        tooth_period_1400_ms = 60 / 1400 / 4 * 1000
        ed.replace("and [X] at 1400 rpm", f"and {w / tooth_period_1400_ms:.1f} at 1400 rpm", self.SRC_CFG)
        ed.replace("dividing the [30] mm tool path", "dividing the 30 mm tool path", self.SRC_PAPER)
        ed.replace("segments of [5] mm.", "segments of 5 mm (in practice the detected cut of each force record, "
                                          f"{r.included.cut_duration_s.min():.1f}–"
                                          f"{r.included.cut_duration_s.max():.1f} s long, "
                                          "was divided into six equal parts).", self.SRC_DATA)
        ed.replace(" [values to be checked against the alloy datasheet; the WAAM deposit may differ]",
                   " (datasheet values for wrought Al 5356; the WAAM deposit may differ)", "alloy datasheet")
        ed.replace("and a small 2D-CNN over the thermal region of interest.",
                   "and a small 2D-CNN over the thermal region of interest (disabled in the runs reported here, since "
                   "no calibrated thermal frames were available).", self.SRC_DATA)
        ed.replace("[Report which variant was retained and the selected λ.]",
                   f"The fixed-weight loss with λ = {CONFIG.train.regression_weight:g} was retained; the uncertainty "
                   "weighting was not used in the reported runs.", self.SRC_CFG)

    # --- 2.9 Evaluation protocol ---------------------------------------------------
    def evaluation_protocol(self) -> None:
        ed, n = self.ed, self.r.n_trials
        ed.replace("cross-validation (20 folds)", f"cross-validation ({n} folds)", self.SRC_DATA)
        ed.replace("on the remaining 19 trials", f"on the remaining {n - 1} trials (one WAAM and one wrought trial "
                                                 "held out for early stopping)", self.SRC_DATA)
        ed.replace(" [Optionally: train on wrought, test on WAAM, to probe transfer across materials.]",
                   " Transfer from wrought to WAAM material was not evaluated.", "this run")
        ed.replace(". Metrics are given as mean ± standard deviation over folds with 95% bootstrap confidence "
                   "intervals.",
                   ". Because each test fold contains only six zones, metrics are computed on the pooled out-of-fold "
                   "predictions and given with 95% confidence intervals from 2000 bootstrap resamples of trials.",
                   "this run")
        ed.replace("paired differences in fold-level F1 and RMSE with the ",
                   "paired differences in fold-level RMSE with the ", "this run")
        ed.replace("With 20 trials the statistical power", f"With {n} trials the statistical power", self.SRC_DATA)
        ed.replace(", and GNNExplainer [36] is used to identify the subgraphs most responsible for high-roughness "
                   "predictions.",
                   ". (GNNExplainer [36] was not applied in the present runs.)", "this run")

    # --- 2.11 Implementation ---------------------------------------------------------
    def implementation(self) -> None:
        info, t = self.r.info, CONFIG.train
        torch_version = info["versions"]["torch"].split("+")[0]          # "2.6.0+cu124" -> "2.6.0"
        seeds = ", ".join(map(str, info["seeds"]))
        self.ed.set_paragraph(
            "Models were implemented in",
            f"Models were implemented in PyTorch {torch_version} (graph convolution "
            "and attention layers written directly in PyTorch; PyTorch Geometric was not required for "
            f"six-node graphs) and trained with Adam (learning rate {t.learning_rate:g}, weight decay "
            f"{t.weight_decay:g}), batch size 1 graph, for up to {info['max_epochs']} epochs with early "
            f"stopping on the inner validation loss (patience {info['patience']}). Hidden dimension "
            f"{t.hidden_dim}, {t.attention_heads} attention heads, temporal kernel K_t = {t.temporal_kernel}, "
            f"dropout {t.dropout:g}. Random seeds were fixed and each configuration was run with "
            f"{len(info['seeds'])} seed(s) ({seeds}). Training and inference "
            f"latency were measured on the CPU of a laptop (AMD Ryzen 5 5500U, 6 cores, no GPU) (Section 5). "
            "Code and processed data are available at [repository – author to supply].",
            "outputs/run_info.json")


# ===========================================================================
# 4. Section 3 - Results (and Tables 1-5)
# ===========================================================================
class ResultsSection:
    SRC = "outputs/ (this run)"

    def __init__(self, editor: DocxEditor, results: ResultsLoader):
        self.ed = editor
        self.r = results

    def fill(self) -> None:
        self.table1_dataset()
        self.class_balance_text()
        self.table2_classification()
        self.table3_regression()
        self.table4_ablations()
        self.table5_attribution_and_attention()

    # --- Table 1: dataset ----------------------------------------------------------
    def table1_dataset(self) -> None:
        ed, r = self.ed, self.r
        lab = r.labels
        t1 = ed.find_table("Quantity")
        T = int(r.included.windows_per_zone.iloc[0])                       # windows per zone
        counts = {m: r.trials_of(m) for m in ("WAAM", "WROUGHT")}
        for col, material in ((1, "WAAM"), (2, "WROUGHT")):
            sub = lab[lab.material_type == material]
            n_unstable = int(sub.chatter_label.sum())
            n_stable = int((sub.chatter_label == 0).sum())
            ed.set_cell(t1, 1, col, f"10 ({counts[material]} usable)", "data/processed/dataset_summary.csv")
            ed.set_cell(t1, 2, col, "6", "src/config.py")
            ed.set_cell(t1, 3, col, str(T), "src/config.py")
            ed.set_cell(t1, 4, col, str(counts[material] * T), self.SRC)
            ed.set_cell(t1, 5, col, f"{n_unstable} / {n_stable}", "data/zone_labels.csv")
            ed.set_cell(t1, 6, col, f"0 (mean of [1] imputed for {len(sub)})", "data/zone_labels.csv")
            ra = CONFIG.labels.ra_material_mean_um[material]
            ed.set_cell(t1, 7, col, f"{ra:.2f} (imputed, single value)", "ref. [1]")
            ed.set_cell(t1, 8, col, "not available", "data archive")
        # Column 3 = total.
        ed.set_cell(t1, 1, 3, f"20 ({r.n_trials} usable)", "data/processed/dataset_summary.csv")
        ed.set_cell(t1, 4, 3, str(r.n_trials * T), self.SRC)
        ed.set_cell(t1, 5, 3, f"{int(lab.chatter_label.sum())} / {int((lab.chatter_label == 0).sum())}",
                    "data/zone_labels.csv")
        ed.set_cell(t1, 6, 3, f"0 / {len(lab)}", "data/zone_labels.csv")

    # --- 3.1 class balance --------------------------------------------------------------
    def class_balance_text(self) -> None:
        r, lab = self.r, self.r.labels
        per_trial = lab.groupby(["trial_id", "material_type"]).chatter_label.sum().reset_index()
        summary = r.included.set_index("trial_id")
        unstable_trials = per_trial[per_trial.chatter_label > 0]
        description = "; ".join(f"{t.trial_id} ({int(summary.loc[t.trial_id, 'spindle_speed_rpm'])} rpm, "
                                f"{summary.loc[t.trial_id, 'depth_of_cut_mm']:.2f} mm): {int(t.chatter_label)}/6"
                                for t in unstable_trials.itertuples())
        wrought_unstable = int(lab[lab.material_type == "WROUGHT"].chatter_label.sum())
        confound_note = ("No wrought trial was labelled unstable, so every unstable zone is a WAAM zone; a "
                         "classifier can therefore partly separate the classes through material-related signal "
                         "differences, and the classification results below must be read with this confound in "
                         "mind. " if wrought_unstable == 0 else "")
        self.ed.set_paragraph(
            "[Describe the class balance",
            f"The dataset is strongly imbalanced: {int(lab.chatter_label.sum())} of {len(lab)} zones "
            f"({lab.chatter_label.mean() * 100:.1f}%) were labelled unstable, in {len(unstable_trials)} trials "
            f"({description}). " + confound_note
            + "Ra targets are the material means of [1] (Table 1), so Ra results measure how well each model "
              "recognises the material from the signals, not zone-level roughness.",
            "data/zone_labels.csv")

    # --- Table 2 + text 3.2: classification ------------------------------------------------
    def table2_classification(self) -> None:
        ed, r = self.ed, self.r
        lab = r.labels
        t2 = ed.find_table("Model", "ROC-AUC")
        for i, row in enumerate(t2.rows[1:], start=1):                       # skip the header row
            key = MODEL_ROWS.get(row.cells[0].text.strip())
            ed.set_cell(t2, i, 2, r.ci_text(key, "accuracy", 1), self.SRC)
            for col, metric in ((3, "precision"), (4, "recall"), (5, "f1"), (6, "roc_auc")):
                ed.set_cell(t2, i, col, r.ci_text(key, metric, 2), self.SRC)
        ed.replace("(mean ± SD over 20 folds)", f"(pooled out-of-fold predictions of {r.n_trials} folds; 95% "
                                                "trial-bootstrap confidence interval in parentheses)", self.SRC)

        gat, fus = "gat", "fusion_mlp"
        delong = r.test("DeLong AUC")
        margin = r.value(gat, "roc_auc") - r.value(fus, "roc_auc")
        with_auc = [m for m in r.main.index if pd.notna(r.main.loc[m, "roc_auc"])]
        best = max(with_auc, key=lambda m: r.main.loc[m, "roc_auc"]) if with_auc else gat

        # Sentence about leave-one-condition-out (only if that run exists).
        loco_text = ""
        if r.loco is not None:
            loco = r.loco.set_index("model")
            if gat in loco.index and fus in loco.index:
                loco_text = (f" Under leave-one-condition-out validation (Table S1) the AUC was "
                             f"{loco.loc[gat, 'roc_auc']:.2f} for the ST-GAT and {loco.loc[fus, 'roc_auc']:.2f} "
                             "for the fusion MLP.")
        else:
            loco_text = " The leave-one-condition-out protocol (Table S1) was not run for this draft."

        delong_text = (f", DeLong p = {delong.p_value:.3f}"
                       if delong is not None and pd.notna(delong.p_value) else "")
        ed.set_paragraph(
            "[Text: state the best model",
            f"The highest ROC-AUC was obtained by the {pretty(best)} ({r.ci_text(best, 'roc_auc')}). "
            f"The ST-GAT reached an AUC of {r.ci_text(gat, 'roc_auc')} against {r.ci_text(fus, 'roc_auc')} for the "
            f"multimodal fusion MLP without graph structure (difference {margin:+.2f}" + delong_text + ")."
            + loco_text +
            f" Recall, the share of unstable zones that were detected, was {r.ci_text(gat, 'recall')} for the "
            f"ST-GAT and {r.ci_text(fus, 'recall')} for the fusion MLP; with only "
            f"{int(lab.chatter_label.sum())} unstable zones these intervals are wide, and accuracy is dominated "
            "by the stable majority class.",
            self.SRC)

    # --- Table 3 + text 3.3: Ra regression ---------------------------------------------------
    def table3_regression(self) -> None:
        ed, r = self.ed, self.r
        t3 = ed.find_table("Model", "RMSE / σ_meas")
        for i, row in enumerate(t3.rows[1:], start=1):
            key = MODEL_ROWS.get(row.cells[0].text.strip())
            ed.set_cell(t3, i, 1, r.ci_text(key, "rmse_um", 3), self.SRC)
            ed.set_cell(t3, i, 2, r.ci_text(key, "mae_um", 3), self.SRC)
            if row.cells[3].text.strip() != "0":
                ed.set_cell(t3, i, 3, r.ci_text(key, "r2", 2), self.SRC)
            ed.set_cell(t3, i, 4, "n/a (σ_meas unavailable)", "data archive")
        ed.replace(" Zone-level Ra prediction under leave-one-trial-out cross-validation.",
                   " Zone-level Ra prediction under leave-one-trial-out cross-validation (pooled out-of-fold; "
                   "Ra targets are material means of [1], see Section 2.3).", self.SRC)

        reference = "Predict training mean (reference)"
        mean_rmse = r.value(reference, "rmse_um")
        with_rmse = [m for m in r.main.index if pd.notna(r.main.loc[m, "rmse_um"]) and m != reference]
        best_ra = min(with_rmse, key=lambda m: r.main.loc[m, "rmse_um"]) if with_rmse else "gat"
        ed.set_paragraph(
            "[Text: include the trivial mean predictor",
            f"Predicting the training-set mean gives an RMSE of {mean_rmse:.3f} µm, the size of the error "
            f"when the material is ignored. The lowest RMSE, {r.ci_text(best_ra, 'rmse_um', 3)} µm, was obtained "
            f"by the {pretty(best_ra)}; the ST-GAT reached {r.ci_text('gat', 'rmse_um', 3)} µm and the fusion MLP "
            f"{r.ci_text('fusion_mlp', 'rmse_um', 3)} µm. Because every zone of a specimen carries the same imputed "
            "Ra, these errors measure material recognition; they do not show whether errors concentrate at the "
            "end zones, which requires measured zone-level Ra. Figure 3 shows the parity plot coloured by "
            "material and Figure 4 the predicted Ra along the wall for two WAAM trials "
            "(outputs/figures).",
            self.SRC)

    # --- Table 4: ablations -----------------------------------------------------------------
    def table4_ablations(self) -> None:
        ed, r = self.ed, self.r
        abl = r.ablation
        t4 = ed.find_table("Configuration")
        # Row label in the Word table -> variant name (None = thermal, not available).
        variant_rows = {"Full model": "full", "Without thermal features": None,
                        "Without force features": "without_force_features",
                        "Without acceleration features": "without_acceleration_features",
                        "Without thermal edges (kinematic only)": "without_thermal_edges",
                        "Without kinematic edges (thermal only)": "without_kinematic_edges",
                        "No spatial edges (self-loops only)": "no_spatial_edges",
                        "Without temporal module": "without_temporal_module",
                        "With material flag as feature": "material_flag",
                        "Single-task (instability only / Ra only)": "single"}
        # Columns: 1 = F1, 2 = change of F1, 3 = RMSE, 4 = change of RMSE (%).
        for i, row in enumerate(t4.rows[1:], start=1):
            variant = variant_rows.get(row.cells[0].text.strip(), "?")

            if variant is None:                                     # thermal features: no data
                for col in (1, 2, 3, 4):
                    ed.set_cell(t4, i, col, NA_THERMAL, "data archive")
                continue

            if abl is None:                                         # ablation stage not run yet
                for col in (1, 2, 3, 4):
                    text = "—" if (variant == "full" and col in (2, 4)) else PENDING
                    ed.set_cell(t4, i, col, text, "")
                continue

            if variant == "single":                                 # two single-task models in one row
                f1 = abl.loc["classification_only", "f1"] if "classification_only" in abl.index else np.nan
                rmse = abl.loc["roughness_only", "rmse_um"] if "roughness_only" in abl.index else np.nan
                full = abl.loc["full"]
                ed.set_cell(t4, i, 1, f"{f1:.2f}", self.SRC)
                ed.set_cell(t4, i, 2, f"{f1 - full.f1:+.2f}", self.SRC)
                ed.set_cell(t4, i, 3, f"{rmse:.3f}", self.SRC)
                ed.set_cell(t4, i, 4, pct((rmse - full.rmse_um) / full.rmse_um * 100), self.SRC)
                continue

            if variant not in abl.index:
                for col in (1, 2, 3, 4):
                    ed.set_cell(t4, i, col, PENDING, "")
                continue

            ed.set_cell(t4, i, 1, f"{abl.loc[variant, 'f1']:.2f}", self.SRC)
            ed.set_cell(t4, i, 3, f"{abl.loc[variant, 'rmse_um']:.3f}", self.SRC)
            if variant != "full":
                ed.set_cell(t4, i, 2, f"{r.delta_f1(variant):+.2f}", self.SRC)
                ed.set_cell(t4, i, 4, pct(r.delta_rmse(variant)), self.SRC)

        if abl is not None:
            finding = (f"Removing all spatial edges changed F1 by {r.delta_f1('no_spatial_edges'):+.2f} and Ra "
                       f"RMSE by {pct(r.delta_rmse('no_spatial_edges'))}%.")
        else:
            finding = "[Ablation stage not yet run - re-run make_report.py after it finishes.]"
        ed.set_paragraph("[The “no spatial edges” row is the cleanest test of RQ1",
                         "The “no spatial edges” row is the cleanest test of RQ1: it keeps every input and parameter "
                         "of the model except message passing between zones. " + finding,
                         self.SRC)

    # --- Table 5 + attention text -------------------------------------------------------------
    def table5_attribution_and_attention(self) -> None:
        ed, r = self.ed, self.r
        att, abl = r.attribution, r.ablation
        t5 = ed.find_table("Modality")
        ablation_for = {"Force": "without_force_features", "Acceleration": "without_acceleration_features"}
        for i, row in enumerate(t5.rows[1:], start=1):
            modality = row.cells[0].text.strip()
            if modality == "Thermal":
                for col in (1, 2, 3):
                    ed.set_cell(t5, i, col, NA_THERMAL, "data archive")
                continue
            # Columns 1 and 2: attribution share for WAAM and for wrought.
            for col, material in ((1, "WAAM"), (2, "WROUGHT")):
                share = np.nan
                if att is not None:
                    match = att[(att.material_type == material) & (att.modality == modality)]
                    share = float(match.share_percent.iloc[0]) if len(match) else np.nan
                ed.set_cell(t5, i, col, f"{share:.1f}" if np.isfinite(share) else "n/a", self.SRC)
            # Column 3: RMSE change when this sensor is removed.
            if modality in ablation_for:
                text = pct(r.delta_rmse(ablation_for[modality])) if abl is not None else PENDING
                ed.set_cell(t5, i, 3, text, self.SRC)
            else:
                ed.set_cell(t5, i, 3, "— (not ablated)", self.SRC)

        attention_text = ""
        if r.attention is not None:
            a = r.attention.set_index("material_type")
            attention_text = " ".join(f"For {m.lower()} trials the mean attention was "
                                      f"{a.loc[m, 'kinematic_attention']:.3f} on kinematic and "
                                      f"{a.loc[m, 'thermal_attention']:.3f} on thermal edges." for m in a.index)
        ed.set_paragraph("[Report mean attention on thermal versus kinematic edges",
                         attention_text + " Thermal images were not available as input, so the expectation from [1] "
                         "that thermal information matters more for WAAM than for wrought material could not be "
                         "tested with thermal features; only the heat-diffusion edges, which carry no measured "
                         "temperature, were evaluated. Figure 5 shows the attention coefficients per material "
                         "(outputs/figures).",
                         self.SRC)


# ===========================================================================
# 5. Sections 4-6, abstract and back matter
# ===========================================================================
class DiscussionSection:
    SRC = "outputs/ (this run)"

    def __init__(self, editor: DocxEditor, results: ResultsLoader):
        self.ed = editor
        self.r = results

    def fill(self) -> None:
        self.research_questions()
        self.limitations()
        self.latency()
        self.conclusions()
        self.abstract()
        self.back_matter()

    def attribution_list(self, material: str) -> str:
        """e.g. "55% to force, 40% to acceleration, 5% to kinematic / geometric"."""
        if self.r.attribution is None:
            return "n/a"
        rows = self.r.attribution[self.r.attribution.material_type == material]
        return ", ".join(f"{row.share_percent:.0f}% to {row.modality.lower()}" for row in rows.itertuples())

    # --- Answers to RQ1-RQ3 ----------------------------------------------------------
    def research_questions(self) -> None:
        ed, r, abl = self.ed, self.r, self.r.ablation
        gat, fus = "gat", "fusion_mlp"

        # RQ1: does message passing help?
        if abl is not None and "no_spatial_edges" in abl.index:
            d_f1, d_rmse = r.delta_f1("no_spatial_edges"), r.delta_rmse("no_spatial_edges")
            # Changes smaller than 0.05 in F1 or 5 % in RMSE are treated as no clear difference.
            helped = d_f1 < -0.05 or d_rmse > 5
            hurt = d_f1 > 0.05 or d_rmse < -5
            if helped and not hurt:
                verdict = "Message passing between zones therefore improved the predictions."
            elif hurt and not helped:
                verdict = "Removing the edges improved the predictions, so message passing did not help on this dataset."
            else:
                verdict = "The effect of message passing was mixed or small relative to the fold-to-fold variation."
            rq1 = (f"Removing message passing between zones (self-loops only) changed F1 by {d_f1:+.2f} and Ra RMSE "
                   f"by {pct(d_rmse)}% relative to the full ST-GAT. The graph representation preserved the "
                   f"predictive information. {verdict}")
        else:
            rq1 = "[Ablation stage not yet run - re-run make_report.py after it finishes.]"
        ed.set_paragraph("[Answer from Table 4, in particular", rq1, self.SRC, keep_label=True)

        # RQ2: graph model vs non-graph multimodal model.
        lo_gat, hi_gat = r.main.loc[gat, "roc_auc_ci_low"], r.main.loc[gat, "roc_auc_ci_high"]
        lo_fus, hi_fus = r.main.loc[fus, "roc_auc_ci_low"], r.main.loc[fus, "roc_auc_ci_high"]
        overlap = not (lo_gat > hi_fus or lo_fus > hi_gat)
        overlap_text = ("The confidence intervals overlap, so with this dataset the improvement over "
                        "single-modality baselines cannot be attributed to the graph rather than to the multimodal "
                        "inputs." if overlap else
                        "The confidence intervals do not overlap, which favours the graph model.")
        ed.set_paragraph("[Answer from Tables 2 and 3.",
                         f"The ST-GAT reached an AUC of {r.ci_text(gat, 'roc_auc')} and an Ra RMSE of "
                         f"{r.ci_text(gat, 'rmse_um', 3)} µm, against {r.ci_text(fus, 'roc_auc')} and "
                         f"{r.ci_text(fus, 'rmse_um', 3)} µm for the multimodal fusion MLP. " + overlap_text,
                         self.SRC, keep_label=True)

        # RQ3: which sensor matters.
        ed.set_paragraph("[Answer from Table 5 and the edge-attention analysis",
                         "Thermal features could not be evaluated because no calibrated infrared frames were "
                         "available. Among the measured modalities, integrated gradients attributed "
                         + self.attribution_list("WAAM")
                         + " of the Ra prediction for WAAM trials, and "
                         + self.attribution_list("WROUGHT")
                         + " for wrought trials. Because the Ra targets are material means, these shares describe "
                           "which signals identify the material, and the relation to the finding of [1] (lower "
                           "vibration amplitude but higher roughness for WAAM) remains open until zone-level Ra is "
                           "measured.",
                         self.SRC, keep_label=True)

    # --- Limitations ------------------------------------------------------------------
    def limitations(self) -> None:
        self.ed.set_paragraph("Instability labels combine a spectral criterion and visual inspection.",
                              "Instability labels rest on a single spectral criterion applied to the full zone record; "
                              "visual inspection of the surface was not available. Any rule that depends on spectra "
                              "shares information with the model inputs; computing it at a much finer frequency "
                              "resolution than the model windows (Section 2.3) reduces but does not eliminate this "
                              "dependence. All unstable zones are WAAM zones, so instability and material are partly "
                              "confounded. Ra targets are specimen means from [1], not zone measurements.",
                              "this run")
        self.ed.replace("[describe how emissivity was handled]",
                        "thermal images were not used as model input in the present runs, so emissivity does not "
                        "affect the reported results; it must be documented if calibrated frames are added",
                        "this run")

    # --- 5. Latency ----------------------------------------------------------------------
    def latency(self) -> None:
        lat = self.r.latency
        within = "within" if lat["within_budget"] else "outside"
        self.ed.set_paragraph(
            "For online use, inference must complete within one window hop.",
            f"For online use, inference must complete within one window hop. With {CONFIG.signal.window_ms:g} ms "
            f"windows and {CONFIG.signal.overlap_fraction * 100:.0f}% overlap, the budget is "
            f"{lat['budget_ms_one_hop']:g} ms. The ST-GAT has {lat['parameters']:,} parameters and required "
            f"{lat['ms_per_graph_one_window']:.1f} ms per graph (six zones, one window) on a laptop CPU "
            f"(AMD Ryzen 5 5500U, {lat['threads']} threads), "
            f"{within} that budget; a whole trial "
            f"({lat['windows_per_graph']} windows) took {lat['ms_per_graph_all_windows']:.0f} ms. Because the "
            "graph is small (N = 6 nodes), the graph layers are cheap; the modality encoders dominate the "
            "computation.",
            "outputs/latency.json", keep_label=True)

    # --- 6. Conclusions -------------------------------------------------------------------
    def conclusions(self) -> None:
        r = self.r
        self.ed.set_paragraph(
            "This work proposed a spatio-temporal graph formulation",
            "This work proposed a spatio-temporal graph formulation for monitoring the milling of WAAM Al 5356 "
            "thin walls, in which cutting zones are nodes, tool-path order and heat diffusion define the edges, "
            "and force, vibration and thermal measurements form the node features. "
            f"On {r.n_trials} trials with force and acceleration input, the ST-GAT detected instability with an "
            f"AUC of {r.ci_text('gat', 'roc_auc')} against {r.ci_text('fusion_mlp', 'roc_auc')} for a non-graph "
            f"multimodal baseline, and recognised the material-level Ra with an RMSE of "
            f"{r.value('gat', 'rmse_um'):.3f} µm. "
            "Thermal information and zone-level Ra could not be evaluated with the available records, so RQ3 "
            "remains open.",
            self.SRC)
        self.ed.replace("[Add or remove items so that each corresponds to a result actually shown.]",
                        "(Contribution 4, on the importance of thermal information, is not supported by the present "
                        "runs and should be removed unless calibrated infrared data are added.)", "this run")
        self.ed.replace(" [Add as needed.]", "", "this run")

    # --- Abstract ---------------------------------------------------------------------------
    def abstract(self) -> None:
        ed, r, abl = self.ed, self.r, self.r.ablation
        gat = "gat"
        base = r.best_baseline()
        delong = r.test("DeLong AUC")

        ed.replace("Cutting forces (25 kHz), triaxial acceleration and infrared thermography from [20] milling trials",
                   f"Cutting forces and triaxial acceleration (both 25 kHz) from {r.n_trials} milling trials",
                   "this run")
        ed.replace("the GAT model reached [XX.X]% accuracy",
                   f"the GAT model reached {r.value(gat, 'accuracy'):.1f}% accuracy", self.SRC)
        ed.replace("ROC curve of [0.XX] for", f"ROC curve of {r.value(gat, 'roc_auc'):.2f} for", self.SRC)
        ed.replace("error of [0.0XX] µm, compared with",
                   f"error of {r.value(gat, 'rmse_um'):.3f} µm, compared with", self.SRC)

        # If the best baseline is a classifier without Ra output, use the fusion MLP's RMSE.
        rmse_model = base if pd.notna(r.value(base, "rmse_um")) else "fusion_mlp"
        ed.replace("compared with [XX.X]%, [0.XX] and [0.0XX] µm for the strongest non-graph multimodal baseline.",
                   f"compared with {r.value(base, 'accuracy'):.1f}%, {r.value(base, 'roc_auc'):.2f} and "
                   f"{r.value(rmse_model, 'rmse_um'):.3f} µm for the strongest "
                   f"non-graph multimodal baseline ({pretty(base)}).", self.SRC)

        if abl is not None:
            vibration = (f"removing acceleration features changed Ra error by "
                         f"{pct(r.delta_rmse('without_acceleration_features'))}% "
                         f"and removing force features by {pct(r.delta_rmse('without_force_features'))}%")
        else:
            vibration = "the vibratory ablation is reported in Table 4"
        ed.replace("Removing thermal information raised Ra error by [XX]%, against [XX]% for vibratory information.",
                   f"Thermal information could not be evaluated because no calibrated infrared frames were "
                   f"available; {vibration}.", self.SRC)

        significant = delong is not None and pd.notna(delong.p_value) and delong.p_value < 0.05
        if significant:
            implication = "Zone-level graph modelling gave a measurable gain for instability monitoring on thin WAAM walls."
        else:
            implication = ("With the present data the graph structure did not significantly improve on a non-graph "
                           "multimodal model, so larger datasets with measured zone-level labels are needed before "
                           "the approach is deployed.")
        ed.replace("[One sentence on the practical implication, written after the results are known.]",
                   implication, self.SRC)

    # --- Back matter --------------------------------------------------------------------------
    def back_matter(self) -> None:
        ed, r = self.ed, self.r
        ed.replace("[repository, DOI]", "[repository, DOI – author to supply]", "author")
        ed.replace("[ ]", "[author to complete]", "author details are not in the data", count=100)
        ed.replace("[tool name, version]", "Claude (Anthropic, Opus 5.5)", "this session")
        ed.replace("[describe, e.g., drafting the initial structure of the text]",
                   "writing the analysis code and drafting results text from the computed outputs", "this session")
        ed.replace("[year]", "2025", "ref. [1] header (Materials 2025, 18)")
        ed.replace("[vol]", "18", "ref. [1] header (Materials 2025, 18)")
        ed.replace("[article no.]", "[article no. – check]", "ref. [1]")
        ra_kind = "IMPUTED from material means" if r.info["ra_labels_imputed"] else "measured"
        ed.replace(" Every number in this section must come from your runs. Table structures and the order "
                   "of reporting are set up so that the text can be completed directly. Delete this note before "
                   "submission.",
                   f" Every number in this section was generated by src/make_report.py from the run in "
                   f"outputs/ ({r.n_trials} trials, seeds {r.info['seeds']}, Ra labels "
                   f"{ra_kind}, no thermal input). "
                   "Check report/fill_log.csv and delete this note before submission.", self.SRC)


# ===========================================================================
# 6. Figures
# ===========================================================================
class FigureMaker:
    """Draws Figures 3-5 and the ROC curves into outputs/figures/."""

    COLOURS = {"WAAM": "#c0392b", "WROUGHT": "#2471a3"}

    def __init__(self, results: ResultsLoader, figure_dir: Path):
        self.r = results
        self.dir = figure_dir
        self.dir.mkdir(parents=True, exist_ok=True)

        # One row per (model, zone): seeds averaged, full variant only.
        deep = results.pred[results.pred.variant == "full"]
        self.zone = (deep.groupby(["model", "trial_id", "material_type", "zone_id"], as_index=False)
                     .agg(y_class=("y_class", "first"), y_reg=("y_reg", "first"),
                          probability=("probability", "mean"), ra_pred=("ra_pred", "mean")))
        self.gat = self.zone[self.zone.model == "gat"]

    def draw_all(self) -> None:
        self.parity_plot()
        self.ra_along_wall()
        self.attention_bars()
        self.roc_curves()

    def save(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.dir / name, dpi=200)
        plt.close(fig)                                   # free the memory

    def parity_plot(self) -> None:
        """Figure 3: predicted vs target Ra for the ST-GAT."""
        g = self.gat
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for material, sub in g.groupby("material_type"):
            # Tiny horizontal jitter so that points with the same target do not hide each other.
            jitter = np.random.default_rng(0).normal(0, 0.005, len(sub))
            ax.scatter(sub.y_reg + jitter, sub.ra_pred, s=18, alpha=0.7,
                       color=self.COLOURS.get(material), label=material.capitalize())
        limits = [min(g.y_reg.min(), g.ra_pred.min()) - 0.05, max(g.y_reg.max(), g.ra_pred.max()) + 0.05]
        ax.plot(limits, limits, "k--", lw=1)             # the "perfect prediction" line
        ax.set(xlabel="Target Ra (µm, material mean of [1])", ylabel="Predicted Ra (µm)",
               xlim=limits, ylim=limits, title="Figure 3 – ST-GAT parity (out-of-fold)")
        ax.legend()
        self.save(fig, "fig3_parity_gat.png")

    def ra_along_wall(self) -> None:
        """Figure 4: Ra along the wall for two WAAM trials."""
        g = self.gat
        trials = sorted(g[g.material_type == "WAAM"].trial_id.unique())[:2] or sorted(g.trial_id.unique())[:2]
        fig, axes = plt.subplots(1, len(trials), figsize=(8, 3.2), sharey=True)
        axes = np.atleast_1d(axes)                       # works for 1 or 2 panels
        for ax, trial in zip(axes, trials):
            sub = g[g.trial_id == trial].sort_values("zone_id")
            position_mm = (sub.zone_id + 0.5) * CONFIG.process.zone_length_mm
            ax.plot(position_mm, sub.y_reg, "k--o", label="Target (imputed)")
            ax.plot(position_mm, sub.ra_pred, "-o", color=self.COLOURS["WAAM"], label="ST-GAT")
            ax.set(title=trial, xlabel="Position along wall (mm)")
        axes[0].set_ylabel("Ra (µm)")
        axes[0].legend()
        fig.suptitle("Figure 4 – Ra along the wall")
        self.save(fig, "fig4_ra_along_wall.png")

    def attention_bars(self) -> None:
        """Figure 5: mean GAT attention on the two edge types, per material."""
        if self.r.attention is None:
            return
        a = self.r.attention.set_index("material_type")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        x = np.arange(len(a))
        ax.bar(x - 0.2, a.kinematic_attention, 0.4, label="Kinematic edges", color="#7f8c8d")
        ax.bar(x + 0.2, a.thermal_attention, 0.4, label="Thermal (heat-kernel) edges", color="#e67e22")
        ax.set(xticks=x, xticklabels=[m.capitalize() for m in a.index], ylabel="Mean attention coefficient",
               title="Figure 5 – GAT attention by edge type")
        ax.legend(fontsize=8)
        self.save(fig, "fig5_attention.png")

    def roc_curves(self) -> None:
        """ROC curves of the deep models and the classical classifiers."""
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        for model in ["Logistic regression", "Random forest", "SVM (RBF)", "cnn1d", "bilstm", "fusion_mlp",
                      "gcn", "gat"]:
            sub = self.zone[(self.zone.model == model) & self.zone.probability.notna()]
            if sub.empty or sub.y_class.nunique() < 2:   # a ROC curve needs both classes
                continue
            fpr, tpr, _ = roc_curve(sub.y_class.astype(int), sub.probability)
            ax.plot(fpr, tpr, lw=2 if model == "gat" else 1,
                    label=f"{pretty(model)} ({self.r.value(model, 'roc_auc'):.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)              # the "random guess" line
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC – instability (out-of-fold)")
        ax.legend(fontsize=7)
        self.save(fig, "roc_curves.png")


# ===========================================================================
# 7. The whole of Step 3
# ===========================================================================
class ReportBuilder:
    def __init__(self, results_dir: Path, filled_path: Path):
        self.results_dir = results_dir
        self.filled_path = filled_path

    def run(self) -> None:
        if not DRAFT.exists():
            raise SystemExit(f"Draft not found: {DRAFT}")

        results = ResultsLoader(self.results_dir)
        editor = DocxEditor(docx.Document(DRAFT))

        # Fill the document section by section.
        MethodsSection(editor, results).fill()
        ResultsSection(editor, results).fill()
        DiscussionSection(editor, results).fill()

        # Save the filled copy and the change log, then draw the figures.
        editor.doc.save(self.filled_path)
        log = pd.DataFrame(editor.log)
        log.to_csv(REPORT_DIR / "fill_log.csv", index=False)
        figure_dir = self.results_dir / "figures"
        FigureMaker(results, figure_dir).draw_all()

        self.print_summary(editor, log, figure_dir)

    def print_summary(self, editor: DocxEditor, log: pd.DataFrame, figure_dir: Path) -> None:
        """Report what was done and which [brackets] are still open in the document."""
        remaining = []
        for paragraph in editor.doc.paragraphs:
            remaining += re.findall(r"\[[^\]]*\]", paragraph.text)
        for table in editor.doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    remaining += re.findall(r"\[[^\]]*\]", cell.text)
        # Drop citations like [1] or [12, 13].
        remaining = [x for x in remaining if not re.fullmatch(r"\[\d[\d,–\- ]*\]", x)]
        missing = log[log.location.str.startswith("NOT FOUND")]

        print(f"Filled report written to {self.filled_path}")
        print(f"{len(log) - len(missing)} changes logged in {REPORT_DIR / 'fill_log.csv'}; figures in {figure_dir}")
        if len(missing):
            print(f"WARNING: {len(missing)} placeholders were not found:\n{missing[['old']].to_string(index=False)}")
        print("Brackets still open (need the author or the ablation stage):")
        for bracket in sorted(set(remaining)):
            print("  ", bracket)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR, help="folder written by train_evaluate.py")
    parser.add_argument("--out", type=Path, default=FILLED, help="filled .docx to write")
    args = parser.parse_args()
    ReportBuilder(args.results, args.out).run()


if __name__ == "__main__":
    main()
