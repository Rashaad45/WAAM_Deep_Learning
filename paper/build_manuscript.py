"""
build_manuscript.py - writes paper/manuscript.docx (MDPI JMMP layout) from results/*.csv.

Every number in the text is read from results/*.csv at build time (check_numbers.py verifies it).
Run:  python paper/build_manuscript.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docbuilder import Manuscript  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import os
RES = Path(os.environ.get("WAAM_RESULTS_DIR", ROOT / "results"))
FIG = Path(os.environ.get("WAAM_FIGURES_DIR", ROOT / "paper" / "figures"))

MET = pd.read_csv(RES / "metrics.csv")
PAIR = pd.read_csv(RES / "paired_tests.csv")
CONST = pd.read_csv(RES / "constants.csv").set_index("name").value
LAB = pd.read_csv(RES / "label_counts.csv").set_index("threshold_db")
SENS = pd.read_csv(RES / "label_sensitivity.csv")
PAR = pd.read_csv(RES / "parameter_counts.csv").set_index("model").parameters
MAP = pd.read_csv(RES / "trial_mapping.csv")
TREND = pd.read_csv(RES / "speed_block_force_trend.csv")
CONV = pd.read_csv(RES / "training_convergence.csv")
FOLD = pd.read_csv(RES / "per_fold.csv")

NAMES = {"logreg": "logistic regression", "rf": "random forest", "svm": "SVM", "cnn1d": "1D-CNN",
         "bilstm": "BiLSTM", "fusion_mlp": "fusion MLP", "small_gnn": "small GNN", "gcn": "ST-GCN", "gat": "ST-GAT"}
TNAMES = {"logreg": "Logistic regression", "rf": "Random forest", "svm": "SVM (RBF)", "cnn1d": "1D-CNN",
          "bilstm": "BiLSTM", "fusion_mlp": "Fusion MLP", "small_gnn": "Small GNN", "gcn": "ST-GCN", "gat": "ST-GAT"}
ORDER = list(NAMES)
CLASSICAL, DEEP = ORDER[:3], ORDER[3:]


# ----------------------------------------------------------------------------- lookups
def c(name, fmt="g"):
    v = float(CONST[name])
    return format(v, fmt)


def met(model, exp="main_15db", var="full"):
    r = MET[(MET.experiment == exp) & (MET.model == model) & (MET.variant == var)]
    assert len(r) == 1, (model, exp, var)
    return r.iloc[0]


def f2(x):
    if pd.isna(x):
        return "n/a"
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s.replace("-", "−")


def auc_ci(model, exp="main_15db", var="full"):
    r = met(model, exp, var)
    return f"{f2(r.auc)} ({f2(r.auc_ci_low)}–{f2(r.auc_ci_high)})"


def pair(ma, mb, exp_b="main_15db", vb="full", va="full", exp_a="main_15db"):
    r = PAIR[(PAIR.model_a == ma) & (PAIR.model_b == mb) & (PAIR.experiment_b == exp_b) & (PAIR.variant_b == vb)
             & (PAIR.variant_a == va) & (PAIR.experiment_a == exp_a)]
    assert len(r) == 1, (ma, mb, exp_b, vb)
    return r.iloc[0]


def dstr(r):
    return f"ΔAUC = {f2(r.delta_auc)} (95% CI {f2(r.delta_ci_low)} to {f2(r.delta_ci_high)}; p = {r.perm_p:.3f})"


def inside_null(model, exp="main_15db", var="full"):
    r = met(model, exp, var)
    return r.null_auc_q025 <= r.auc <= r.null_auc_q975


main = MET[(MET.experiment == "main_15db") & (MET.variant == "full")].set_index("model").reindex(ORDER)
best = main.auc.idxmax()
best_deep = main.loc[DEEP].auc.idxmax()
above_null = [m for m in ORDER if not inside_null(m)]
null_lo, null_hi = main.null_auc_q025.mean(), main.null_auc_q975.mean()
n_pos15 = int(LAB.loc[15.0, "n_positive"]); n_pos20 = int(LAB.loc[20.0, "n_positive"]); n_pos25 = int(LAB.loc[25.0, "n_positive"])
mism = MAP[MAP.nominal_matches_measured == "NO"].trial_id.str[-2:].tolist()
blocks = TREND.dropna(subset=["spearman_rho"])
sig_vs_lr = PAIR[(PAIR.model_b == "logreg") & (PAIR.experiment_a == "main_15db") & (PAIR.perm_p < 0.05)]
edges = {m: pair(m, m, "ablation_15db", "no_edges") for m in ("gat", "gcn", "small_gnn")}
flagged = MAP[MAP.accel_alignment_r < float(CONST["alignment_r_min"])].trial_id.str[-2:].tolist()
_al_auc = MET[(MET.experiment == "alignment_15db") & (MET.variant == "full")].set_index("model").auc
FEATURE_MODELS = [m for m in ORDER if m != "cnn1d"]
_drop = [m for m in FEATURE_MODELS if _al_auc[m] < main.auc[m]]
AL_DROP_TXT = ("every feature-based model" if len(_drop) == len(FEATURE_MODELS)
               else f"{len(_drop)} of {len(FEATURE_MODELS)} feature-based models")


def names_list(models):
    items = [NAMES[m] for m in models]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


TITLE = ("A Leakage-Free Small-Data Benchmark for Force- and Vibration-Based Chatter Detection in "
         "Thin-Wall Milling of Wire-Arc Additively Manufactured and Wrought Al–Mg Walls")


def build() -> Manuscript:
    ms = Manuscript()
    d = ms.doc
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("Article"); r.italic = True
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(TITLE); r.bold = True; r.font.size = ms.doc.styles["Normal"].font.size * 1.8
    ms.para("[Author names and affiliations – author to complete]", indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)

    # ------------------------------------------------------------------ abstract
    lr_m = met("logreg")
    ms.para(
        "**Abstract:** Small machining datasets make optimistic chatter-detection results easy to obtain. We "
        "re-analysed force and vibration records from face-milling of thin wire-arc additively manufactured (WAAM) "
        f"ER5356 and wrought AA 5052 walls ({c('n_trials_used')} trials, {c('n_zones')} zones) with a leakage-free "
        "protocol. Spindle speeds measured from the force spectra contradicted the archived trial mapping for "
        f"{len(mism)} trials, and no trial had a confirmed material, so no result uses material identity. Zones were "
        "labelled by a fixed spectral rule (a non-synchronous 2.5–5.5 kHz force peak above a prominence threshold). "
        f"The labels depended strongly on the threshold: {n_pos15}, {n_pos20} and {n_pos25} of {c('n_zones')} zones "
        "were unstable at 15, 20 and 25 dB, so only 15 dB could be evaluated. Under leave-one-trial-out validation "
        f"with five seeds, pooled ROC-AUC ranged from {f2(main.auc.drop('cnn1d').min())} to {f2(main.auc.max())} for "
        f"all models using hand-crafted features ({NAMES[best]} highest), against a permutation-null band of "
        f"{f2(null_lo)}–{f2(null_hi)}. No model differed from logistic regression ({f2(lr_m.auc)}) in trial-level "
        "paired tests, graph edges added nothing, and a 1D-CNN on raw signals performed at chance. Removing two "
        f"weakly aligned trials lowered the AUC of {AL_DROP_TXT}. The dominant uncertainty is "
        "the label definition, not the "
        "classifier.", indent=False)
    ms.para("**Keywords:** chatter detection; thin-wall milling; wire arc additive manufacturing; Al–Mg alloys; "
            "cutting force; vibration; graph neural networks; leakage-free evaluation; small data", indent=False)

    # ------------------------------------------------------------------ 1 introduction
    ms.heading("1. Introduction")
    ms.para(
        "Wire arc additive manufacturing (WAAM) builds near-net metal parts bead by bead, and almost every functional "
        "WAAM part is finish-machined. The deposited material differs from wrought stock in ways that matter for "
        "machining: WAAM deposits are prone to defects such as porosity [2], Al–Mg deposits can lose magnesium and "
        "show anisotropic microstructure and properties [3], and each new layer reheats the layers below it, which acts "
        "as an uncontrolled in-situ heat treatment [2]. In our earlier study of WAAM ER5356 walls [1], the deposited "
        "material was stronger than a wrought Al–Mg reference yet gave rougher machined surfaces, and porosity was "
        "proposed as a trigger of dynamic instability during machining. Thin walls add low and position-dependent "
        "stiffness, so finish milling of WAAM walls is a natural setting for chatter monitoring.")
    ms.para(
        "Machine-learning chatter detection has been studied on turning and milling signals with a range of "
        "featurisations, from wavelet packets and empirical mode decomposition to topological and similarity-based "
        "descriptors [4,5], and public multi-sensor datasets now exist for turning [6] and for industrial CNC milling "
        "with force, acceleration and controller signals [7,8]. Graph neural networks [9,10] are an attractive "
        "way to let neighbouring cutting zones share information. Small machining datasets, however, make it easy to "
        "report optimistic results: zones of the same cut are correlated, labels derived from spectral rules can hinge "
        "on arbitrary thresholds, early stopping on tiny validation sets leaks information, and baselines are often "
        "weaker than the proposed model by construction.")
    ms.para(
        "This paper is a deliberately conservative re-analysis of the force and vibration records of [1]. A pre-analysis "
        "audit of the data archive and of an earlier modelling pipeline found that (i) the trial-to-condition mapping "
        "was partly wrong and the trial-to-material mapping could not be confirmed, (ii) the earlier labels used nominal "
        "spindle speeds, which are too inaccurate for a ±2 Hz harmonic test near 3.5 kHz, and (iii) several evaluation "
        "choices could inflate or deflate scores. The contributions are:")
    ms.numbered([
        "an evidence-based reconstruction of the trial-to-cutting-condition mapping from the measured spindle speed, "
        "the force level and file provenance, without inferring material from signal differences;",
        "a chatter labelling rule that uses the measured spindle frequency, and a quantification of how strongly the "
        "labels depend on the prominence threshold;",
        "a leakage-free leave-one-trial-out benchmark of three classical and six deep models, including two "
        f"spatio-temporal graph networks, a graph network with fewer than {c('small_gnn_param_limit')} parameters and no-edge ablations, "
        "with trial-bootstrap confidence intervals, label-permutation nulls and trial-level paired tests;",
        "a frank statement of what the data cannot support, as a basis for the additional measurements needed."])

    # ------------------------------------------------------------------ 2 methods
    ms.heading("2. Materials and Methods")
    ms.heading("2.1. Specimens, Cutting Experiments and Sensors", 2)
    ms.para(
        f"The specimens and cutting plan are those of [1]. Walls of WAAM ER5356 and of wrought AA 5052 were face-milled "
        f"with a {c('flutes')}-flute carbide end mill of {c('tool_d_mm')} mm diameter at a constant feed of "
        f"{c('feed_mm_min')} mm/min across a wall width of {c('wall_mm')} mm, at two nominal spindle speeds "
        f"({c('rpm_low_nominal')} and {c('rpm_high_nominal')} rpm) and depths of cut from {c('depth_min_mm')} to "
        f"{c('depth_max_mm')} mm in steps of {c('depth_step_mm')} mm. Cutting forces were recorded with a Kistler "
        f"9257B dynamometer at {c('force_fs_khz')} kHz and vibration with a triaxial accelerometer at "
        f"{c('accel_fs_khz')} kHz on a separate acquisition computer; the accelerometer model and mounting position "
        "are not recorded in the data archive. Infrared images were recorded in [1] but are not used here "
        "(Section 5).")
    ms.para(
        f"The archive contains {c('n_trials_recorded')} trial recordings. Force records exist for "
        f"{c('n_trials_force')} of them; one force record starts mid-cut, leaving {c('n_trials_used')} usable trials. "
        "The two computers were not synchronised, so the accelerometer record was aligned to the force record by "
        "maximising the correlation between the log-RMS envelope of the acceleration and a step template marking the "
        f"cut in the force record. The alignment correlation ranged from {c('align_r_min', '.2f')} to "
        f"{c('align_r_max', '.2f')}; {c('n_align_flagged')} trials (trials {' and '.join(flagged)}) fell below "
        f"r = {c('alignment_r_min')} and were flagged for a sensitivity analysis.")

    ms.heading("2.2. Trial Mapping from Measured Spindle Speed and Force", 2)
    ms.para(
        "The trial-to-condition and trial-to-material assignments in the archive were provisional (assigned by trial "
        "number, without a laboratory log). Because the labelling rule needs an accurate spindle frequency, the "
        "spindle-rotation frequency of every trial was measured from its force record with a harmonic comb: with "
        "$P(f)$ the periodogram of the summed, de-meaned three-axis force over the retained cut,")
    ms.equation(r"\hat{f}_0=\mathrm{arg\,max}_{f_0\in[12,30]\,\mathrm{Hz}}\;\frac{1}{60}\sum_{n=1}^{60}\log_{10}P(n f_0)")
    ms.para(
        f"evaluated on a {c('comb_grid_hz')} Hz grid and refined to {c('comb_refine_hz', '.5f')} Hz. The search range covers both nominal speeds with a "
        "wide margin and does not use the nominal value. Speeds were then compared with the provisional mapping, and "
        "the mean resultant cutting force was compared with the trial order inside each block of equal measured speed, "
        "since the cutting plan of [1] runs five depths per speed. Material was never inferred from signal or "
        "material-property differences; the only admissible material evidence is file provenance.")

    ms.heading("2.3. Zones and Windows", 2)
    ms.para(
        "The cut was detected in the force envelope as the longest sustained burst above the air-cutting level "
        f"(detected cuts of {c('cut_s_min', '.1f')}–{c('cut_s_max', '.1f')} s, i.e., "
        f"{c('cut_mm_min', '.1f')}–{c('cut_mm_max', '.1f')} mm of tool travel). Because the cutter is "
        f"{c('tool_d_mm')} mm wide, zones narrower than the tool overlap heavily, and the first and last zones contain "
        f"the engagement transients. The detected cut $[t_s,t_e]$ was therefore divided into {c('zones')} zones of "
        f"equal length, and one tool radius ({c('exclusion_mm')} mm, {c('exclusion_s', '.2f')} s at the feed used) was "
        "removed at each end:")
    ms.equation(r"Z_k=\left[\mathrm{max}\!\left(t_s+\tfrac{k-1}{3}\Delta,\;t_s+\delta_e\right),\;"
                r"\mathrm{min}\!\left(t_s+\tfrac{k}{3}\Delta,\;t_e-\delta_e\right)\right],\quad \Delta=t_e-t_s,\;k=1,2,3")
    ms.para(
        f"with $\\delta_e$ = {c('exclusion_s', '.2f')} s. Retained zones lasted {c('zone_s_min', '.2f')}–"
        f"{c('zone_s_max', '.2f')} s. A zone is a bin of tool-centre positions; each window integrates material within "
        f"one tool radius of the centre. In each zone, $T$ = {c('windows_per_zone')} windows of {c('window_ms')} ms were "
        "spread evenly over the retained span. This layout was fixed from the tool geometry before any model was rerun.")

    ms.heading("2.4. Chatter Labelling Rule", 2)
    ms.para(
        f"Each zone was labelled from the Welch power spectral density of its full retained force record "
        f"(segments of {c('label_nperseg')} samples, {c('label_bin_hz', '.2f')} Hz bins), summed over the three axes, "
        f"$S_z(f)$. Within the band $B$ = {c('band_low_khz')}–{c('band_high_khz')} kHz, which contains the dominant "
        f"structural modes near {c('mode1_khz')} and {c('mode2_khz')} kHz reported in [1], the highest peak "
        "$f_z$ and its prominence over the band median are")
    ms.equation(r"f_z=\mathrm{arg\,max}_{f\in B}S_z(f),\qquad \Pi_z=10\log_{10}\frac{S_z(f_z)}{\mathrm{median}_{f\in B}S_z(f)}")
    ms.para("The distance of the peak to the nearest spindle-rotation harmonic, using the measured $\\hat{f}_0$, is")
    ms.equation(r"d_z=\left|f_z-\mathrm{round}\!\left(f_z/\hat{f}_0\right)\hat{f}_0\right|")
    ms.para("and the zone is labelled unstable when the peak is not forced and is prominent enough:")
    ms.equation(r"y_z=\mathrm{I}\left[d_z>\delta_h\;\wedge\;\Pi_z\geq\tau\right]")
    ms.para(
        f"where $\\mathrm{{I}}[\\cdot]$ is 1 if the condition holds and 0 otherwise, "
        f"$\\delta_h$ = {c('tol_hz')} Hz and $\\tau$ ∈ {{15, 20, 25}} dB. Spindle-rotation rather than "
        "tooth-passing harmonics are excluded because cutter runout forces vibration at every multiple of the spindle "
        f"frequency. In the chatter band the harmonic numbers of the observed peaks range from {c('harmonic_number_min')} to "
        f"{c('harmonic_number_max')}, so a ±{c('tol_hz')} Hz test needs the spindle frequency to about "
        f"{c('freq_precision_needed_hz', '.2f')} Hz; nominal speeds are far less accurate (Section 3.1).")

    ms.heading("2.5. Features", 2)
    ms.para(
        f"For each window and each of the six channels (three force axes, three acceleration axes), "
        f"{c('features_per_channel')} features were computed: RMS, peak-to-peak value, kurtosis, crest factor, the "
        f"log energy within ±{c('mode_hw_hz')} Hz of each structural mode ({c('welch_segment')}-point Welch spectrum), "
        f"a chatter indicator, the wavelet energy entropy and log energy of a complex Morlet transform with "
        f"{c('wavelet_scales')} scales, and the spectral centroid in the chatter band. The chatter indicator uses the "
        f"same harmonic definition as the label. It is computed on a {c('context_s')} s context centred on the window "
        f"(periodogram, {c('context_bin_hz')} Hz bins), because {c('window_ms')} ms windows cannot resolve spindle "
        "harmonics:")
    ms.equation(r"\mathrm{CI}=\frac{\sum_{f\in B,\;d(f)>\delta_h}P_c(f)}{\sum_{f\in B}P_c(f)}")
    ms.para(
        f"where $P_c$ is the context periodogram and $d(f)$ the distance to the nearest spindle harmonic. Two "
        f"kinematic/geometric inputs were added per zone: the measured spindle speed and the zone-centre position along "
        f"the wall, giving {c('n_features')} inputs per window. Depth of cut was not used because its mapping to trials "
        "is not confirmed. Raw windows were decimated to "
        f"{c('encoder_fs_khz')} kHz (anti-aliased) for the convolutional encoders.")

    ms.heading("2.6. Models", 2)
    ms.para(
        f"Classical models (logistic regression, a random forest with {c('rf_trees')} trees, and an RBF support vector machine, all "
        "class-balanced) used the window-averaged features of each zone. The deep models were a 1D-CNN on raw "
        "windows only, a bidirectional LSTM over the window sequence of features, a fusion MLP that combines features "
        "and CNN encodings per zone without any edges, two spatio-temporal graph networks (ST-GCN and ST-GAT) and a "
        f"small GNN. In the graph models each trial is a path graph of {c('zones')} zones whose kinematic edges join "
        "consecutive zones along the tool path ($A_{ij}=1$ if $|i-j|=1$). The graph convolution [9] is")
    ms.equation(r"H^{(l+1)}=\sigma\!\left(\tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)")
    ms.para("and the attention layer [10] weights the neighbours $j$ of zone $i$ (including $i$ itself) with a "
            "learned bias $b$ on the edge weight, the softmax running over those neighbours:")
    ms.equation(r"\alpha_{ij}=\mathrm{softmax}_{j}\left(\mathrm{LeakyReLU}"
                r"\left(a_1^{\top}W h_i+a_2^{\top}W h_j+b\,A_{ij}\right)\right)")
    ms.para(
        "Both ST models apply two blocks of a graph layer followed by a gated temporal convolution over the windows. "
        f"The small GNN projects the hand-crafted features to {c('small_hidden')} channels and applies two graph "
        "convolutions. Every model maps a zone to one logit using the same window-to-zone rule, the mean over the "
        "$T$ windows of the last per-window representation:")
    ms.equation(r"s_z=w^{\top}\,\frac{1}{T}\sum_{t=1}^{T}h_{z,t}+c")
    ms.para(
        "where $h_{z,t}$ is the last hidden representation of window $t$ in zone $z$, and $w$, $c$ are the weights "
        "and bias of the output layer; $\\sigma(s_z)$ is the predicted probability of instability. "
        "No-edge ablations replace $A$ by zero (self-loops only), so the graph models reduce to per-zone models with "
        "otherwise identical architecture. Force-only and acceleration-only ablations zero the other modality. "
        "Table 1 lists the parameter counts.")
    rows = [[TNAMES[m], f"{int(PAR[m])}"] for m in ["logreg", "cnn1d", "small_gnn", "fusion_mlp", "bilstm", "gcn", "gat"]]
    ms.table(["Model", "Trainable parameters"], rows,
             f"Model capacity. The dataset has {c('n_zones')} zones from {c('n_trials_used')} trials.",
             note="Logistic regression: one weight per input plus an intercept. Random forest and SVM are "
                  "non-parametric.", widths=[6, 4])

    ms.heading("2.7. Training and Evaluation Protocol", 2)
    ms.para(
        "Evaluation used leave-one-trial-out (LOTO) cross-validation: each trial is the test fold once and never "
        f"appears in training. Every model was trained with {c('n_seeds')} seeds, and probabilities were averaged over "
        "seeds per zone before scoring. Inputs were standardised with training-trial statistics. The deep models were "
        f"trained for a fixed {c('epochs')} epochs (Adam, learning rate {c('lr')}, decoupled weight decay "
        f"{c('weight_decay')}, one trial per step) without a validation split or early stopping, because most trials "
        "contain no unstable zone and an inner validation set would usually have no positives. The epoch count was "
        "fixed before the runs and not tuned. The only loss is the class-weighted binary cross-entropy")
    ms.equation(r"\mathcal{L}=-\frac{1}{N}\sum_{z=1}^{N}\left[w_+\,y_z\ln\sigma(s_z)+(1-y_z)\ln\left(1-\sigma(s_z)\right)\right],"
                r"\qquad w_+=\frac{N_{\mathrm{neg}}}{N_{\mathrm{pos}}}")
    ms.para(
        "computed on the training trials of each fold. A fold whose training trials contain no unstable zone cannot "
        "train a classifier; it predicts the training prevalence and is reported.")
    ms.para(
        "Metrics are computed on the pooled out-of-fold predictions: ROC-AUC, average precision (AP), F1, precision "
        f"and recall at a probability of {c('decision_threshold')}. Confidence intervals ({c('ci_level_pct')}%) come from "
        f"{c('bootstrap')} bootstrap resamples of whole trials. The chance level of AUC is shown by a null distribution "
        f"from {c('permutations')} permutations of the zone labels. Model comparisons use only trial-level paired "
        "tests: a paired trial bootstrap of the AUC difference, and a paired permutation test that swaps the two "
        "models' scores for randomly chosen whole trials. Tests that treat zones as independent (e.g., DeLong) were not "
        "used, because zones of one trial are correlated. Sensitivity runs repeated the benchmark without the two "
        "weakly aligned trials and re-labelled the zones with harmonic tolerances of 0.5, 1 and 2 Hz, with measured "
        "and nominal spindle frequencies.")

    # ------------------------------------------------------------------ 3 results
    results_section(ms)
    discussion_section(ms)
    back_matter(ms)
    return ms


def results_section(ms: Manuscript):
    ms.heading("3. Results")
    ms.heading("3.1. The Trial Mapping Had to Be Reconstructed", 2)
    lo, hi = c("rpm_low_measured_mean", ".0f"), c("rpm_high_measured_mean", ".0f")
    ms.para(
        f"The measured spindle speeds formed two tight classes, {lo} and {hi} rpm, against nominal "
        f"{c('rpm_low_nominal')} and {c('rpm_high_nominal')} rpm (Figure 1a). The comb contrast was at least "
        f"{c('comb_contrast_min_db', '.1f')} dB in every trial. The provisional mapping gave the wrong speed for "
        f"{len(mism)} trials ({', '.join(mism)}). Trials with equal measured speed formed consecutive blocks, and inside "
        "every block of three or more trials the mean cutting force rose monotonically with trial order (Spearman "
        f"ρ = {f2(blocks.spearman_rho.min())} in each block; Figure 1b). This is the pattern expected if each block runs "
        "the planned depths in increasing order. The largest recording gap and a change of the force-file folder both "
        "occur between trials 11 and 12. Together these suggest that one specimen was cut in trials 12–20 "
        "(complete speed blocks of five and four usable trials) and another up to trial 11, with a boundary one trial "
        "later than the provisional mapping assumed.")
    ms.para(
        "None of this identifies the material. The only file-level material evidence concerns two trials recorded in "
        "a different session without force data, so no trial with a force record has a confirmed material. All results "
        "below are therefore reported without reference to material, and the material-specific analysis planned as "
        f"supplementary material was omitted (it required at least {c('min_confirmed_trials_per_material')} confirmed trials per material). The full "
        "evidence table is provided with the code (audit/trial_mapping_evidence.md).")
    ms.figure(FIG / "fig1_mapping.png",
              "(a) Spindle speed per trial: provisional (nominal) value from the archive and value measured from the "
              "force spectrum with the harmonic comb (Eq. 1). (b) Mean resultant cutting force over the retained cut, "
              "connected within blocks of equal measured speed.")

    ms.heading("3.2. Labels Depend Strongly on the Prominence Threshold", 2)
    rows = []
    for t in (15.0, 20.0, 25.0):
        r = LAB.loc[t]
        rows.append([f"{t:.0f}", f"{int(r.n_positive)} / {int(r.n_zones)}", f"{int(r.n_trials_with_positive)}",
                     f"{int(r.zones_within_2db)}"])
    ms.table(["Threshold τ (dB)", "Unstable zones", "Trials with ≥1 unstable zone", "Zones within ±2 dB of τ"], rows,
             "Zone labels at the three prominence thresholds (measured spindle frequency, ±2 Hz harmonic tolerance).",
             widths=[3.2, 3.2, 4.6, 4.2])
    nom15 = SENS[(SENS.spindle_frequency == "nominal") & (SENS.tolerance_hz == 2.0) & (SENS.threshold_db == 15.0)].iloc[0]
    tol05 = SENS[(SENS.spindle_frequency == "measured") & (SENS.tolerance_hz == 0.5) & (SENS.threshold_db == 15.0)].iloc[0]
    ms.para(
        f"At 15 dB, {n_pos15} of {c('n_zones')} zones were unstable, spread over "
        f"{int(LAB.loc[15.0, 'n_trials_with_positive'])} trials. At 20 dB only {n_pos20} zone remained, and at 25 dB "
        f"none (Table 2, Figure 2). {int(LAB.loc[15.0, 'zones_within_2db'])} of {c('n_zones')} zones lie within ±2 dB "
        "of the 15 dB threshold, so small changes in the threshold or in the spectral estimate flip many labels; at "
        f"15 dB the 'unstable' class even contains {f2(main.prevalence.iloc[0])} of all zones, which is a high "
        "rate for chatter in finishing cuts. A classifier can only be trained and tested in LOTO when unstable zones occur in "
        "several trials, so the benchmark is reported at 15 dB. At 20 dB, the single unstable zone means that "
        "every fold has either no positive training zone or no positive test zone; the pooled AUC then rests on one "
        "zone and is not a meaningful estimate (Section 3.5). Using the nominal instead of the measured spindle "
        f"frequency changed {int(nom15.n_changed_vs_main)} labels at 15 dB, and narrowing the harmonic tolerance to "
        f"0.5 Hz changed {int(tol05.n_changed_vs_main)} labels.")
    ms.figure(FIG / "fig2_prominence.png",
              "Prominence of the dominant chatter-band force peak over the band median for every zone, with the three "
              "thresholds. Crosses mark peaks within ±2 Hz of a spindle-rotation harmonic (forced, never labelled "
              "unstable).")

    ms.heading("3.3. Benchmark at 15 dB", 2)
    rows = []
    for m in ORDER:
        r = main.loc[m]
        rows.append([TNAMES[m], f"{f2(r.auc)} ({f2(r.auc_ci_low)}–{f2(r.auc_ci_high)})",
                     f"{f2(r.average_precision)} ({f2(r.ap_ci_low)}–{f2(r.ap_ci_high)})", f2(r.f1),
                     f2(r.precision), f2(r.recall), f"{r.perm_p_auc:.3f}"])
    ms.table(["Model", "ROC-AUC (95% CI)", "AP (95% CI)", "F1", "Precision", "Recall", "Perm. p"], rows,
             f"Leave-one-trial-out results at τ = 15 dB ({n_pos15} unstable of {c('n_zones')} zones; "
             f"prevalence {f2(main.prevalence.iloc[0])}). Pooled out-of-fold predictions, {c('n_seeds')} seeds averaged.",
             note=f"CI: {c('bootstrap')} trial-bootstrap resamples. Permutation p: two-sided, {c('permutations')} "
                  f"zone-label permutations; the 95% null band of AUC is {f2(null_lo)}–{f2(null_hi)}. F1, precision "
                  "and recall at probability 0.5; Perm. p: permutation p-value.",
             widths=[3.4, 3.1, 3.1, 1.3, 1.8, 1.5, 1.6])
    ms.para(
        f"The highest pooled AUC was obtained by the {NAMES[best]}, {auc_ci(best)}, and the best deep model was the "
        f"{NAMES[best_deep]}, {auc_ci(best_deep)} (Table 3, Figure 3). The 95% permutation null band of AUC was "
        f"{f2(null_lo)}–{f2(null_hi)}. "
        + (f"Of the {len(ORDER)} models, {len(above_null)} scored above this band; "
           f"{names_list([m for m in ORDER if m not in above_null])} did not. " if above_null else
           "No model scored outside this band. ")
        + "The trial-bootstrap intervals are wide for every model and overlap almost completely. Average precision "
          f"should be read against the prevalence of {f2(main.prevalence.iloc[0])}. The 1D-CNN, the only model "
          f"without hand-crafted features, reached {auc_ci('cnn1d')}, inside the null band.")
    ms.para(
        f"Pooled AUC mixes two abilities: ranking whole trials and ranking zones within a trial. Only "
        f"{int(main.n_trials_both_classes.iloc[0])} trials contain both stable and unstable zones, each with "
        f"{c('zones')} zones, so within-trial AUC is coarse. Averaged over those trials it was "
        f"{f2(main.loc['logreg'].mean_within_trial_auc)} for logistic regression and between "
        f"{f2(main.drop('logreg').mean_within_trial_auc.min())} and {f2(main.drop('logreg').mean_within_trial_auc.max())} "
        "for all other models. Most of the pooled discrimination therefore comes from separating trials, not from "
        "locating unstable zones along a wall.")
    ms.figure(FIG / "fig3_benchmark.png",
              "Pooled ROC-AUC at τ = 15 dB with trial-bootstrap 95% CIs, for all trials and without the two weakly "
              "aligned trials. The shaded band is the 95% label-permutation null.", width_cm=12.5)

    ms.heading("3.4. Paired Comparisons and Ablations", 2)
    vs = PAIR[(PAIR.model_b == "logreg") & (PAIR.experiment_a == "main_15db") & (PAIR.experiment_b == "main_15db")]
    rows = []
    for _, r in vs.iterrows():
        rows.append([f"{TNAMES[r.model_a]} − Logistic regression", f2(r.delta_auc),
                     f"{f2(r.delta_ci_low)} to {f2(r.delta_ci_high)}", f"{r.perm_p:.3f}"])
    for m in ("gat", "gcn", "small_gnn"):
        r = pair(m, "fusion_mlp")
        rows.append([f"{TNAMES[m]} − Fusion MLP", f2(r.delta_auc), f"{f2(r.delta_ci_low)} to {f2(r.delta_ci_high)}",
                     f"{r.perm_p:.3f}"])
    ab = PAIR[PAIR.experiment_b == "ablation_15db"]
    for _, r in ab.iterrows():
        rows.append([f"{TNAMES[r.model_a]}: full − {r.variant_b.replace('_', ' ')}", f2(r.delta_auc),
                     f"{f2(r.delta_ci_low)} to {f2(r.delta_ci_high)}", f"{r.perm_p:.3f}"])
    ms.table(["Comparison (A − B)", "ΔAUC", "95% CI (trial bootstrap)", "p (paired trial permutation)"], rows,
             "Trial-level paired comparisons at τ = 15 dB.", widths=[7.0, 2.0, 4.0, 3.5])
    n_sig = len(sig_vs_lr)
    sig_txt = ("No model differed from logistic regression at p < 0.05" if n_sig == 0 else
               f"{names_list(sig_vs_lr.model_a.tolist())} differed from logistic regression at p < 0.05")
    edge_txt = "; ".join(f"{NAMES[m]}: {dstr(edges[m])}" for m in edges)
    gat_f, gat_a = pair("gat", "gat", "ablation_15db", "no_force"), pair("gat", "gat", "ablation_15db", "no_acceleration")
    lr_f, lr_a = (pair("logreg", "logreg", "ablation_15db", "no_force"),
                  pair("logreg", "logreg", "ablation_15db", "no_acceleration"))
    ms.para(
        f"{sig_txt} (Table 4, Figure 4). Removing the graph edges did not produce a reliable change for any graph "
        f"model ({edge_txt})."
        if all(edges[m].perm_p >= 0.05 for m in edges) else
        f"{sig_txt} (Table 4, Figure 4). Removing the graph edges gave: {edge_txt}.")
    ms.para(
        f"For the modality ablations, removing force from the ST-GAT gave {dstr(gat_f)} and removing acceleration "
        f"gave {dstr(gat_a)}. For logistic regression the corresponding values were {dstr(lr_f)} and {dstr(lr_a)}. "
        "A positive ΔAUC means the full model was better.")
    ms.figure(FIG / "fig4_paired.png",
              "Trial-level paired differences in AUC (A − B) with trial-bootstrap 95% CIs; p-values from the paired "
              "trial-level permutation test.", width_cm=14.0)

    ms.heading("3.5. Sensitivity Analyses and Training Behaviour", 2)
    al = MET[(MET.experiment == "alignment_15db") & (MET.variant == "full")].set_index("model")
    ms.para(
        f"Without the two weakly aligned trials ({' and '.join(flagged)}), the best AUC was {auc_ci(al.auc.idxmax(), 'alignment_15db')} "
        f"({NAMES[al.auc.idxmax()]}); logistic regression gave {auc_ci('logreg', 'alignment_15db')} and the ST-GAT "
        f"{auc_ci('gat', 'alignment_15db')} (Figure 3). At τ = 20 dB the classical models, the BiLSTM and the small "
        f"GNN were run for completeness. With one unstable zone, {int(met('logreg', 'main_20db').folds_no_positive_training)} "
        "fold had no positive training zone, and the pooled AUC "
        f"(logistic regression {f2(met('logreg', 'main_20db').auc)}) reflects only where that single zone was ranked; "
        "it is not an estimate of detection performance. At 25 dB no zone was unstable and no model could be "
        "evaluated.")
    cv = CONV[(CONV.experiment == "main_15db") & (CONV.variant == "full")].set_index("model")
    fit = cv[cv.loss_last < 0.1].index.tolist()
    nofit = cv[cv.loss_last >= 0.1].index.tolist()
    ms.para(
        "The fixed epoch count was not tuned, and the deep models responded to it differently (Figure 5). At the last "
        f"epoch the {names_list(fit)} had driven their mean training loss to "
        f"{f2(cv.loc[fit].loss_last.min())}–{f2(cv.loc[fit].loss_last.max())} (from "
        f"{f2(cv.loc[fit].loss_epoch1.min())}–{f2(cv.loc[fit].loss_epoch1.max())} after the first epoch), i.e., they "
        f"fitted the training trials almost perfectly, while the {names_list(nofit)} remained at "
        f"{f2(cv.loc[nofit].loss_last.min())}–{f2(cv.loc[nofit].loss_last.max())}. Neither near-perfect fitting nor "
        "under-fitting translated into better held-out AUC than the linear model: the bottleneck is data, not "
        "optimisation.")
    ms.figure(FIG / "fig5_training_loss.png",
              "Mean training loss per epoch (all folds and seeds) of the deep models at τ = 15 dB.", width_cm=11.5)


def discussion_section(ms: Manuscript):
    lr = met("logreg")
    ms.heading("4. Discussion")
    rf_lr = pair("rf", "logreg")
    ms.para(
        "The main result is negative for model complexity and cautionary for labels. With "
        f"{c('n_zones')} zones from {c('n_trials_used')} trials, every model that used the physics-based spectral "
        f"features reached a pooled AUC between {f2(main.auc.drop('cnn1d').min())} and {f2(main.auc.max())}, and "
        f"none differed from logistic regression in trial-level paired tests (the largest difference, for the "
        f"{NAMES['rf']}, was {dstr(rf_lr)}). Deep and graph networks with up to {int(PAR.max())} parameters therefore "
        "added no measurable value over a linear model, whereas the raw-signal 1D-CNN without these features failed. "
        "The useful information sits in the hand-crafted spectral features, not in the architecture. Graph message "
        "passing between zones did not matter: the no-edge ablations were indistinguishable from the full graph "
        "models. With three zones per wall and one kinematic edge type, a graph is a weak inductive bias. The "
        "heat-kernel edges of an earlier version were constant across trials and were removed.")
    ms.para(
        "Two observations limit even this modest result. First, within-trial ranking was near chance for every model "
        "except logistic regression, so the models mostly separate trials rather than locate instability along a wall. "
        f"Second, removing the two weakly aligned trials, which contain no unstable zone, lowered the pooled AUC of "
        f"{AL_DROP_TXT} "
        "(Figure 3), consistent with part of the discrimination coming from easy-to-separate negative trials.")
    ms.para(
        "The labels are the larger problem. A fixed spectral rule is transparent and reproducible, but its output "
        f"changed from {n_pos15} to {n_pos20} to {n_pos25} unstable zones as the threshold moved from 15 to 25 dB. "
        f"Replacing the measured by the nominal spindle frequency changed {int(SENS[(SENS.spindle_frequency == 'nominal') & (SENS.tolerance_hz == 2.0) & (SENS.threshold_db == 15.0)].iloc[0].n_changed_vs_main)} "
        "labels at 15 dB, so labels built on nominal speeds, as in the earlier pipeline, are partly arbitrary. "
        "Without an independent reference, such as chatter marks on the surface, zone roughness or expert-labelled "
        "signals, the reported AUC measures agreement with a rule, not detection of chatter. External validation of "
        "the rule on the public turning dataset [6], whose signals were labelled by inspection, is the most direct "
        "next check (Section 5).")
    ms.para(
        "For WAAM machining, the results argue against reading material differences into monitoring scores on data of "
        "this size. Porosity, magnesium loss and layer-wise reheating [1–3] are plausible reasons for zone-to-zone "
        "variation in cutting dynamics, and position-dependent wall stiffness adds to them. Testing these mechanisms "
        "requires confirmed material identity per trial, repeated walls per material and zone-resolved surface "
        "measurements. None of these is available here, and no conclusion about WAAM versus wrought behaviour is "
        "drawn.")
    lr_na, lr_nf = met("logreg", "ablation_15db", "no_acceleration"), met("logreg", "ablation_15db", "no_force")
    ms.para(
        "Practically, dynamometers are rarely available on production machines. For logistic regression, force "
        f"features alone gave an AUC of {auc_ci('logreg', 'ablation_15db', 'no_acceleration')} and acceleration "
        f"features alone {auc_ci('logreg', 'ablation_15db', 'no_force')}, with overlapping intervals. Because the "
        "labels are themselves derived from the force spectrum, force features have an inherent advantage here. "
        "The accelerometer's model and mounting were not recorded, so the acceleration-only result cannot yet be "
        "generalised.")

    ms.heading("5. Limitations")
    ms.bullets([
        "**Material mapping unresolved.** No trial with a force record has a confirmed material; the specimen grouping "
        "rests on timing, measured speed and file provenance. No material-specific result is reported.",
        f"**Label threshold sensitivity.** Labels change strongly between 15, 20 and 25 dB; only 15 dB yields enough "
        "unstable zones for evaluation, and the rule has not been validated against an independent reference.",
        f"**Small sample.** {c('n_trials_used')} trials and {c('n_zones')} zones; unstable zones are clustered in few "
        "trials, so confidence intervals are wide and per-fold AUC is mostly undefined.",
        "**No zone roughness.** Zone-level Ra was not measured, so no roughness model is fitted; specimen-mean Ra "
        "from [1] cannot serve as a zone target.",
        "**Infrared not used.** Thermal images were recorded in [1] but could not be calibrated (vendor format, "
        "emissivity and frame rate not recorded, no thermocouple reference, no link to trial numbers).",
        "**Accelerometer metadata missing.** Model, sensitivity and mounting position are unknown; two trials have weak "
        "force–acceleration alignment.",
        "**Dynamometer dependence.** Force-based features are not practical on production machines.",
        "**External validation not yet run.** The public turning dataset [6] was not available locally when this "
        "draft was prepared, so the planned check of the labelling rule against its published labels and the "
        "leave-one-overhang-out feature test were not run. The KIT milling dataset [7,8] contains a single chatter "
        "trial and was not used.",
        "**Compute-driven settings.** Window count per zone and epoch count were fixed by the compute budget before "
        "any model was evaluated; they were not tuned."])

    ms.heading("6. Conclusions")
    ms.para(
        "A leakage-free re-analysis of force and vibration records from thin-wall milling of WAAM and wrought Al–Mg "
        "walls showed that: (i) the archived trial-to-condition mapping was partly wrong, and the measured spindle "
        "speed and force level allow it to be reconstructed without assuming material; (ii) spectral chatter labels "
        "depend on the prominence threshold more than on any classifier; (iii) on this small dataset, simple "
        "models on physics-based spectral features perform as well as deep and graph networks, graph edges between "
        "zones add no measurable value, and a raw-signal CNN without these features fails; and (iv) most of the "
        "discrimination separates whole trials rather than locating unstable zones within a wall. Before stronger "
        "modelling claims can be made, the confirmed trial log, "
        "repeated walls, zone-level surface measurements and an externally validated labelling rule are needed.")


def back_matter(ms: Manuscript):
    ms.para("**Supplementary Materials:** The planned material-specific analysis is omitted because no trial with a "
            "force record has a confirmed material (Section 3.1). Code, result tables (results/*.csv) and the audit "
            "are provided in the repository.", indent=False)
    ms.para("**Author Contributions:** [author to complete].", indent=False)
    ms.para("**Funding:** [author to complete].", indent=False)
    ms.para("**Data Availability Statement:** Processed results, code and audit files are available at [repository "
            "DOI – author to supply]. Raw force and vibration records are available from the authors on request. The "
            "public datasets referred to are the turning chatter dataset of Khasawneh, Otto and Yesilli "
            "(Mendeley Data, doi:10.17632/hvm4wh3jzx.1) [6] and the KIT multimodal CNC milling dataset "
            "(doi:10.35097/hvvwn1kfwf7qt48z) [8], described in [7]; neither was used in the analyses reported here.",
            indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    ms.para("**Acknowledgments:** During the preparation of this manuscript, the authors used Claude (Anthropic) to "
            "audit and rewrite the analysis code, run the analyses and draft the text from the computed results. The "
            "authors reviewed and edited the output and take full responsibility for the content of this publication.",
            indent=False)
    ms.para("**Conflicts of Interest:** The authors declare no conflicts of interest.", indent=False)
    ms.heading("References")
    refs = [
        "Sundaram, S.; Mohanavelu, T.; Arangot, S.; Petru, J.; Ruthramoorthy, M.; Rowther, R.S.; Senthilkumar, K. "
        "Comparative Evaluation of Mechanical Behaviour and Machinability of WAAM-Fabricated Aluminium Alloys. "
        "*Materials* **2026**, *19*, 2653. https://doi.org/10.3390/ma19122653",
        "Wu, B.; Pan, Z.; Ding, D.; Cuiuri, D.; Li, H.; Xu, J.; Norrish, J. A review of the wire arc additive "
        "manufacturing of metals: properties, defects and quality improvement. *J. Manuf. Process.* **2018**, *35*, "
        "127–139. https://doi.org/10.1016/j.jmapro.2018.08.001",
        "Zhao, D.; Long, D.; Niu, T.; Zhang, T.; Hu, X.; Liu, Y. Effect of Mg Loss and Microstructure on Anisotropy of "
        "5356 Wire Arc Additive Manufacturing. *J. Mater. Eng. Perform.* **2022**, *31*, 8473–8482. "
        "https://doi.org/10.1007/s11665-022-06802-8",
        "Yesilli, M.C.; Khasawneh, F.A.; Otto, A. Chatter Detection in Turning Using Machine Learning and Similarity "
        "Measures of Time Series via Dynamic Time Warping. *arXiv* **2019**, arXiv:1908.01678.",
        "Yesilli, M.C.; Khasawneh, F.A.; Mann, B.P. Transfer Learning for Autonomous Chatter Detection in Machining. "
        "*arXiv* **2022**, arXiv:2204.05400.",
        "Khasawneh, F.; Otto, A.; Yesilli, M. Turning Dataset for Chatter Diagnosis Using Machine Learning. Mendeley "
        "Data, V1, 2019. https://doi.org/10.17632/hvm4wh3jzx.1",
        "Ströbel, R.; Kuck, M.; Oexle, F.; Kader, H.; Puchta, A.; Noack, B.; Fleischer, J. A multimodal dataset for "
        "process monitoring and anomaly detection in industrial CNC milling. *Data Brief* **2025**, *63*, 112207. "
        "https://doi.org/10.1016/j.dib.2025.112207",
        "Ströbel, R.; Kuck, M.; Oexle, F.; Kader, H.; Puchta, A.; Noack, B.; Fleischer, J. A Multimodal Dataset for "
        "Process Monitoring and Anomaly Detection in Industrial CNC Milling [Dataset]. Karlsruhe Institute of "
        "Technology, 2025. https://doi.org/10.35097/hvvwn1kfwf7qt48z",
        "Kipf, T.N.; Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of "
        "the International Conference on Learning Representations (ICLR), 2017. arXiv:1609.02907.",
        "Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. Graph Attention Networks. In "
        "Proceedings of the International Conference on Learning Representations (ICLR), 2018. arXiv:1710.10903.",
    ]
    for k, ref in enumerate(refs, 1):
        p = ms.doc.add_paragraph()
        p.paragraph_format.left_indent = __import__("docx").shared.Cm(0.7)
        p.paragraph_format.first_line_indent = __import__("docx").shared.Cm(-0.7)
        ms._add_runs(p, f"{k}.\t{ref}", size=9)


if __name__ == "__main__":
    out = Path(os.environ.get("WAAM_MANUSCRIPT_OUT", ROOT / "paper" / "manuscript.docx"))
    build().save(out)
    print("wrote", out)
