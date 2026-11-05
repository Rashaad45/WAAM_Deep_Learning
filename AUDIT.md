# AUDIT.md — Step 0 (read-only audit)

Date: 2026-11-05. Scope: whole codebase in `src/`, `run_pipeline.py`, `data/`, `outputs/`, `report/`, and the external files.
Nothing in the pipeline, data or manuscript was changed. Files written in Step 0: this file, `requirements.txt` (pinned), `external/DATASETS.md`, and `audit/` (the read-only analysis scripts and their outputs, so every number below can be regenerated).

Reproduce the audit numbers:
```
python audit/audit_analysis.py .        > audit/audit_analysis_output.txt
python audit/audit_analysis2.py .       > audit/audit_analysis2_output.txt
python audit/label_sensitivity.py . audit
python audit/inspect_docs.py . audit
```

> **Scope amendment (2026-11-05, after the audit; binding for Steps 1–5): see section H.**
> **IR_EXPORT = no, permanently.** Thermal imaging is out of the project. Where an item below proposes thermal work, section H overrides it.

Verdict legend: **CONFIRMED** (shown in code and/or numbers), **PARTLY** (part confirmed, part different from the brief), **NOT CONFIRMED** (the suspected cause is not what happens; the real cause is given).

---

## A. Environment and folder layout

| Check | Result |
|---|---|
| Python | 3.11.4 |
| Packages | numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, PyWavelets 1.8.0, matplotlib 3.11.2, python-docx 1.2.0, torch 2.6.0 (+cu124), pypdf 6.20.1, h5py 3.14.0. `requirements.txt` now pins these. The old unpinned list is kept as a comment. |
| Seeds | `train_evaluate.py:73-77` seeds `random`, `numpy` and `torch`. It does **not** set `torch.use_deterministic_algorithms`, `PYTHONHASHSEED` or the cuDNN flags. `config.py:151` defaults to `seeds=(42,)`, but the reported run used 5 seeds via the CLI (`outputs/training_log_main.txt`). The inner-validation RNG is fixed (`train_evaluate.py:442`). To fix in Step 1: one seed list in config and deterministic flags. |
| LibreOffice / poppler | **Not installed** (`soffice`, `pdftoppm` not found). Step 5 needs LibreOffice for docx→PDF and page rendering. |

**Layout differs from the brief:**

| Expected | Actual |
|---|---|
| `external/papers/{2204.05400v1,1908.01678v1,main}.pdf` | At the project root. |
| `external/_quarantine/*.mat` | `robotConfigurations.mat`, `dataSetRobotConfig1_int.mat`, `dataSetRobotConfig2_int.mat` are at the project root (**not** quarantined). |
| `external/turning_chatter_mendeley/` | **Missing** (not downloaded). |
| `external/kit_cnc_milling/` | **Missing** (not downloaded). |
| `WAAM_STGNN_Materials_REVISED.docx` | **Missing.** Present instead: `report/WAAM_STGNN_Materials_{draft,filled,FINAL}.docx`. |
| `WAAM_STGNN_Materials_humanized.pdf` | Present at `report/`. |
| Unlisted root files | `images.jpg`, `images (1).jpg`, `Screenshot 2026-10-31 161338.png`. Not referenced by any code; I did not open them. |

**Update 2026-11-05:** with the author's approval, the three PDFs were moved to `external/papers/` and the three `.mat` files to `external/_quarantine/`. No other file was moved.

---

## B. NEW critical finding (not in the brief) — the material label of every trial is an assumption, and the archive contradicts it

- `data/trial_manifest.csv` column `material_source` says, for every trial: *"author assumption: trials 1-10 wrought, 11-20 WAAM (no lab log in archive)"*. The speed/depth per trial is also *"Table 3 order of ref. [1], mapped by trial number (provisional)"*. `README.md:85-86` says the same.
- `data/reference/trial_mapping_evidence.csv` contradicts that assumption:
  - **trial_01 and trial_03 are WAAM (high confidence).** Their vibration files are byte-identical (MD5) to files under `VIBRATION DATA/AM ER5356/ACCELEROMETER/TRIAL 1|3/`. Under the assumption, trials 1–10 are *wrought*.
  - **The specimen change is probably between trial_11 and trial_12, not between 10 and 11.** The inter-trial gap is 646 s there against a median of ~92 s. The force files also switch folder at trial_12 (`25.03.25.01` → `25.03.25.02`, see `trial_manifest.csv`).
  - For trials 2 and 4–20, no file evidence assigns a material.
- Force-file pairing confidence is **low** for trial_16 and trial_18, and **medium** for trial_10 and trial_13. For trial_16 the two PC clocks differ by 102 s (`force_start_time` 15:57:36 vs `acceleration_start_time` 15:59:18). The aligner only searches ±6 s (`config.py:93`), so it cannot detect a mispairing.

**Consequences.** Everything material-dependent rests on this unverified split:
- the imputed Ra (1.22 / 0.83 µm);
- the "all 10 unstable zones are WAAM" statement;
- the WAAM-vs-wrought comparisons;
- the material-flag ablation;
- the within-WAAM analysis requested in item 14.

trial_16 holds 5 of the 10 positives and has the weakest file pairing.

**Needed from you (blocking for Step 1):** the lab log or notebook giving material, spindle speed and depth for each trial number, and confirmation of the trial_16 / trial_18 force files. Without it, the paper can only say the material assignment is provisional, and that undermines Framings A and B.

---

## C. Code and analysis bugs (items 1–10)

### 1. Loss weighting — **CONFIRMED (paper ≠ code)**
- Code: `train_evaluate.py:343-352` returns `bce + regression_weight * mse`, with `regression_weight = 1.0` (`config.py:150`, comment *"loss = BCE + lambda * MSE"*). The Ra head is trained with **equal weight** to the BCE, not zero weight.
- Manuscript (`report/WAAM_STGNN_Materials_FINAL.docx`, eq. near para 107): `L = λ·L_BCE + (1−λ)·L_MSE + β‖Θ‖²`. Para 111 says *"The fixed-weight loss with λ = 1 was retained"*. Read literally, that gives the Ra head zero weight, which contradicts the reported Ra results.
- The `β‖Θ‖²` term is implemented as Adam `weight_decay=1e-4` (`train_evaluate.py:359-360`). That is coupled L2, not an explicit loss term; the paper should say so.
- The uncertainty-weighted loss (para 110) is not in the code at all.
- The BCE uses `pos_weight = n_neg/n_pos` of the training fold (`train_evaluate.py:364-365`), about 8.6–18 depending on the fold. So the effective BCE:MSE balance changes from fold to fold.
- **Fix:** the paper states `L = L_BCE(pos_weight) + λ_R·L_MSE`, λ_R = 1, with decoupled weight decay stated separately. Or, under Framing C, the Ra head is removed.

### 2. Material-flag ablation — **NOT CONFIRMED as a plumbing bug; the real cause is training/early stopping**
- The flag is appended as column 65 (`graph_data.py:164-169`) and standardised with training statistics (`train_evaluate.py:131-135, 149`). Because both materials are in every training fold, its std is 0.46–0.49. It reaches the network as a constant **−1.50 (wrought) / +0.79 (WAAM)** in every test fold (`audit/audit_analysis2_output.txt`). It is not dropped, zeroed or mis-indexed.
- **Why RMSE stays at 0.098:** the per-material *mean* prediction is already right without the flag (GAT full: 1.190 / 0.869 µm vs truth 1.22 / 0.83). Material is already linearly decodable from the features; logistic regression separates it. The residual error is **within-material zone-to-zone scatter** (GAT+flag: RMSE_WAAM 0.116, RMSE_wrought 0.054). The flag cannot remove that scatter, because the network is not trained to convergence:
  - early stopping fires after the first epoch in many folds (item 7);
  - the joint loss is dominated by the pos-weighted BCE;
  - 64 other noisy standardised inputs pass through dropout.

  A fair "material oracle" (predict the training mean of the test trial's material) gives RMSE **0.000**, as expected.
- **Fix:** a unit test that a flag-only linear model reaches RMSE ≈ 0 on two-valued targets, plus a material-oracle reference row (item 9). Ra results must be stated as "material recognition", or dropped (Framing C).

### 3. Below-chance AUCs — **PARTLY CONFIRMED: no misalignment, no inversion; the values are consistent with chance**
Numbers from `audit/audit_analysis_output.txt` (seeds averaged per zone, 96 zones, 10 positives):

| Model | Pooled AUC | Perm-null 95% band | Perm p (2-sided) | Within-trial-centred AUC | WAAM-only AUC | Per-fold AUC (trials 12/14/15/16/17) | r(train prevalence, mean held-out prob) |
|---|---|---|---|---|---|---|---|
| Logistic regression | **0.888** | 0.31–0.69 | <0.001 | 0.683 | 0.818 | 0.38/1.00/0.80/0.60/0.80 | −0.85 |
| Random forest | 0.717 | 0.31–0.69 | 0.022 | 0.571 | 0.523 | 0.56/1.00/0.40/1.00/0.20 | −0.21 |
| SVM (RBF) | 0.524 | 0.31–0.69 | 0.79 | 0.713 | 0.438 | 0.50/1.00/0.80/1.00/0.40 | 0.08 |
| 1D-CNN | 0.460 | 0.31–0.69 | 0.68 | 0.490 | 0.502 | 0.38/0.40/1.00/0.80/0.20 | 0.22 |
| BiLSTM | 0.308 | 0.31–0.69 | 0.043 | 0.274 | 0.254 | 0.25/0.00/0.40/0.20/0.40 | 0.25 |
| Fusion MLP | 0.635 | 0.31–0.69 | 0.17 | 0.598 | 0.592 | 0.38/0.20/0.80/0.60/0.60 | −0.28 |
| ST-GCN | 0.550 | 0.31–0.69 | 0.62 | 0.514 | 0.460 | 0.50/0.00/0.80/0.20/0.60 | −0.01 |
| ST-GAT | 0.471 | 0.31–0.69 | 0.77 | 0.462 | 0.404 | 0.50/0.00/0.80/0.40/0.60 | 0.04 |

- **(a) Label/zone misalignment: not present.** `y_class` in `predictions_oof.csv` matches `zone_labels.csv` on all 3072 rows. Each fold holds exactly one trial. Window → zone indexing is by construction (`prepare_dataset.py:310-333`).
- **(b) Inversion: not present.** Probabilities come from `sigmoid(logits)` (`train_evaluate.py:483`), and BCE-with-logits uses the same `y_class` (`:346`). sklearn uses `predict_proba[:, 1]` (`:309`).
- **(c) Prior-shift artefact: real in principle, but it does not explain these values.** For the deep models, the correlation between fold training prevalence and the mean held-out probability is about 0 (−0.28 to +0.25). Centring scores within each trial, which removes any per-fold calibration offset, leaves the deep AUCs where they were (0.27–0.60). LR shows r = −0.85 only because it correctly gives trial_16 (5 positives, lowest training prevalence) a mean probability of 0.90.
- **Where chance sits.** With 10 positives in 96, the zone-label permutation null has a 95% band of **0.31–0.69**. ST-GAT, ST-GCN, 1D-CNN, fusion MLP and SVM are all inside it. BiLSTM is at the lower edge (p = 0.043, not robust to multiplicity). **Only logistic regression is clearly above chance.** This zone-level permutation ignores within-trial clustering, so a trial-structured null would be even wider.
- **Per-fold AUC** is defined for only 5 of 16 folds (the only trials containing both classes), each on 6 zones. It cannot be estimated well; report it but do not test on it.
- **Extra cause found:** in folds 1 and 6 the inner validation set contains trial_16, so the model trains with only **5** positives. See item 7.

### 4. Window → zone aggregation — **CONFIRMED: inconsistent across models**
- Rule used (at representation level, not prediction level):

| Model | Location | Rule |
|---|---|---|
| ST-GCN / ST-GAT | `models.py:228` | mean over the T = 73 window embeddings |
| Fusion MLP | `models.py:270` | mean of per-window MLP outputs |
| 1D-CNN | `models.py:305` | mean of encoder outputs |
| **BiLSTM** | `models.py:285-287` | **last** forward and backward hidden states, not a mean |
| Classical models | `train_evaluate.py:289` | mean of the raw window features |

- **Fix (Step 1):** one documented rule for all models. I recommend the mean over windows at representation level, with BiLSTM switched to the mean of its outputs. A unit test asserts every model follows it.

### 5. Harmonic definitions — **CONFIRMED, plus a resolution problem**
- Label: `signal_features.py:209-219` excludes ±2 Hz (`config.py:75`) around **spindle-rotation** harmonics (16.67 / 23.33 Hz spacing) on a PSD with 0.763 Hz bins.
- Window feature: `signal_features.py:79-98` excludes ±½ bin = **±12.2 Hz** around **tooth-passing** harmonics (66.7 / 93.3 Hz) on a 1024-point Welch PSD with **24.4 Hz bins**.
- At 24.4 Hz resolution the window feature *cannot* resolve spindle harmonics, whose spacing is smaller than one bin. Its "synchronous" mask is also a different set from the label's.
- **Proposed consistent definition (decide now, before reruns):** *spindle-rotation harmonics* everywhere. With runout, a 4-flute cutter forces vibration at every multiple of the spindle frequency, not only at tooth-passing multiples.
  - The label keeps spindle harmonics.
  - The window chatter indicator is computed on a **0.5 s context PSD** centred on each 100 ms window (2 Hz bins), with the same spindle-harmonic mask and tolerance as the label.
  - All other window features stay at 100 ms.
  - Justification: forced response sits at spindle-rotation multiples; tooth-passing is a subset of them.
- Alternative if you prefer to keep 100 ms everywhere: drop the chatter indicator from the window features rather than keep a mask that does not match the label.

### 6. DeLong on correlated zones — **CONFIRMED**
- `train_evaluate.py:226-250` (implementation) and `:686-698` (use). It compares the GAT and fusion-MLP AUCs on 96 zones as if they were independent; p = 0.0148 (`outputs/stat_tests.csv`).
- The Wilcoxon test on fold RMSE (`:700-708`) is trial-level but is run on the imputed Ra.
- **Fix:** a trial-level paired bootstrap of ΔAUC (resampling trials, seeds averaged) plus a trial-level paired sign-flip permutation test. DeLong is removed.

### 7. Early stopping — **CONFIRMED**
- `FoldSplitter.pick_validation` (`train_evaluate.py:116-125`) draws 1 WAAM + 1 wrought trial at random. **6 of 16 folds have zero positives in validation** (`audit/audit_analysis2_output.txt`), so early stopping is driven only by negatives in those folds.
- In 2 folds, validation holds trial_16 (5 positives), which removes half the positives from training.
- Epochs trained, from `predictions_oof.csv`:
  - The minimum is 11 for every model. Under patience = 10 (`config.py:143`, logic `train_evaluate.py:386-396`), 11 means the best validation loss came at **epoch 1**.
  - Median 12 epochs for 1D-CNN, 15 for BiLSTM, 19 for GAT.
- **Fix (recommended):** a fixed epoch count, chosen **before** the reruns from training-loss convergence on the training trials only. No early stopping, and no test-fold look. Fallback: a stratified inner split that always puts one positive-containing trial in validation. That is often impossible here, since only 5 trials have positives.

### 8. Constant adjacency — **CONFIRMED**
- `graph_data.py:57-85` computes one adjacency and `:104, :124` copies it to every trial. All 16 trials are identical (checked).
- The code's kernel is `exp(−d²/(4α·d/v)) = exp(−d·v/(4α))`. With α = 4.88e-5 m²/s and v = 1.33 mm/s, A_th = **0.97 / 0.93 / 0.90 / 0.87 / 0.84** for neighbours 1–5 apart. That is near-uniform all-to-all smoothing.
- The diffusion length for Δt = 3.75 s is **27.1 mm**, which is the wall width (30 mm).
- With the kinematic channel added, the GCN normalised adjacency is dense and almost constant. This explains the uniform GAT attention, and it is consistent with the ablations: `no_spatial_edges` AUC 0.56, `without_thermal_edges` 0.56, full 0.47 (all inside the null band).
- The edges carry no sample-specific information unless measured IR is supplied (DATA STATUS: IR_EXPORT).

### 9. "Predict training mean" baseline — **CONFIRMED**
- `train_evaluate.py:321-324` predicts the global training mean. Under LOTO with two-valued targets, it is pulled toward the other material:
  - prediction 1.064 µm for WAAM vs 1.090 µm for wrought;
  - RMSE 0.201 overall, 0.260 on wrought.
- The fair reference is the **training mean of the test trial's material**, which gives RMSE 0.000 for imputed Ra. That shows the 0.098 µm "improvement" is 0.098 µm *worse* than a material lookup.
- With measured zone Ra, report both references, plus RMSE/σ_meas.

### 10. Capacity vs data — **CONFIRMED**

| Model | Parameters |
|---|---|
| ST-GAT | 68,082 |
| ST-GCN | 67,938 |
| BiLSTM | 66,818 |
| Fusion MLP | 13,986 |
| 1D-CNN | 5,730 |
| Logistic regression | 65 |

Data: 96 zones, 10 positives in 5 trials. All counts will be reported, and a ≤ 5k-parameter GNN added in Step 3.

---

## D. Experimental and labelling problems (items 11–16)

### 11. Label sensitivity — **CONFIRMED (the threshold matters far more than the tolerance)**
- Recomputed from raw force with the same cuts and zones. This reproduces the stored labels exactly at ±2 Hz / 20 dB. Sources: `audit/label_sensitivity_grid.csv`, `audit/label_sensitivity_zones.csv`.

| Tolerance (Hz) | Prominence (dB) | Unstable (WAAM / wrought) | Labels changed vs baseline | Trials with positives |
|---|---|---|---|---|
| 0.5 | 15 | 60 (49 / 11) | 50 | 14 |
| 0.5 | 20 | 11 (11 / 0) | 1 | 5 |
| 0.5 | 25 | 0 | 10 | 0 |
| 1.0 | 15 | 55 (45 / 10) | 45 | 13 |
| 1.0 | 20 | 10 (10 / 0) | 0 | 5 |
| 1.0 | 25 | 0 | 10 | 0 |
| 2.0 | 15 | 46 (37 / 9) | 36 | 13 |
| 2.0 | 20 | **10 (10 / 0)** baseline | 0 | 5 |
| 2.0 | 25 | 0 | 10 | 0 |

- **27 of 96 zones** have prominence within ±2 dB of the 20 dB threshold.
- The WAAM median prominence is ~18 dB and the wrought median ~14 dB. So the label behaves mainly as **"WAAM zone whose ~3.45 kHz mode is a bit more prominent"**, not as a clear chatter/no-chatter separation.
- At 15 dB, wrought zones become unstable too, which removes the "all positives are WAAM" pattern.
- The tolerance matters less than feared at 20 dB, because only 1 zone with prominence ≥ 20 dB lies on a harmonic. That zone is trial_16 zone 1: 3477.5 Hz, 0.81 Hz from 149 × 23.33 Hz, labelled stable. Its neighbours at 3484.3 and 3485.9 Hz (7.7 and 9.2 Hz off) are labelled unstable. This is the Fig. 2 flip, confirmed.
- Chance that a random peak falls within the tolerance of a spindle harmonic:

| Speed (spindle frequency) | ±0.5 Hz | ±1 Hz | ±2 Hz |
|---|---|---|---|
| 1000 rpm (16.67 Hz) | 6.0% | 12.0% | **24.0%** |
| 1400 rpm (23.33 Hz) | 4.3% | 8.6% | **17.1%** |

  ±0.5 Hz is below one PSD bin (0.763 Hz), so it is not meaningful at this resolution.
- No external or physical ground truth anchors the 20 dB choice. The repository cannot verify the claim that it was "fixed before training". It needs either E1 (turning labels) or surface evidence such as chatter marks.

### 12. Zone definition — **CONFIRMED; decision recorded here, before any rerun**
- Current layout: 6 equal zones over the *detected* cut (`config.py:55-56`, `prepare_dataset.py:301-305`).
  - Detected cuts last 20.95–24.34 s = 27.9–32.5 mm at 80 mm/min. The nominal zone is 3.75 s; actual zones are 3.49–4.06 s.
  - Per the manuscript (para 42), this is face milling with a 4-flute **Ø10 mm** end mill across the 30 mm wall width. The cutter footprint along the feed is therefore up to 10 mm, i.e. it spans **two** 5 mm zones at every instant.
  - Zones 1 and 6 contain the engagement ramp at entry and exit.
- **Decision** (made from geometry only; no result of the new layout has been computed or seen):
  - **3 zones**, each one third of the detected cut (≈ 9.3–10.8 mm, about one tool diameter).
  - Entry/exit exclusion: windows whose tool centre is within **one tool radius (5 mm = 3.75 s)** of the detected cut start or end are dropped, for both features and the label PSD.
  - Zones 1 and 3 therefore keep ≈ 4.3–5.8 mm (≈ 3.2–4.3 s, ≥ 2 Welch segments at 32768 points). Zone 2 is complete.
  - The paper will state that a zone is a *tool-centre position bin*, and that each window's signal integrates material within ±5 mm of the centre.
- **Reason:** making the zones as wide as the cutter footprint limits the overlap to adjacent-zone edges. Removing one radius at each end removes the engagement transient for full-immersion entry. Three nodes still form a (small) graph.
- **Cost:** 48 zones instead of 96. The graph has only 3 nodes, which weakens any graph claim. That is a property of the specimen geometry, not something to tune away.
- **Assumption to confirm:** "depth of cut 0.5–1.5 mm" is the axial depth of a face pass across the full 30 mm width with the tool axis normal to the wall. If the cut was a partial-immersion peripheral pass, the footprint is shorter (√(D·a_e) ≈ 2.2–3.9 mm), and I would revise this before any rerun.

### 13. Force–acceleration alignment — **CONFIRMED**
- Source: `data/processed/dataset_summary.csv`, aligner at `prepare_dataset.py:147-197`. The aligner correlates the accel log-envelope with a 0/1 cut template from the force record, searching ±6 s.
- `min_alignment_r = 0.3` (`config.py:94`) only labels a trial "WEAK"; it never excludes one (`prepare_dataset.py:261-264`).

| Trial | r | Trial | r | Trial | r | Trial | r |
|---|---|---|---|---|---|---|---|
| 04 | **0.431** | 09 | 0.826 | 13 | 0.797 | 17 | 0.968 |
| 06 | **0.652** | 10 | 0.823 | 14 | 0.847 | 18 | 0.939 |
| 07 | 0.838 | 11 | 0.846 | 15 | 0.771 | 19 | 0.946 |
| 08 | 0.706 | 12 | 0.888 | 16 | 0.959 | 20 | 0.972 |

- r < 0.7: **trial_04, trial_06**, both wrought. trial_08 is borderline at 0.706.
- Step 1 adds an `accel_ok` flag. Acceleration-dependent models get a sensitivity run without trials 04/06.
- trial_16's high r (0.959) does not settle the force-file pairing in section B, because the lag search cannot reach the 102 s clock offset.

### 14. Confound — **CONFIRMED (and it rests on section B)**
- All 10 positives are in trials labelled WAAM (12, 14, 15, 16, 17). Pooled AUC partly measures material discrimination.
- WAAM-only AUC (60 zones, from existing predictions):

| Model | WAAM-only AUC |
|---|---|
| Logistic regression | 0.818 |
| Fusion MLP | 0.592 |
| Random forest | 0.523 |
| 1D-CNN | 0.502 |
| ST-GCN | 0.460 |
| SVM | 0.438 |
| ST-GAT | 0.404 |
| BiLSTM | 0.254 |

- Step 3 will report within-WAAM and pooled results side by side.

### 15. Missing metadata — **CONFIRMED**
- Manuscript para 44 (accelerometer model and mount: `[author to supply]`) and para 45 (IR camera model, frame rate, emissivity: `[author to confirm]`).
- Para 130 (`[author to …]` in the software/repository text), paras 417–420 (author contributions, funding, repository DOI, acknowledgments).
- `data/ra_measurements.csv` is an **empty template**: 96 rows, no values.
- The DATA STATUS block in your prompt is **unfilled** (ZONE_RA, IR_EXPORT, NEW_TRIALS, ACCEL_INFO, EXTERNAL_DOWNLOADED). I cannot fill any of these from the archive.

### 16. Broken PDF — **PARTLY CONFIRMED (cause narrowed, not proven)**
- `report/WAAM_STGNN_Materials_humanized.pdf` (17 pp.) was made by **LibreOffice 24.2 Writer** on 2026-11-01 17:22Z. Display and inline math is blank, e.g.:
  - p. 5: *"a sequence of graphs , ."*
  - p. 6: *"self-loops are added, , with degree matrix ."*
  - p. 4: *"divided into  cutting zones"*
- The newest docx in the folder, `report/WAAM_STGNN_Materials_FINAL.docx`, still holds **99 valid `m:oMath` elements** (40 display), correctly placed as children of `w:p` with the `m:` namespace declared. The same sentence there contains the `N` equation.
- So the math was lost **after** `FINAL.docx`, at one of two points:
  - (i) the "humanize" rewrite (its source `.docx`, the expected `WAAM_STGNN_Materials_REVISED.docx`, is not in the folder); or
  - (ii) LibreOffice's OMML import during headless conversion.
- The blank gaps keep their surrounding spaces, which fits OMML not being rendered on conversion. I cannot reproduce it here because LibreOffice is not installed.
- A related latent bug: `src/make_report.py:223-241` (`DocxEditor.set_paragraph`) **deletes every equation** in any paragraph it rewrites (line 238-239).
- **Cyan-highlight note:** not found.
  - 0 highlights, revisions or comments in draft, filled or FINAL `.docx`.
  - No non-black fill colours in any PDF page's content stream.
  - I cannot render pages (no poppler or LibreOffice). Please tell me the page, or provide the REVISED docx.
  - Step 5 builds a fresh docx, so the note cannot carry over either way.

---

## E. External files (items 17–18)

### 17. `*.mat` robot files — **CONFIRMED; keep quarantined, do not use or cite**
- All three are MATLAB 5.0 MAT-files (PCWIN64), created 14 Sep 2017 (`robotConfigurations`) and 2 Oct 2017 (`dataSetRobotConfig{1,2}_int`).
- `dataSetRobotConfig{1,2}_int.mat` hold a 1×1 struct `dataSet` with fields `training` and `test`. `robotConfigurations.mat` holds a 2×7 numeric array `robotConfigurations`.
- I did not re-derive the row counts or column semantics beyond this; your description (tabular stability-lobe style labels, no sensor signals) is consistent with what is there. They contain no time series, so they cannot test any signal-based method in this paper.
- There is no source paper or licence. **They will not be used or cited.** They are now in `external/_quarantine/` and excluded from version control.

### 18. External datasets — see `external/DATASETS.md`
- Facts were checked against the three PDFs; the datasets themselves are **not downloaded**, so their contents and licences are unverified.
- Two corrections or additions to the brief:
  - Yesilli et al. 2019 report their accuracies **within each overhang** (random train/test splits per stickout). They are not leave-one-overhang-out, so the E2 comparison is not like-for-like.
  - Ströbel et al. state a complementary second-machine dataset exists (DOI 10.35097/vnnu3n9z7ndsnhfd). Noted only; it will not be used unless you ask.

---

## F. Other issues found during the audit

1. **Labelling threshold provenance.** 20 dB, the 2.5–5.5 kHz band and ±2 Hz are all `[choice]` in `config.py`. Nothing in the repo shows they were fixed before results.
2. **Cut-detection threshold** (30% of the rise, `config.py:90`) sets the zone boundaries, and the detected cuts vary from 27.9 to 32.5 mm against a 30 mm wall.
3. **Imputed Ra.** `Ra_um` is the material constant for all 96 zones (`prepare_dataset.py:368-387`). Every reported R² (0.73 etc.) measures how well the models separate the two materials. It says nothing about roughness.
4. **Latency claim** (`train_evaluate.py:511-546`). Timing is for one window through an untrained model on a 6-node graph. That is fine as a compute figure, but it says nothing about detection delay.
5. **Ablation bookkeeping.**
   - `without_force_features` zeros the force columns in `x`, the raw force windows **and** the force CNN output (`models.py:212-215`).
   - The 1D-CNN baseline has no hand-crafted features while the GNN does, so the comparison mixes input sets with architecture.
6. **The attribution** (`train_evaluate.py:401-428`) is integrated gradients on the **imputed Ra** output. It attributes material recognition, not roughness.

---

## G. What Step 1 will change (pending your answers)

**Blocking — I need these from you:**

1. The material, speed and depth per trial from the lab log (section B), and confirmation of the trial_16 / trial_18 force files.
2. The filled DATA STATUS block (ZONE_RA, IR_EXPORT, NEW_TRIALS, ACCEL_INFO, EXTERNAL_DOWNLOADED).
3. Approval of the decisions recorded above:
   - zones: 3 × ⅓ of the cut, minus one tool radius at each end (item 12);
   - harmonics: spindle harmonics everywhere, with a 0.5 s context for the window chatter indicator (item 5);
   - early stopping replaced by a fixed epoch count chosen on training loss (item 7);
   - mean-over-windows aggregation for every model (item 4).
4. Permission to move the PDFs and `.mat` files into `external/papers/` and `external/_quarantine/`.

**Then:** fix items 1–10 and 12–13, and add the unit tests listed in the brief. Their output will be pasted into this file.

---

## H. Scope amendment — thermal imaging removed (decided 2026-11-05; applies to every later step)

**IR was recorded, but it is not used.**
- 28 radiometric `.irgd` files (213 MB) from a Sonel thermal imager are in `WAAM/RESULTS-*-1-001/RESULTS/DATA/thermal camera/`, plus plots in `RESULTS/PLOTS/THERMAL/`.
- Reasons they are not used:
  1. The `.irgd` format needs the vendor software (Sonel ThermoAnalyze) for a calibrated temperature export; that export was never made.
  2. The camera model, emissivity setting and frame rate are not recorded in the archive.
  3. No thermocouple reference exists to correct the emissivity of a freshly machined aluminium surface, whose emissivity changes as it is cut.
  4. No file links each IR recording to a trial number.
- Rather than build on uncalibrated temperatures, the author decided to remove thermal sensing from the project permanently.
- The `.irgd` files stay in the raw archive, are excluded from version control (`.gitignore`) and are listed under "Data (available on request)" in `README.md`.

**Standing assumption for all later steps:** **IR_EXPORT = no.** No step may ask for, simulate or substitute thermal data.

**Removed in Step 1** (not yet done; code, results and manuscript are unchanged as of this amendment):

| Component | Where (current code) |
|---|---|
| Thermal-property settings and heat-kernel switch | `src/config.py:119-133` (`GraphConfig`: k, ρ, c, `thermal_edge_cutoff`, `use_thermal_edges`, `thermal_diffusivity_m2_s`) |
| Heat-kernel thermal edge channel | `src/graph_data.py:57-85` (channel 1 of `build_adjacency`) |
| `without_thermal_edges` ablation | `src/graph_data.py:153-155`; ablation list `src/train_evaluate.py:68-69` |
| 2-channel edge bias and thermal attention summary | `src/models.py:118` (`edge_bias = nn.Linear(2, heads)`), `src/models.py:239-251`; `thermal_attention` column in `outputs/attention*.csv` |
| Thermal figures and report text | `src/make_report.py`, `src/finalize_report.py`, `src/make_journal_figures.py` (edge-weight figure), `src/draw_architecture.py` |
| Manuscript | thermal feature vector ϕT (para 78), thermal edges and Eq. 10, RQ3, the IR methods paragraph (para 45), the limitation (para 399), "thermal" in the title and abstract: 36 mentions in `FINAL.docx` |

**What stays:**
- Kinematic edges (consecutive zones along the tool path).
- A **no-edge ablation** (self-loops only), which is the test of whether message passing helps at all.
- After item 12 there are 3 zones per trial, so the graph is a 3-node path.

**Effect on the plan:**
- Item 8 is resolved by removing the constant heat-kernel channel; no measured thermal adjacency will replace it.
- The "measured-thermal vs heat-kernel edges" ablation and RQ3 are dropped.
- Framings are rewritten without thermal:
  - **B′:** zone-level roughness on WAAM vs wrought explained by force and vibration. Needs measured zone Ra and the confirmed trial mapping; target *Materials*.
  - **C:** leakage-free chatter-detection benchmark with external validation of the labelling rule; target *JMMP* or *Sensors*.
  - Framing A, if it survives, uses kinematic edges only.

`CLAUDE_CODE_FIX_PROMPT.md` is not in this folder, so this amendment is recorded here. Paste the line *"IR_EXPORT = no (permanent; see AUDIT.md section H)"* into your copy of the prompt.

> Note: the Step 0 scripts in `audit/` (`audit_analysis*.py`, `label_sensitivity.py`) ran against the **pre-audit** code and outputs. These are now in `_archive/src_pre_audit/` and `_archive/outputs_pre_audit/`. To re-run them, point them at those folders.

---

## I. Step 1 — fixes applied (2026-11-05)

Fixed scope (from the author):
- ZONE_RA = no, IR_EXPORT = no, NEW_TRIALS = 0, ACCEL_INFO = unknown.
- The turning dataset is not downloaded (`external/turning_chatter_mendeley/` is absent), so E1/E2 were skipped. KIT = no.
- Framing C.

| Audit item | Fix | Where |
|---|---|---|
| 1 Loss | Ra head and MSE removed. The only loss is class-weighted BCE; weight decay is Adam's, stated separately. | `src/run_experiments.py: bce_loss`, `src/models.py` (one logit per zone) |
| 2 Material flag | Material is not used anywhere, since the mapping is unconfirmed. The plumbing test is replaced by a generic appended-feature test. | `tests/test_pipeline.py` |
| 3 AUC validity | Permutation null, trial-bootstrap CIs and per-fold AUC (where defined) are reported. No inversion or misalignment: tested. | `src/evaluate.py` |
| 4 Aggregation | One rule for all models: `window_mean` (BiLSTM now averages its outputs); the classical models use window-mean features. Tested. | `src/models.py`, `src/run_experiments.py` |
| 5 Harmonics | Spindle-rotation harmonics everywhere. The chatter indicator now uses a 0.5 s context (2 Hz bins). **New:** the spindle frequency is *measured* from each force spectrum (harmonic comb), because the nominal rpm is far too inaccurate for a ±2 Hz test at harmonic number 104–215. | `src/signal_features.py`, `src/prepare_dataset.py: measure_spindle_hz` |
| 6 DeLong | Removed. Trial-level paired bootstrap of ΔAUC and paired trial-swap permutation test instead. | `src/evaluate.py: paired_test` |
| 7 Early stopping | Removed; a fixed 30 epochs, no validation split. | `src/config.py`, `src/run_experiments.py` |
| 8 Constant adjacency | Thermal heat-kernel channel removed (section H). Only kinematic edges, plus a no-edge ablation. | `src/graph_data.py` |
| 9 Mean baseline | No Ra regression, so no longer applicable. Classification is compared with the permutation null and with logistic regression. | — |
| 10 Capacity | Parameter counts for every model (`results/parameter_counts.csv`). Small GNN ≤ 5k parameters added. | `src/models.py: SmallGNN` |
| 11 Labels | Labels at 15/20/25 dB. Sensitivity grid over tolerance 0.5/1/2 Hz × threshold, with measured vs nominal spindle frequency. | `results/label_counts.csv`, `results/label_sensitivity.csv` |
| 12 Zones | 3 zones per cut, with 5 mm (3.75 s) removed at each end. 16 windows of 100 ms per zone. | `src/prepare_dataset.py: zone_spans` |
| 13 Alignment | `accel_ok = r ≥ 0.7`. A sensitivity run without trials 04 and 06. | `results/metrics.csv` (experiment `alignment_15db`) |
| 14 Confound / material | Material is unresolved, so no material-dependent analysis. Evidence: `audit/trial_mapping_evidence.md`. | `src/mapping_evidence.py` |
| B Trial mapping | Measured speed contradicts the manifest for trials 06, 11 and 16. The speed blocks and force trends point to a specimen change between trials 11 and 12. **Speed is now taken from the measurement; depth of cut is not used.** | `results/trial_mapping.csv` |

Removed code (thermal, Ra, pre-audit reporting) has been moved to `_archive/src_pre_audit/`. The old results are in `_archive/outputs_pre_audit/`.

### Unit tests: `python -m pytest tests -v` → **23 passed** (full log: `audit/unit_tests_output.txt`)

```
test_harmonic_mask                                              PASSED
test_label_rule_peak_on_and_off_harmonic[on harmonic -> stable] PASSED
test_label_rule_peak_on_and_off_harmonic[3 Hz off -> unstable]  PASSED
test_label_rule_threshold                                       PASSED
test_chatter_indicator_uses_spindle_harmonics                   PASSED
test_measured_spindle_frequency                                 PASSED
test_zone_segmentation                                          PASSED
test_kinematic_adjacency_only                                   PASSED
test_loto_split_has_no_overlap                                  PASSED
test_external_samples_cannot_enter_waam_splits                  PASSED
test_window_mean_aggregation_is_order_invariant[fusion_mlp]     PASSED
test_window_mean_aggregation_is_order_invariant[cnn1d]          PASSED
test_window_mean_aggregation_is_order_invariant[small_gnn]      PASSED
test_every_model_uses_window_mean[cnn1d|bilstm|fusion_mlp|gcn|gat|small_gnn]  6 x PASSED
test_classical_aggregation_is_window_mean                       PASSED
test_appended_feature_survives_standardisation_and_reaches_network  PASSED
test_loss_is_weighted_bce_only                                  PASSED
test_small_gnn_capacity                                         PASSED
============================= 23 passed =============================
```

One test failed on its first run. The fault was in the test: its "mid-harmonic" frequency was 1.5 Hz from a harmonic. After correcting the test, the code was unchanged and all 23 tests passed.

There is no real external-data loader, because E1/E2 were not run. The guard test checks that `loto_folds` rejects any sample whose `source` is not `WAAM_LAB`.
