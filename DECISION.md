# DECISION.md — framing, title, key results, decisions taken, items to check

## Framing and target
- **Framing C**, fixed by the author: a leakage-free small-data chatter-detection benchmark.
  No thermal data, no roughness head, no claims that depend on material identity.
- **Target journal:** MDPI *Journal of Manufacturing and Materials Processing* (JMMP).
- **Title:** *A Leakage-Free Small-Data Benchmark for Force- and Vibration-Based Chatter Detection in
  Thin-Wall Milling of Wire-Arc Additively Manufactured and Wrought Al–Mg Walls*

## Key results (all from `results/*.csv`)
1. **The archived trial mapping was partly wrong.** The spindle speed measured from the force spectra
   (two classes: ~1031 and ~1462 rpm) contradicts the manifest for trials 06, 11 and 16.
   - The speed blocks and a monotonic force rise inside every block (Spearman ρ = 1.00) point to a
     specimen change between trials 11 and 12, one trial later than assumed.
   - **No trial with a force record has a confirmed material**, so the material-specific supplement
     is omitted.
2. **The labels are threshold-driven.** Unstable zones number 25 / 1 / 0 of 48 at 15 / 20 / 25 dB.
   - At 15 dB, 24 of 48 zones lie within ±2 dB of the threshold, and the "unstable" class is 52% of
     zones.
   - Using nominal instead of measured spindle speed changes 11 labels.
   - Only 15 dB can be evaluated.
3. **At 15 dB, every model using the spectral features scores pooled AUC 0.65–0.74.**
   - Random forest is highest at 0.74 (0.55–0.89); logistic regression scores 0.69 (0.50–0.86).
   - The permutation-null band is 0.34–0.67. 7 of 9 models score above it.
   - **No model differs from logistic regression in trial-level paired tests** (all p ≥ 0.06).
   - The raw-signal 1D-CNN is at chance: 0.44 (0.21–0.66).
4. **Graphs add nothing.** No-edge ablations change AUC by +0.01, +0.02 and −0.01 (ST-GAT, ST-GCN,
   small GNN; p = 0.72, 0.64, 0.87). The large networks fit their training trials almost perfectly
   (final loss 0.01–0.09) without generalising better.
5. **The discrimination is mostly between trials.**
   - Mean within-trial AUC is 0.81 for logistic regression and 0.44–0.62 for every other model
     (8 trials with both classes, 3 zones each).
   - Removing the two weakly aligned trials (04, 06; no unstable zones) lowers the AUC of every
     feature-based model.
   - For logistic regression, force-only gives 0.77 and acceleration-only 0.71 (overlapping CIs).

## Decisions taken without stopping (as instructed)

| Decision | Reason |
|---|---|
| **Spindle frequency measured from each force spectrum** (harmonic comb, 12–30 Hz search), used in labels, the chatter indicator and inputs | A ±2 Hz test at harmonic numbers 104–215 needs ~0.01 Hz accuracy; the nominal rpm is ~3–4% off. This is a correctness fix, not tuning |
| **Depth of cut not used as an input** | Its trial mapping is unverified |
| **Main benchmark at 15 dB only**; 20 dB reported for the classical models, BiLSTM and small GNN; 25 dB has no positives | Follows from the label counts, not from model scores |
| **16 windows per zone, 30 fixed epochs** | Compute budget. Fixed after a timing test and before any model result; not tuned. The first attempt with 32 windows and 40 epochs was too slow (~10 h) |
| **Workers run single-threaded**, non-deterministic torch kernels allowed | Throughput; seeds are fixed |
| **E1/E2 skipped** | `external/turning_chatter_mendeley/` is not present. Stated in the Limitations |
| **Official MDPI template not applied** | mdpi.com refused the automated download (HTTP 403). The .docx imitates the MDPI layout (A4, Palatino, section/back-matter structure, three-line tables); paste the text into the official JMMP template before submission |
| **PDF conversion with LibreOffice 26.8.2**, from an administrative MSI extract in the session scratchpad | winget stalled and a per-user install failed (MSI 1603). Nothing was installed system-wide; set `SOFFICE_PATH` to reuse it |
| **Blank-symbol checks added to `check_pdf.py`** | Beyond `", , "` / `"where is"`: per-equation text presence, ink in every non-ASCII glyph box (render-based), and LibreOffice's red "¿" error marker. These caught 3 blank bold/script glyphs and one malformed subscript, which were fixed |
| **`check_numbers.py` ignores row-level CSVs** (predictions, per-epoch losses, per-zone details) | Otherwise almost any number would "match". A planted-number test fails as it should |
| **Pre-audit code and outputs moved to `_archive/`** | They contain the thermal/Ra components removed per AUDIT.md §H |

## Run status
- Unit tests: **23/23 passed** (`audit/unit_tests_output.txt`).
- `check_numbers.py`: **PASS** (447 numbers traced).
- PDF page check: **PASS** (11 pages, 37 equations, no blank glyphs).
- Total runtime: about 2 h 10 min. The LOTO grid itself took 102 min on 6 cores.

## What you must check
1. **Material and condition mapping** (`audit/trial_mapping_evidence.md`). Confirm it from the lab log
   (TODO_AUTHOR #5). If at least 4 trials per material are confirmed, the supplementary material
   section can be added.
2. **Paper 1 setup details** in §2.1: tool Ø10 mm 4-flute carbide, 80 mm/min, 30 mm wall width,
   1000/1400 rpm, 0.50–1.50 mm in 0.25 mm steps, Kistler 9257B. These came from your earlier draft.
   The Crossref record of paper 1 was read, but the MDPI page itself refused access (403), so I could
   not verify them against the published article.
3. **Introduction citations [2] and [3].** I could open only their metadata, not their abstracts.
   Confirm that they support "porosity", "Mg loss/anisotropy" and "layer reheating (in-situ heat
   treatment)". Replace [2] for the in-situ annealing sentence if needed.
4. **Prevalence at 15 dB is 0.52.** Decide whether you consider a rule that flags half of all zones
   a credible "chatter" label. The paper says plainly that it is questionable.
5. **"Conflicts of Interest: none"** is a default; confirm it.
6. **Before submission, run E1/E2:** download the Mendeley turning dataset into
   `external/turning_chatter_mendeley/`. This is the main open reviewer risk (`REVIEWER_RISK.md` #1, #15).
