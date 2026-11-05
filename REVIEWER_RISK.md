# REVIEWER_RISK — likely objections and where the paper answers them

Target: MDPI *Journal of Manufacturing and Materials Processing* (Framing C).
Status column: **Answered** = the paper deals with it adequately; **Partly** = the paper states it, but the
data cannot fully resolve it; **Open** = only new data can fix it.

| # | Likely objection | Where answered | Status |
|---|---|---|---|
| 1 | "Your labels are produced by your own spectral rule, so you are measuring agreement with a rule, not chatter." | §2.4 (rule), §3.2 (threshold sensitivity, Table 2, Fig. 2), §4 para 2, §5 bullet 2 | **Open.** Needs an independent reference: surface chatter marks, zone Ra, or E1 on the turning dataset |
| 2 | "The thresholds (15/20/25 dB, ±2 Hz, 2.5–5.5 kHz) are arbitrary." | §3.2 (all three thresholds, tolerance grid, measured vs nominal speed), §5 | Partly. Sensitivity is shown, but there is no ground truth |
| 3 | "Only 15 dB is evaluated; is that cherry-picking?" | §3.2, §3.5: 20 dB gives 1 unstable zone and 25 dB gives 0, so no classifier can be evaluated there. The choice follows from label counts, not model scores | Answered |
| 4 | "Material identity is unknown — then why WAAM in the title?" | §2.2, §3.1, §5 bullet 1. The specimens of [1] are WAAM and wrought, but the per-trial assignment is unconfirmed, so no claim depends on it | Partly. The lab log would resolve it (TODO_AUTHOR #5) |
| 5 | "16 trials / 48 zones is too small for deep learning." | That is the paper's point: §2.6 Table 1 (parameter counts), §3.3–3.4, §4 para 1 | Answered (as a finding) |
| 6 | "Zones within a trial are correlated; your statistics assume independence." | §2.7: trial-bootstrap CIs and trial-level paired permutation tests, no DeLong. The zone-level permutation null is labelled as narrower than a trial-level null | Answered |
| 7 | "Deep models were undertrained / not tuned." | §2.7: fixed 30 epochs, chosen a priori. §3.5 and Fig. 5: training loss convergence. No hyperparameter search was done for any model, and the paper says so | Partly. A reviewer may ask for nested tuning, which is infeasible with these few positive trials |
| 8 | "Your graph is trivial (3 nodes, one edge type)." | §2.6, §4 para 1. No-edge ablations (Table 4). Heat-kernel edges were removed because they were constant | Answered (negative result) |
| 9 | "Why remove the thermal data from paper 1?" | §2.1, §5 bullet 5: uncalibrated vendor format, no emissivity, frame rate or thermocouple reference, no trial link | Answered |
| 10 | "No surface roughness, so the result isn't relevant to quality." | §5 bullet 4, §6 | **Open.** Zone Ra measurement needed |
| 11 | "A dynamometer isn't practical in production." | §4 para 4, §5 bullet 7. The acceleration-only ablation is reported (Table 4) | Partly. Accelerometer metadata is missing |
| 12 | "Accelerometer model and mounting unknown." | §2.1, §5 bullet 6 | Open (TODO_AUTHOR #4) |
| 13 | "Force/vibration alignment between unsynchronised PCs is unreliable." | §2.1 (method, r range). §3.5: sensitivity run without trials 04 and 06 | Answered |
| 14 | "Spindle speed was 'measured' — how accurate is the comb?" | §2.2 Eq. 1, §3.1 (comb contrast, two tight speed classes). Unit test on a synthetic signal (repository) | Answered |
| 15 | "Where is external validation?" | §5 bullet 8. The turning dataset [6] was not downloaded, so E1/E2 were not run; KIT [7,8] has a single chatter trial | **Open.** Run E1/E2 before submission |
| 16 | "Comparison with published accuracies [4,5]?" | Not compared. Those use random within-configuration splits on other processes, so a comparison would be misleading. External validation (E2) would have to state this | Partly |
| 17 | "Results depend on window count / decimation choices." | §5 last bullet: fixed before evaluation by the compute budget, not tuned. Not varied | Partly |
| 18 | "Reproducibility?" | Data Availability; repository with pinned `requirements.txt`, unit tests (23 passing), `check_numbers.py`, all tables in `results/*.csv` | Answered once the repository DOI is supplied |
| 19 | "AI use?" | Acknowledgments disclosure | Answered |
| 20 | "The intro cites porosity, anisotropy and in-situ annealing — evidence?" | §1 cites [1–3]. [2] and [3] were checked by title and metadata only (abstracts not accessible); the author should confirm that they support these specific statements | Partly (author to verify) |
