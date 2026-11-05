# External datasets — what they are, checked against their papers

**Rules.** External data never enters WAAM training or test splits. It is never relabelled as, merged with, or described as our measurements. It is never used as a target for our zones, and each external analysis gets its own section and citation.

**Status 2026-11-05.** None of the datasets below are downloaded. Every fact here was checked against the PDF named in the row. Facts that can only be confirmed from the downloaded files are marked *unverified*.

---

## 1. Turning chatter dataset (Khasawneh, Otto, Yesilli)

| Field | Value | Source |
|---|---|---|
| Citation | F. Khasawneh, A. Otto, M. Yesilli, "Turning dataset for chatter diagnosis using machine learning", Mendeley Data v1, 2019, doi:10.17632/hvm4wh3jzx.1 | ref. [34] of arXiv:1908.01678 |
| Process | **Turning** (boring bar on an engine lathe); **not milling** | 1908.01678 §2.1; 2204.05400 §2 |
| Tool / configurations | Carbide-insert boring bar (S10R-SCLCR3S), 4 stickout (overhang) lengths: 5.08 / 6.35 / 8.89 / 11.43 cm | 1908.01678 §2.1 |
| Speeds | 5.08 cm: 320, 425, 570, 770 rpm; 6.35 cm: 570, 770; 8.89 cm: 570, 770, 1030; 11.43 cm: 570, 770, 1030 | 2204.05400 Table 1 |
| Depth / feed | 0.0254–1.27 mm, depending on stickout; feed 0.00508 cm/rev | 2204.05400 Table 1 |
| Sensors | 2 × PCB 352B10 uniaxial + 1 × PCB 356B11 triaxial accelerometer; also a microphone and laser tachometer. **Accelerometer only — no force** | 1908.01678 §2.1 |
| Sampling | 160 kHz (NI USB-6366), Butterworth low-pass (order 100), downsampled to **10 kHz**. Raw and filtered data both in the repository | 1908.01678 §2.2 |
| Channel used for labels | x-axis of the triaxial accelerometer (best SNR) | 1908.01678 §2.3 |
| Labels | stable (s), intermediate/mild chatter (i), chatter (c), unknown (u), assigned by **manual time- and frequency-domain inspection** (< 5 kHz). Checked against surface-finish photos. One cutting signal can contain regions with different labels | 1908.01678 §2.3 |
| Counts (s / i / c) | 5.08 cm: 17/8/11 (36); 6.35: 7/4/3 (14); 8.89: 7/2/2 (11); 11.43: 13/4/5 (22). Unknown not counted | 1908.01678 Table 1 |
| Published accuracies | 2-class (chatter incl. intermediate vs. no chatter), **random train/test splits within each stickout**, not leave-one-stickout-out. Best DTW: 99.5 / 75.8 / 94.6 / 81.5 %. WPT: 93.9 / 100.0 / 84.0 / 78.8 % | 1908.01678 Table 2 |
| Licence | *Unverified* (Mendeley Data page not opened) | — |
| File layout | *Unverified* | — |

Files needed: the whole Mendeley record, filtered 10 kHz signals plus the label/tag files, into `external/turning_chatter_mendeley/`.

How it may be used here:
- **E1** — label-rule agreement.
- **E2** — feature generalisation, leave-one-stickout-out. This is **not** comparable like-for-like with the within-stickout accuracies above, and the paper must say so.
- **E3** — encoder pretraining.

Its labels are manual, so E1 measures agreement with expert labelling, not with ground truth. The "intermediate" class (low time-domain, high frequency-domain amplitude) is defined in a way that a spectral-prominence rule will tend to call unstable. Both mappings will be reported.

---

## 2. KIT multimodal CNC milling dataset (Ströbel et al. 2025)

| Field | Value | Source |
|---|---|---|
| Paper | R. Ströbel, M. Kuck, F. Oexle, H. Kader, A. Puchta, B. Noack, J. Fleischer, "A multimodal dataset for process monitoring and anomaly detection in industrial CNC milling", *Data in Brief* 63 (2025) 112207, doi:10.1016/j.dib.2025.112207 | `main.pdf` p. 1 |
| Data DOI | 10.35097/hvvwn1kfwf7qt48z (KIT repository; direct URL https://publikationen.bibliothek.kit.edu/1000182633) | `main.pdf` specifications table |
| Licence | Article: CC BY 4.0 (stated). Dataset licence: *unverified* until the repository page is opened | `main.pdf` p. 1 |
| Sensors | Kistler **9255C** force platform via Kistler 5015A charge amplifier, and **PCB 356A33** accelerometer on the main spindle, both at **10 kHz**. SINUMERIK 840D controller signals at **500 Hz** via SINUMERIK Edge (JSON), PCHIP-interpolated to 10 kHz for alignment | `main.pdf` §2–3 |
| Storage | Force/acceleration as MATLAB timetable files; controller data as JSON | `main.pdf` §3 |
| Trials we need | **IM-02F-A01** (chatter, induced by **doubling the feed rate** during finishing; chatter marks shown in Fig. 16; 303.95 s). References: **IM-01F** (normal finishing, 592.98 s) and **IM-01F-A01** (irregular stock allowance, 592.98 s). All on S235JR steel with a 10 mm solid-carbide end mill (SCEM) | `main.pdf` Table 2, §4.2.2, §5.3, §5.5 |
| Chatter count | **One** chatter trial in the whole dataset | `main.pdf` Table 2 |
| Chatter interval | Figs. 15–16 show the evaluated interval and the X-axis current with chatter vs. reference. Exact start/end times must be read from the downloaded data and figure, *unverified* | `main.pdf` §5.3 |
| Related | A complementary second-machine dataset (DMG Mori CMX 600V): DOI 10.35097/vnnu3n9z7ndsnhfd. Not planned for use | `main.pdf` footnote 1 |

Files needed: only the IM-01F, IM-01F-A01 and IM-02F-A01 folders, into `external/kit_cnc_milling/`.

Use here: **E4**, a qualitative force-domain sanity check only. It has one chatter trial, so no accuracy figures will be reported. Note the differences from our setup:
- 10 kHz sampling (Nyquist 5 kHz), so our 2.5–5.5 kHz band must be truncated to 2.5–5.0 kHz and this must be stated;
- steel workpiece;
- different machine and dynamometer.

The "irregular stock allowance" reference is itself an anomaly trial.

---

## 3. Al 7050 milling data in arXiv:2204.05400 (Yesilli, Khasawneh, Mann)

| Field | Value | Source |
|---|---|---|
| Paper | M.C. Yesilli, F.A. Khasawneh, B.P. Mann, "Transfer Learning for Autonomous Chatter Detection in Machining", arXiv:2204.05400v1 (2022) | `2204.05400v1.pdf` p. 1 |
| Process | Down milling on Al 7050-T7451. Ingersoll machining centre, Fischer 40 000 rpm spindle. 19.05 mm two-flute end mill, 106 mm overhang, 5% radial immersion, 11 206–32 161 rpm, 0.381–3.556 mm depth | §2, Table 1 |
| Sensors | Lion Precision **capacitive** tool-displacement probes (x, y), **25 kHz**, low-pass filtered and downsampled to 12.5 kHz. Laser tachometer | §2 |
| Labels | PSD plots and Poincaré sections | §2 |
| Public availability | The paper does **not** state that the milling data are public (no repository or DOI found in the text) | full-text search |

Use here: **related work only.** It will not be used unless an official public release is found and you approve it.

Journal versions of arXiv:1908.01678 and arXiv:2204.05400 have **not yet been checked**; this is Step 5. Until verified, they are cited as arXiv preprints.

---

## 4. Quarantined, not used

`robotConfigurations.mat`, `dataSetRobotConfig1_int.mat`, `dataSetRobotConfig2_int.mat` (MATLAB 5.0, created 2017). They contain tabular robotic-milling stability-type data with no sensor signals, and there is no source paper or licence. **Not used, not cited.** See AUDIT.md item 17.
