# WAAM GNN pipeline -- data readiness report

Status as of this pass: **NOT READY TO TRAIN.** `validate_data_readiness.py` fails
with 2 blocking errors (no cutting-window timestamps, no zone-level labels).
Everything that could be done without inventing data has been done; the
remaining gaps require source records this repository does not contain.

Run `run_training.bat` for the authoritative, up-to-date status -- it runs
the same validator before doing anything else and will refuse to train while
either blocker remains.

## 1. What this pass verified (file inventory)

`WAAM/` contains 329 files across 3 extracted Google-Drive parts
(`RESULTS-20260817T081727Z-1-00{1,2,3}`), fully listed in
`source_file_inventory.csv` (previously only 226 of 329 files were listed;
this pass rebuilt it from a full filesystem walk). Breakdown:

| Modality | Count | Verified facts |
|---|---|---|
| Vibration (cutting trials) | 21 CSVs (`trial 1.csv`..`trial 21.csv`) | 3-axis accelerometer, units g. Sample rate measured directly from each file's own Time column: trial_01=32768 Hz, trial_03=20000 Hz, all others=25000 Hz exactly. Durations vary 14.3-60.0 s (not fixed-length). |
| Vibration session sidecars | 21 `.rdf` + 21 `.dac` (one pair per trial) | Each `.rdf` embeds real DAQ $ARM/$START date+time fields (see section 2). |
| Force (cutting trials) | 24 Kistler DynoWare recordings (14 in batch `25.03.25.01`, 10 in `25.03.25.02`), each with `_1.csv` numeric export + `.cfg` + `.dwd` | Verified: 25 kHz, 40.00004 s, 1,000,001 samples/channel, columns `Fx,Fy,Fz,Mx,My,Mz` (+ raw bridge channels) in N / Nm. `_hdr.csv` gives a real wall-clock start Date/Time per recording. |
| Thermal video | 28 `.irgd` + 28 matching `.mp4` pairs | `.irgd` header parsed directly: magic `GDIR384*288-PSZ` -> 384x288 radiometric frames; embedded date/time matches filename (2025-03-25, 03:10-04:08). `.mp4` confirmed visible-light (OpenCV: 1024x768, ~25.9 fps). |
| Thermal point traces | 27 `data#_avg.csv` ("P1" column) | Matches the "P1" point marker seen in the paper's Figure 10/11 IR screenshots. Not spatially resolved (single ROI point, not a frame). |
| Natural-frequency (modal) test | 2 trials, separate namespace, own `.rdf`/`.dac`/`.csv` | 4th channel is Pa (microphone), not Acc-Z -- an impact-hammer test, not a cutting trial. Dated 2025-03-07 (`$START_DATE`), the earliest of three distinct acquisition dates found in this archive. Co-located ANSYS Workbench "Natural frequency" FE project (`.agdb`/`.mechdb`/`.rst`/...). |
| Tensile | `ADDITIVE.csv`/`.pdf`, `WROUGHT.csv`/`.pdf` | One representative specimen's stress-strain curve per material; matches paper Section 3.1 aggregate numbers. |

Full per-decision evidence is in `trial_mapping_evidence.csv`; full raw
listing is in `source_file_inventory.csv`.

## 2. The key new finding: three different acquisition dates

Every one of the 21 vibration trials has its own `.rdf` session file (not
just trial_01/03, as first suspected), and each embeds real `$ARM_DATE` /
`$START_DATE` / `$START_TIME` fields read directly from the binary:

- **trial_01, trial_02, trial_03** -- `$START_DATE = 2025-03-14`, times
  15:14:27 / 15:31:05 / 15:32:35.
- **trial_04 .. trial_21** -- `$START_DATE = 2025-03-25`, `$START_TIME`
  increasing monotonically from 15:26:52 (trial_04) to 16:07:15 (trial_21),
  i.e. in the same order as the trial numbers.
- **Natural-frequency hammer test** -- `$START_DATE = 2025-03-07`.

All 24 force recordings are also dated 2025-03-25 (14:44:00-16:07:02).
Consequences:

- **trial_01 and trial_03 have no force recording at all** (no force data
  exists for 14 March in this archive). Both are independently confirmed
  WAAM material via a byte-identical MD5 duplicate stored under
  `VIBRATION DATA/AM ER5356/ACCELEROMETER/TRIAL {1,3}/`.
- **trial_04 .. trial_21 pair against the 25 March force recordings by
  timestamp**, not by file-list order. Pairing each trial's verified
  `$START_TIME` against the nearest force-recording start time, in
  chronological rank order (so no file is claimed twice), gives deltas of
  0-17 s for 15 of 18 trials and two weaker outliers (trial_16: 102 s,
  trial_18: 39 s). This is now filled into `force_csv` /
  `force_csv_confidence` in `raw_trial_manifest.csv`. 6 of the 24 force
  recordings (25.03_000/001/002/003/004/006) do not pair with any trial and
  are left unassigned (possibly calibration/setup cuts).
- **A 646-second gap sits between trial_11 and trial_12's start times** --
  more than double the next-largest inter-trial gap (302 s) and 4-9x the
  typical gap (~90 s). This is consistent with a specimen change on the
  milling machine and casts doubt on the pre-existing provisional
  `doe_block 1 = trials 1-10 / block 2 = 11-20` split: the real
  WAAM/wrought boundary may sit between trial_11 and trial_12 instead. This
  is flagged in `trial_mapping_evidence.csv` but NOT acted on -- both
  possible splits are guesses without a lab log, and overwriting one guess
  with another would not improve correctness.

## 3. Material identity: 2 of 21 trials confirmed, 19 unresolved

Only `trial_01` and `trial_03` have direct evidence of material
(WAAM/AM ER5356, via byte-identical duplication under a folder explicitly
named for that filler alloy). No file, folder name, or log anywhere in
`WAAM/` ties any of trials 2, 4-21 to WAAM or wrought. The paper's own
Table 3 (10 conditions x 2 materials = 20 trials) is never cross-referenced
to a trial number anywhere in the manuscript or the archive -- the DOE
order in `raw_trial_manifest.csv` (`doe_block`, `doe_condition_order`) is
carried over from the pre-existing provisional numbering and remains
**unverified**, not newly confirmed by this pass.

## 4. Thermal data: not usable for the zone-level pipeline, and why

- `.irgd` files are a proprietary radiometric container (header magic
  `GDIR384*288-PSZ`). Decoding to calibrated per-pixel temperature requires
  the vendor's export/analysis software (consistent with the
  `thermal analysis using sonel3` folder name) with documented emissivity,
  reflected-temperature, and distance settings -- **none of which are
  present in this repository**. The header's apparent temperature-range
  field (-40.0/150.0, assuming a x10 fixed-point scale) is a camera
  configuration bound, not a per-pixel calibration.
- `.mp4` files are confirmed visible-light video and must never be treated
  as temperature data (enforced in code: `utils_vision.load_radiometric_array`
  rejects any non-`.npy`/`.npz` input).
- The 27 `P1[Avg(?)]` CSVs are very likely single-point temperature traces
  extracted from individual `.irgd` clips by a third-party "Sonel" tool
  (matches the "P1" marker shown in the paper's own Figure 10), but: (a)
  their exported "Time" column is non-monotonic and dated a week after the
  acquisition (almost certainly the post-processing date, not a capture
  time) and cannot be used for alignment; (b) there is no filename
  correspondence between the 27 CSVs and the 28 `.irgd` clips; (c) even if
  that link were recovered, a single point reading cannot populate the
  pipeline's 6 independent per-zone 2D temperature arrays.
- **What would be needed to use thermal data at all:** (1) the original
  Sonel (or compatible) IR-analysis project/license to re-export full
  384x288 calibrated frames (not single points) from each `.irgd` clip with
  documented emissivity/reflected-temperature settings; (2) a log tying
  specific `.irgd` clips to specific trial numbers (the camera's own clock
  is ~11-12 h offset from the DAQ PC's clock on the same day, so filename
  timestamps alone cannot establish this); (3) a measured or published
  thermal diffusivity for the WAAM Al 5356 / wrought Al 5052 specimens if
  thermal-diffusion graph edges are wanted.

**The pipeline no longer requires any of this.** `config.py`'s
`VisionConfig.include_thermal` now defaults to `False`, and
`GraphConfig.include_thermal_edges` defaults to `False`. `dataset_graph.py`,
`models.py`, and `train_evaluate.py` were refactored so the whole path (NPZ
schema, `ModalityEncoder`/`ThermalEncoder` wiring, CLI flags, and the
`without_thermal_features`/`without_thermal_edges` ablations) works
force+acceleration-only and only activates thermal when explicitly asked for
with `--include-thermal` (and the matching `--thermal-fps` /
`--thermal-radiometric` / `--ambient-c`). This was verified with a synthetic
NPZ round-trip through `features_from_npz` -> `make_zone_graph` ->
`SpatioTemporalGraphNet`/`MultimodalFusionMLP`, both with and without
thermal, plus a check that requesting `without_thermal_features` on a
thermal-less graph raises a clear error instead of crashing on a missing
tensor.

## 5. Labels: zero valid chatter or Ra values exist anywhere

`zone_labels.csv` has 126 rows (21 trials x 6 zones) and both
`chatter_label` and `measured_Ra_um` are blank in all of them. This is
correct, not an oversight:

- The paper reports Ra as **two numbers total** -- one material-level
  average per material, each itself an average over 3 measurement
  *positions* (LEFT/CENTRE/RIGHT) on one representative specimen (WAAM
  1.22 um, wrought 0.83 um, "46.73% increase"). There is no per-trial or
  per-6-zone breakdown in the paper, and no raw roughness-tester export
  file exists anywhere under `WAAM/`.
- The paper's chatter discussion is qualitative: a PSD/spectrogram
  comparison at one representative condition (1000 rpm / 1.5 mm / 80 mm/min,
  chosen as "a severe cutting condition"), stating "the same pattern was
  seen for other combinations," plus one unlocated mention of "a separate
  chatter peak at 970 Hz" in the conclusion. No table of per-trial or
  per-zone stable/unstable labels exists anywhere.
- Per the task instructions, aggregate material-level numbers must not be
  copied into per-trial/per-zone label cells, and no fabricated label was
  written.

A chatter indicator *could* in principle be computed from the raw
force/vibration signals via an objective, documented criterion (`config.py`'s
`SignalConfig.chatter_analysis_band_hz` / `chatter_indicator()` in
`utils_signal.py` already implements a non-synchronous-energy-fraction
measure) -- but that would be a derived feature or a researcher-defined
threshold rule, not an observed ground-truth label, and must not be written
into `zone_labels.csv` as if it were a measurement.

## 6. Code changes made this pass

- `config.py`: added `VisionConfig.include_thermal` (default `False`);
  changed `GraphConfig.include_thermal_edges` default from `True` to
  `False` (an unverified thermal diffusivity must no longer be implicitly
  required).
- `dataset_graph.py`: `features_from_npz`, `make_zone_graph`, and
  `prepare_raw_trial_cache` now treat `temperature_windows_c`/`thermal_npz`
  as optional, gated by `config.vision.include_thermal`; the CLI gained
  `--include-thermal` and no longer requires `--thermal-fps` unconditionally.
- `models.py`: `SpatioTemporalGraphNet` and `MultimodalFusionMLP` take a
  `use_thermal` flag that skips building/calling `ThermalEncoder` and
  adjusts the fusion input dimension (32 vs 48) accordingly.
- `train_evaluate.py`: `make_model` derives `use_thermal` from the sample
  graph's `has_thermal` flag; `apply_variant` raises a clear, specific error
  if `without_thermal_features`/`without_thermal_edges` is requested on data
  that has no thermal modality/edges, instead of crashing on a missing
  attribute; CLI thermal args are now optional and gated behind
  `--include-thermal`.
- `validate_data_readiness.py` (new): checks manifest/label completeness
  and file resolvability and exits non-zero with a specific, itemised report
  if training inputs are not real and complete. Does not fabricate or patch
  around anything.
- `raw_trial_manifest.csv`: rewritten with verified `material_type`
  (2/21), measured `acceleration_fs`/`duration`, verified
  `acquisition_date`/`start_time`, and timestamp-matched `force_csv` (18/21,
  with a confidence string per row). `cut_start_time_s` remains blank for
  all 21 trials -- no source establishes it.
- `trial_mapping_evidence.csv` (new): full evidence/confidence/notes trail
  for every assignment and non-assignment decision described above.
- `source_file_inventory.csv`: regenerated from a full filesystem walk
  (329 files; the previous version only listed 226).

## 7. Exactly what is still needed from you

Each of these is a real missing source record, not a code fix:

1. **Cutting start time within each force/accelerometer recording**
   (`cut_start_time_s`). Needed for every trial before any feature window
   can be extracted. A rough pattern was observed in one spectrum plot
   (`PLOTS/FORCE/WAAM Spectrum/WAAM1.png`: cutting burst runs roughly from
   ~7 s to ~29 s inside a 40 s force recording) but this was not assumed to
   generalize to other trials/files, since it is a single uncorroborated
   example.
2. **The DOE condition order and material block assignment**, ideally from
   a lab notebook/log that states which physical trial number corresponds
   to which row of Table 3 and which material. The timing evidence in
   section 2 is suggestive (chronological order for 4-21 looks consistent;
   the 11/12 gap looks like a material swap) but is not a substitute for
   the actual log.
3. **Zone-level (6 per trial) chatter annotations** -- e.g. from reviewing
   the force/vibration time series or machined-surface photos against a
   documented stability criterion, or an analyst's original per-zone
   stable/unstable call.
4. **Zone-level (6 per trial) measured Ra** -- from the original
   roughness-tester exports/printouts, if they still exist; the paper only
   publishes 2 material-level averages.
5. **If thermal input is wanted at all**: the Sonel (or compatible)
   software/license to re-export full calibrated 384x288 frames per
   `.irgd` clip, the emissivity/reflected-temperature settings used at
   capture time, and a log tying `.irgd` clip filenames to trial numbers
   (the camera clock does not match the DAQ PC clock).
6. **The analysis window length** (`signal.window_ms`, currently `None` in
   `config.py`) and a verified thermal diffusivity for Al 5356/5052 if
   thermal-diffusion graph edges are ever enabled.

## 8. The one command to run

```
run_training.bat <WINDOW_MS> [ACCELERATION_FS_HZ]
```

This checks readiness, prepares synchronized per-zone feature caches from
`raw_trial_manifest.csv`, and runs leave-one-trial-out training/evaluation
-- but **only if `validate_data_readiness.py` passes**. Right now it will
stop at step 1 with the two blocking errors above
(`cut_start_time_s` and zone labels), by design: this is not yet ready to
train, and the script says so instead of claiming otherwise.
