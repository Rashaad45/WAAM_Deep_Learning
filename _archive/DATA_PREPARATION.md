# WAAM raw data intake status

> **See `DATA_READINESS_REPORT.md` for the current, authoritative status** --
> it supersedes several details below (notably: every trial 1-21 has its own
> acquisition-timestamp sidecar, not just trial 1/3; force recordings are now
> timestamp-matched to 18 of 21 trials; and thermal input is now an optional,
> code-level toggle rather than a hard requirement). This file is kept for
> history.

The manifest and zone-label CSVs beside this note are intake files, not ready-to-train data.

- `raw_trial_manifest.csv` has one row for each of the 21 primary vibration files named `trial 1.csv` through `trial 21.csv`. It now includes the Table 3 DOE sequence for candidate trials 1-20 (the ten rows repeated for the second material block), plus the paper's 80 mm/min feed. `doe_block` is only a block label; it does not identify WAAM versus wrought. Trial 21 is left without a DOE assignment. The paper does not establish that folder trial numbers are the physical run order, so verify that mapping against the acquisition notebook before treating these as confirmed trial settings.
- The manifest records a nominal accelerometer rate estimated from the timestamps: trial 1 is approximately 32.768 kHz, trial 3 is 20 kHz, and the other primary files are 25 kHz. Their time columns are the source of truth; the nominal column is for review and the preprocessing path should resample onto one common rate.
- `zone_labels.csv` has six zone rows per candidate trial. Fill chatter labels from the experimental observations and fill Ra only where that zone has an actual measurement. Do not copy one trial-level Ra value into all six zones unless the report's measurement protocol explicitly defines that operation.
- `source_file_inventory.csv` lists candidate raw files and exports found under `WAAM/`. It is an inventory, not an association table.
- `force_recording_index.csv` links all 24 numeric force CSV exports to their companion acquisition headers. The headers document 25 kHz, 40 seconds, and 1,000,001 samples. The 24 recordings have not been assigned to the 21 candidate vibration trials.
- `thermal_p1_summary.csv` summarizes all 27 P1 scalar exports (row count, observed min/max/mean, and exported timestamps). The column header's unit is not established and the exported clock does not match the March 25 acquisition date, so these summaries must not be used as calibrated temperature labels yet.
- The paper reports aggregate material-level outcomes and a percentage roughness difference, not per-trial or per-zone Ra values or chatter labels. Those aggregate claims cannot safely fill `zone_labels.csv`.

## What is already supported by the source files

- The force recordings include numeric DynoWare CSV exports in addition to the DWD files. The raw preprocessing reader recognizes the DynoWare metadata/header/units layout and selects columns named Fx, Fy, and Fz.
- Primary vibration CSVs use a time column and three acceleration axes. Confirm their units and sample rate against the acquisition record before extracting features.
- The `.mp4` files are visible-light video. Matching `.irgd` files are proprietary infrared recordings. Neither format by itself provides calibrated temperature arrays to the Python preprocessing path.
- The `PLOTS/THERMAL/thermal analysis using sonel3` CSVs contain a P1 ROI scalar series, but exported timestamps use an unrelated April 2025 clock. They do not establish alignment with the March 25 sensor recordings and are not full radiometric frames.

## Still required before preprocessing/training

1. Verify that primary vibration file numbers 1-20 correspond to the paper's ordered DOE and identify which ten belong to WROUGHT versus WAAM from the acquisition notes. If runs were randomized, replace the provisional DOE ordering.
2. Confirm the force, video, and infrared file association and cutting start time for each trial using acquisition records. Do not infer identity only from nearby timestamps.
3. Supply `material_type` and cut start time for each trial from the experiment log. The paper provides provisional speed, depth, and feed values in the manifest, but confirm the run order against the acquisition log.
4. Confirm accelerometer rate/units and the analysis window length from acquisition records or the protocol. The CSV time steps indicate the nominal per-file rates listed in the manifest.
5. Export each `.irgd` recording to calibrated temperature arrays, preserving frame timestamps. Confirm frame rate, emissivity/reflected-temperature calibration, ROI P1 geometry, and the clock offset to the force/vibration streams.
6. Supply six zone-level chatter labels per trial and the actual measured zone-level Ra values.
7. Confirm the thermal diffusivity value/source if thermal-diffusion graph edges are enabled.

Once these are resolved, set the corresponding values in `config.py`, complete both CSVs, and use `dataset_graph.py`'s raw-cache preparation CLI before running `train_evaluate.py`.
