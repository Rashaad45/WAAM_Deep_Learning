# SENSORS

Filled only from the code and data files in this repository. "not recorded" = not found in the archive.

| channel | sensor | sampling rate | acquisition computer | units | calibration information | alignment method |
|---|---|---|---|---|---|---|
| Fx | Kistler 9257B dynamometer (src/config.py, README.md; DynoWare .dwd/.csv export) | 25000 Hz (data/reference/force_recording_index.csv) | force acquisition PC (DynoWare); identity not recorded | N (CSV units row) | not recorded (charge-amplifier settings not in the archive) | reference clock |
| Fy | Kistler 9257B dynamometer (src/config.py, README.md; DynoWare .dwd/.csv export) | 25000 Hz (data/reference/force_recording_index.csv) | force acquisition PC (DynoWare); identity not recorded | N (CSV units row) | not recorded (charge-amplifier settings not in the archive) | reference clock |
| Fz | Kistler 9257B dynamometer (src/config.py, README.md; DynoWare .dwd/.csv export) | 25000 Hz (data/reference/force_recording_index.csv) | force acquisition PC (DynoWare); identity not recorded | N (CSV units row) | not recorded (charge-amplifier settings not in the archive) | reference clock |
| Acc-X | triaxial accelerometer; model not recorded | 25 000 Hz (from CSV time column, checked to ±1 Hz in prepare_dataset.py) | separate acquisition computer, not synchronised with the force PC; identity not recorded | g (CSV units row) | not recorded (sensitivity and mounting not recorded) | log-RMS envelope vs force-cut step template, max correlation, lag ≤ 6 s, 10 ms bins (prepare_dataset.align_acceleration) |
| Acc-Y | triaxial accelerometer; model not recorded | 25 000 Hz (from CSV time column, checked to ±1 Hz in prepare_dataset.py) | separate acquisition computer, not synchronised with the force PC; identity not recorded | g (CSV units row) | not recorded (sensitivity and mounting not recorded) | log-RMS envelope vs force-cut step template, max correlation, lag ≤ 6 s, 10 ms bins (prepare_dataset.align_acceleration) |
| Acc-Z | triaxial accelerometer; model not recorded | 25 000 Hz (from CSV time column, checked to ±1 Hz in prepare_dataset.py) | separate acquisition computer, not synchronised with the force PC; identity not recorded | g (CSV units row) | not recorded (sensitivity and mounting not recorded) | log-RMS envelope vs force-cut step template, max correlation, lag ≤ 6 s, 10 ms bins (prepare_dataset.align_acceleration) |

## Alignment per trial (data/processed/dataset_summary.csv)

| trial | included | accel lag (s) | alignment r | r ≥ 0.7 | measured spindle rpm |
|---|---|---|---|---|---|
| trial_04 | True | -2.08 | 0.431 | False | 1030.66 |
| trial_05 | False | not computed | not computed |  |  |
| trial_06 | True | -2.92 | 0.652 | False | 1032.03 |
| trial_07 | True | -1.67 | 0.838 | True | 1461.28 |
| trial_08 | True | -2.57 | 0.706 | True | 1460.44 |
| trial_09 | True | -1.69 | 0.826 | True | 1461.2 |
| trial_10 | True | -1.29 | 0.823 | True | 1461.87 |
| trial_11 | True | -3.65 | 0.846 | True | 1461.43 |
| trial_12 | True | 0.46 | 0.888 | True | 1031.64 |
| trial_13 | True | -0.67 | 0.797 | True | 1030.66 |
| trial_14 | True | -1.29 | 0.847 | True | 1030.84 |
| trial_15 | True | -0.19 | 0.771 | True | 1031.44 |
| trial_16 | True | -0.04 | 0.959 | True | 1031.95 |
| trial_17 | True | -0.77 | 0.968 | True | 1462.66 |
| trial_18 | True | -1.25 | 0.939 | True | 1462.38 |
| trial_19 | True | -0.67 | 0.946 | True | 1462.83 |
| trial_20 | True | -1.91 | 0.972 | True | 1461.34 |
