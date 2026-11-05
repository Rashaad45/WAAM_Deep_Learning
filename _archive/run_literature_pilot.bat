@echo off
setlocal
cd /d "%~dp0"

rem EXPLORATORY ONLY. This pilot repeats published material-level Ra means
rem across zones and uses force-derived cut-start candidates. Its CV metrics
rem are not experimental predictive performance and must not be reported as such.

set "WINDOW_MS=%~1"
set "ACCEL_FS=%~2"
if "%WINDOW_MS%"=="" set "WINDOW_MS=50"
if "%ACCEL_FS%"=="" set "ACCEL_FS=25000"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Activate your Python environment first.
    exit /b 1
)

echo Literature-imputed Ra pilot: specimen means are repeated across six zones.
echo Chatter classification is disabled because no chatter labels exist.
echo The 50 ms window is a pilot setting, not a reported experimental setting.

python validate_data_readiness.py --manifest raw_trial_manifest_literature_pilot.csv --labels zone_labels_literature_pilot.csv
if errorlevel 1 exit /b 1

python dataset_graph.py --prepare-raw-manifest raw_trial_manifest_literature_pilot.csv --cache-output processed_features_literature_pilot --window-ms %WINDOW_MS% --acceleration-fs %ACCEL_FS%
if errorlevel 1 exit /b 1

python train_evaluate.py --root processed_features_literature_pilot --manifest processed_features_literature_pilot\trial_manifest.csv --labels zone_labels_literature_pilot.csv --output outputs_literature_pilot --models gcn --ablations roughness_only --window-ms %WINDOW_MS% --acceleration-fs %ACCEL_FS%
if errorlevel 1 exit /b 1

echo Pilot run completed. Outputs are in outputs_literature_pilot\ and are not valid measured-Ra model metrics.
