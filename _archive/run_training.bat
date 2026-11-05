@echo off
setlocal enabledelayedexpansion

rem One-command entry point: validate data readiness, prepare synchronized
rem features from raw_trial_manifest.csv, then run leave-one-trial-out
rem training/evaluation -- but ONLY if the data validator confirms real,
rem non-fabricated inputs and labels exist. If it does not, this script
rem stops at the validator step and prints exactly what is missing; it will
rem not silently train on placeholder or invented data.
rem
rem Usage:
rem   run_training.bat [WINDOW_MS] [ACCELERATION_FS_HZ]
rem
rem   WINDOW_MS           Analysis window length in milliseconds. No default:
rem                       this value is not yet confirmed from the acquisition
rem                       protocol (see DATA_READINESS_REPORT.md). You must
rem                       pass it explicitly once you have it.
rem   ACCELERATION_FS_HZ  Nominal accelerometer sample rate used only for
rem                       spectral feature extraction (raw resampling always
rem                       uses each file's own measured timestamps). Defaults
rem                       to 25000, the rate measured directly for 18 of the
rem                       21 vibration files; trial_01 (32768 Hz) and trial_03
rem                       (20000 Hz) were recorded on a different system -- see
rem                       raw_trial_manifest.csv.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [run_training] ERROR: "python" was not found on PATH.
    echo Run install_dependencies.bat first, or activate your Python environment.
    exit /b 1
)

echo [run_training] Step 1/3: checking data readiness...
python validate_data_readiness.py
if errorlevel 1 (
    echo.
    echo [run_training] STOPPED: the data readiness check above found blocking
    echo issues. Training will NOT start on incomplete or fabricated data.
    echo Resolve the listed items ^(see DATA_READINESS_REPORT.md^) and re-run.
    exit /b 1
)

set "WINDOW_MS=%~1"
set "ACCEL_FS=%~2"
if "%ACCEL_FS%"=="" set "ACCEL_FS=25000"

if "%WINDOW_MS%"=="" (
    echo.
    echo [run_training] STOPPED: data passed the readiness check, but no
    echo analysis WINDOW_MS was supplied and none is safe to assume by default.
    echo Re-run as: run_training.bat ^<WINDOW_MS^> [ACCELERATION_FS_HZ]
    exit /b 1
)

if not exist "processed_features" mkdir "processed_features"

echo [run_training] Step 2/3: preparing synchronized per-trial feature caches...
python dataset_graph.py --prepare-raw-manifest raw_trial_manifest.csv --cache-output processed_features --window-ms %WINDOW_MS% --acceleration-fs %ACCEL_FS%
if errorlevel 1 (
    echo [run_training] STOPPED: feature preparation failed -- see error above.
    exit /b 1
)

echo [run_training] Step 3/3: leave-one-trial-out training and evaluation...
python train_evaluate.py --root processed_features --manifest processed_features\trial_manifest.csv --labels zone_labels.csv --output outputs --window-ms %WINDOW_MS% --acceleration-fs %ACCEL_FS%
if errorlevel 1 (
    echo [run_training] STOPPED: training failed -- see error above.
    exit /b 1
)

echo [run_training] Done. Results written to outputs\
exit /b 0
