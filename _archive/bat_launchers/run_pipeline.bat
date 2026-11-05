@echo off
rem ============================================================================
rem  WAAM ST-GNN pipeline - one command per step.
rem
rem    run_pipeline.bat prepare     Step 1: raw WAAM files -> features + labels   (~5 min)
rem    run_pipeline.bat train       Step 2a: main comparison, Tables 2, 3, 5      (~2 h on CPU)
rem    run_pipeline.bat ablations   Step 2b: ST-GAT ablations, Table 4            (~3 h on CPU)
rem    run_pipeline.bat loco        optional: leave-one-condition-out, Table S1   (~1.5 h)
rem    run_pipeline.bat report      Step 3: fill the manuscript + figures         (~1 min)
rem    run_pipeline.bat quick       5-minute check that everything runs
rem    run_pipeline.bat all         prepare + train + ablations + report
rem
rem  Re-run "report" after every training step; it fills whatever is available.
rem ============================================================================
setlocal
cd /d "%~dp0src"

if "%~1"=="" goto :usage
if /i "%~1"=="prepare"   goto :prepare
if /i "%~1"=="train"     goto :train
if /i "%~1"=="ablations" goto :ablations
if /i "%~1"=="loco"      goto :loco
if /i "%~1"=="report"    goto :report
if /i "%~1"=="quick"     goto :quick
if /i "%~1"=="all"       goto :all
goto :usage

:prepare
python prepare_dataset.py --workers 3 || exit /b 1
goto :eof

:train
python train_evaluate.py --stage main || exit /b 1
goto :eof

:ablations
python train_evaluate.py --stage ablations || exit /b 1
goto :eof

:loco
python train_evaluate.py --stage main --protocol loco --models fusion_mlp gat || exit /b 1
goto :eof

:report
python make_report.py || exit /b 1
goto :eof

:quick
python train_evaluate.py --quick --stage all --output ..\outputs_quick_check || exit /b 1
echo Quick check finished. Its numbers are meaningless (3 epochs, 2 folds); delete outputs_quick_check when done.
goto :eof

:all
python prepare_dataset.py --workers 3 || exit /b 1
python train_evaluate.py --stage main || exit /b 1
python train_evaluate.py --stage ablations || exit /b 1
python make_report.py || exit /b 1
goto :eof

:usage
echo Usage: run_pipeline.bat [prepare ^| train ^| ablations ^| loco ^| report ^| quick ^| all]
exit /b 1
