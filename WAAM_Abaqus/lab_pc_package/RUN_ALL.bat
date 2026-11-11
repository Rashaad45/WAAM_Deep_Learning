@echo off
rem ======================================================================
rem  WAAM thermal FEA - runs the production model and the 4 verification
rem  models one after another. Double-click, or:  RUN_ALL.bat 8
rem  (the number = CPU cores to use; default 4). Resumable: finished jobs
rem  are skipped. Progress is written to run_log.txt.
rem ======================================================================
cd /d "%~dp0"
set CPUS=%1
if "%CPUS%"=="" set CPUS=4
set ABQ=abaqus
where abaqus >nul 2>&1 || set ABQ=C:\SIMULIA\Commands\abaqus.bat

call :job final final
call :job verification_runs\v_q110_L4 v_q110_L4
call :job verification_runs\v_q090_L4 v_q090_L4
call :job verification_runs\v_hb100_L4 v_hb100_L4
call :job verification_runs\v_r2_L4 v_r2_L4
echo %date% %time%  ALL JOBS FINISHED>> run_log.txt
echo.
echo All jobs finished. Now run:  powershell -ExecutionPolicy Bypass -File postprocess_all.ps1
pause
goto :eof

:job
findstr /c:"COMPLETED SUCCESSFULLY" "%1\%2.sta" >nul 2>&1 && (echo %date% %time%  %2 already done>> run_log.txt & goto :eof)
echo %date% %time%  START %2 (cpus=%CPUS%)>> run_log.txt
echo Running %2 ...
pushd %1
call %ABQ% job=%2 input=%2.inp cpus=%CPUS% ask_delete=OFF interactive > %2.console.txt 2>&1
popd
echo %date% %time%  END   %2>> run_log.txt
goto :eof
