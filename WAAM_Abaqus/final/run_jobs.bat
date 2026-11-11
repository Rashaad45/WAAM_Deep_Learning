@echo off
rem Usage:  run_jobs.bat <cpus> <job1> [job2 ...]
rem Runs the listed jobs one after another, each in its own folder <job>\<job>.inp.
rem A job whose .sta already says COMPLETED SUCCESSFULLY is skipped. Log: run_jobs.log
cd /d "%~dp0"
set CPUS=%1
shift
:next
if "%~1"=="" goto done
call :runjob %~1
shift
goto next
:done
echo %date% %time%  queue finished>> run_jobs.log
goto :eof

:runjob
findstr /c:"COMPLETED SUCCESSFULLY" "%1\%1.sta" >nul 2>&1 && (echo %date% %time%  %1 already complete, skipped>> run_jobs.log & goto :eof)
echo %date% %time%  START %1 (cpus=%CPUS%)>> run_jobs.log
pushd %1
call C:\SIMULIA\Commands\abaqus.bat job=%1 input=%1.inp cpus=%CPUS% ask_delete=OFF interactive > %1.console.txt 2>&1
popd
echo %date% %time%  END   %1>> run_jobs.log
goto :eof
