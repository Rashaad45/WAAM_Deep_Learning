@echo off
setlocal

rem Installs Python dependencies for the WAAM GNN pipeline into whichever
rem "python" is first on PATH. Run this once before run_training.bat.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [install_dependencies] ERROR: "python" was not found on PATH.
    echo Install Python 3.10+ and re-run this script.
    exit /b 1
)

echo [install_dependencies] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [install_dependencies] Installing requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [install_dependencies] Done. torch_geometric and torch are CPU builds by
echo default; if you have a CUDA GPU, reinstall torch from
echo https://pytorch.org/get-started/locally/ for a matching CUDA wheel
echo before using this environment.
exit /b 0

:fail
echo [install_dependencies] FAILED -- see pip output above.
exit /b 1
