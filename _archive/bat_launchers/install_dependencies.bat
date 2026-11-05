@echo off
rem Installs the Python packages listed in requirements.txt (run once).
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed - see the pip messages above.
    exit /b 1
)
echo Done.
