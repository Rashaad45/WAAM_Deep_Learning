"""
install_dependencies.py
=======================
Installs the Python packages listed in requirements.txt (run this once).

    python install_dependencies.py

It does exactly the same as typing:
    python -m pip install -r requirements.txt
"""
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"


def main() -> None:
    # sys.executable makes sure the packages go into the SAME Python that runs this file.
    command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        print("Installation failed - see the pip messages above.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
