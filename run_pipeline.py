"""
run_pipeline.py - runs the post-audit pipeline step by step.

    python run_pipeline.py prepare     raw force/acceleration CSVs -> zone features + labels   (~1 min)
    python run_pipeline.py mapping     trial-mapping evidence (measured speed, force trend, provenance)
    python run_pipeline.py train       LOTO benchmark, all models, 5 seeds   (~2-3 h on a 6-core CPU)
    python run_pipeline.py evaluate    metrics, CIs, permutation nulls, paired tests -> results/*.csv
    python run_pipeline.py figures     paper/figures/* from results/*.csv only
    python run_pipeline.py paper       paper/manuscript.docx, then check_numbers.py
    python run_pipeline.py pdf         LibreOffice PDF + page render + blank-equation check
    python run_pipeline.py test        unit tests
    python run_pipeline.py all         everything above in order
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = {
    "prepare": [[PY, "prepare_dataset.py", "--workers", "4"], "src"],
    "mapping": [[PY, "mapping_evidence.py"], "src"],
    "train": [[PY, "run_experiments.py", "--workers", "6"], "src"],
    "evaluate": [[PY, "evaluate.py"], "src"],
    "figures": [[PY, "make_figures.py"], "."],
    "paper": [[PY, "paper/build_manuscript.py"], "."],
    "check": [[PY, "check_numbers.py"], "."],
    "pdf": [[PY, "paper/check_pdf.py"], "."],
    "test": [[PY, "-m", "pytest", "tests", "-q"], "."],
}
ORDER = ["test", "prepare", "mapping", "train", "evaluate", "figures", "paper", "check", "pdf"]


def run(name: str) -> int:
    cmd, cwd = STEPS[name]
    print(f">>> {name}: {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=ROOT / cwd).returncode


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = ORDER if step == "all" else (["paper", "check"] if step == "paper" else [step])
    for n in names:
        if n not in STEPS:
            sys.exit(f"unknown step {n!r}; choose from {', '.join(list(STEPS) + ['all'])}")
        if run(n) != 0:
            sys.exit(f"step {n} failed")


if __name__ == "__main__":
    main()
