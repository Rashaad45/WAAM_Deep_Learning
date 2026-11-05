"""
check_numbers.py - every number in paper/manuscript.docx must come from results/*.csv.

Extracts each numeric token from the body text and tables (equations, the reference list,
citation brackets, figure/table/section/equation labels, DOIs and instrument / alloy designations
are excluded) and checks that some value in results/*.csv rounds to it at the token's own precision.
Exit code 1 and a list of offenders if any number cannot be traced.

    python check_numbers.py [paper/manuscript.docx]
"""
import re
import sys
from pathlib import Path

import docx
import pandas as pd

ROOT = Path(__file__).resolve().parent
DOC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "manuscript.docx"

# Tokens that are names, not quantities.
NOT_QUANTITIES = [
    r"\b(?:ER|AA|Al)\s?\d{4}\b", r"\b\d{4}[A-Z]\d*\b", r"\b[A-Z]{2,4}-\d{2}[A-Z]?(?:-[A-Z]\d{2})?\b",
    r"\bS235JR\b", r"\b7050-T7451\b", r"\b\d+[A-Z]+\d*\b", r"\b1D\b", r"\bST-G[AC][TN]\b",
    r"doi:\S+", r"https?://\S+", r"\b10\.\d{4,5}/\S+", r"arXiv:\d{4}\.\d{4,5}(?:v\d)?",
    r"\[\d+(?:[,–-]\s?\d+)*\]",                                  # citations
    r"\b(?:Figure|Fig\.|Table|Section|Sections|Equation|Eq\.|Eqs\.|Appendix|Step)\s?[A-Z]?\d+(?:\.\d+)*(?:\s?(?:and|–|-)\s?\d+(?:\.\d+)*)?",
    r"\(\d+\)",                                                    # equation numbers
    r"^\s*\d+(?:\.\d+)*\.\s",                                      # numbered headings
    r"\bCC BY 4\.0\b", r"\bE[1-4]\b", r"\bF\d\b", r"\bL\d\b",
]


# Row-level files (per-zone predictions, per-epoch losses, per-zone label details) are EXCLUDED: with
# thousands of values they would make almost any number "traceable". Only summary tables count.
ROW_LEVEL = {"predictions.csv", "training_loss.csv", "training_loss_curves.csv", "zone_label_details.csv",
             "per_fold.csv"}


def allowed_values() -> list[float]:
    vals = []
    import os
    for f in Path(os.environ.get("WAAM_RESULTS_DIR", ROOT / "results")).glob("*.csv"):
        if f.name in ROW_LEVEL or f.name.startswith("predictions_"):
            continue
        d = pd.read_csv(f)
        for c in d.columns:
            s = d[c]
            num = pd.to_numeric(s, errors="coerce")
            vals += [abs(float(v)) for v in num.dropna()]
            for txt in s[num.isna()].dropna().astype(str):
                vals += [abs(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return vals


def body_text(path: Path) -> list[str]:
    d = docx.Document(path)
    out, in_refs = [], False
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for block in d.element.body.iter(f"{W}p"):
        text = "".join(t.text or "" for t in block.iter(f"{W}t"))          # w:t only: equations excluded
        if re.match(r"^\s*References\s*$", text):
            in_refs = True
        if not in_refs and text.strip():
            out.append(text)
    return out


def tokens(text: str) -> list[str]:
    for pat in NOT_QUANTITIES:
        text = re.sub(pat, " ", text)
    return re.findall(r"(?<![\w.])[−-]?\d+(?:[.,]\d+)?(?![\w])", text)


def matches(tok: str, vals: list[float]) -> bool:
    t = tok.replace("−", "-").replace(",", "").lstrip("-")
    dec = len(t.split(".")[1]) if "." in t else 0
    x = float(t)
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return any(abs(v - x) <= tol or abs(100 * v - x) <= tol for v in vals)


def main() -> int:
    vals = allowed_values()
    bad = []
    n = 0
    for para in body_text(DOC):
        for tok in tokens(para):
            n += 1
            if not matches(tok, vals):
                bad.append((tok, para[:110]))
    print(f"check_numbers: {n} numeric tokens checked against results/*.csv")
    if bad:
        print(f"FAIL: {len(bad)} number(s) not traceable:")
        for tok, ctx in bad:
            print(f"  {tok!r:>10}  in: {ctx}")
        return 1
    print("PASS: every number is traceable to results/*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
