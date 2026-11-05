"""
check_pdf.py - docx -> PDF with LibreOffice, every page -> PNG, and a blank-equation check.

Fails (exit 1) if
  * any inline or display equation of manuscript.docx is missing from the PDF text at its position
    (>= 80 % of its characters must appear right after the preceding words / right before its number), or
  * the PDF text contains ", , " or "where is" (signatures of dropped symbols).

    python paper/check_pdf.py
"""
import re
import unicodedata
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
from lxml import etree

HERE = Path(__file__).resolve().parent
DOCX, PDF, PAGES = HERE / "manuscript.docx", HERE / "manuscript.pdf", HERE / "pages"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
SOFFICE = [Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
           Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe")]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("−", "-").replace("\u2061", "").replace("\u2062", "").replace("\u200b", "")
    return re.sub(r"\s+", "", s)


def convert() -> None:
    import os
    env = os.environ.get("SOFFICE_PATH")
    exe = ((Path(env) if env and Path(env).exists() else None)
           or next((p for p in SOFFICE if p.exists()), None) or shutil.which("soffice"))
    if exe is None:
        sys.exit("LibreOffice (soffice) not found")
    subprocess.run([str(exe), "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(DOCX)], check=True,
                   timeout=600)


def render() -> str:
    PAGES.mkdir(exist_ok=True)
    for old in PAGES.glob("*.png"):
        old.unlink()
    doc = fitz.open(PDF)
    texts = []
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=110).save(PAGES / f"page_{i:02d}.png")
        texts.append(page.get_text())
    print(f"rendered {len(doc)} pages to {PAGES}")
    return "\n".join(texts)


def equations():
    """Yield (kind, anchor_text, math_text) in document order."""
    root = etree.fromstring(zipfile.ZipFile(DOCX).read("word/document.xml"))
    for p in root.iter(f"{W}p"):
        before = ""
        children = list(p)
        for k, ch in enumerate(children):
            if ch.tag == f"{W}r":
                before += "".join(t.text or "" for t in ch.iter(f"{W}t"))
            elif ch.tag == f"{M}oMath":
                math = "".join(t.text or "" for t in ch.iter(f"{M}t"))
                after = "".join("".join(t.text or "" for t in c.iter(f"{W}t")) for c in children[k + 1:])
                m = re.match(r"\s*\((\d+)\)", after)
                if m and not before.strip():
                    yield "display", f"({m.group(1)})", math
                else:
                    yield "inline", before[-30:], math
                before += math


def missing_glyphs(dpi: int = 200) -> list[tuple]:
    """Non-ASCII characters whose box on the rendered page contains no ink (the glyph is blank).

    Every character above U+007E (Greek letters, math operators, bold/script math letters, dashes...) is
    located with PyMuPDF, its bounding box is cropped from a 200 dpi greyscale render, and the box must
    contain at least one pixel darker than mid-grey.
    """
    doc = fitz.open(PDF)
    out = []
    zoom = dpi / 72.0
    for pno, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
        w, h, buf = pix.width, pix.height, pix.samples
        for b in page.get_text("rawdict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    for ch in sp["chars"]:
                        c = ch["c"]
                        if ord(c) <= 0x7E or not c.strip() or unicodedata.category(c).startswith("Z"):
                            continue
                        x0, y0, x1, y1 = (int(v * zoom) for v in ch["bbox"])
                        x0, y0 = max(x0, 0), max(y0, 0)
                        x1, y1 = min(max(x1, x0 + 1), w), min(max(y1, y0 + 1), h)
                        dark = any(buf[y * w + x] < 128 for y in range(y0, y1) for x in range(x0, x1))
                        if not dark:
                            out.append((pno, sp["font"], c, hex(ord(c))))
    return out


def main() -> int:
    if "--no-convert" not in sys.argv:
        convert()
    text = render()
    flat = norm(text)
    failures, n, cursor = [], 0, 0
    for kind, anchor, math in equations():
        n += 1
        need = Counter(c for c in norm(math) if c.isalnum())
        if not need:
            need = Counter(norm(math))
        a = norm(anchor)
        if kind == "inline":
            key = a[-15:]
            pos = flat.find(key, cursor) if key else -1
            if pos >= 0:
                cursor = pos
            window = flat[pos + len(key): pos + len(key) + 3 * len(norm(math)) + 25] if pos >= 0 else ""
        else:
            # display equation: its number "(k)" is the first "(k)" at or after the current reading position
            # that is preceded by the equation's own characters
            pos = flat.find(a, cursor)
            if pos >= 0:
                cursor = pos + len(a)
            window = flat[max(0, pos - 4 * len(norm(math)) - 40): pos] if pos >= 0 else ""
        have = Counter(window)
        found = sum(min(v, have[k]) for k, v in need.items())
        if pos < 0 or found < 0.8 * sum(need.values()):
            failures.append((kind, anchor.strip()[-30:], math[:40]))
    # ", , " / "where is": dropped symbols; "\u00bf": LibreOffice Math syntax-error marker (red inverted ?)
    bad_patterns = [pat for pat in (", , ", "where is", "\u00bf") if pat in text]
    glyphs = missing_glyphs()
    for g in sorted(set(glyphs)):
        failures.append(("missing glyph (renders blank)", f"page {g[0]} font {g[1]}", f"{g[2]} {g[3]}"))
    print(f"equations checked: {n}; missing/blank: {len(failures)}")
    for f in failures:
        print("  MISSING", f)
    for pat in bad_patterns:
        print(f"  PDF text contains {pat!r}")
    ok = not failures and not bad_patterns
    print("PDF PAGE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
