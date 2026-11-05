import re
import sys
import zipfile
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(sys.argv[1])
for name in ["WAAM_STGNN_Materials_draft.docx", "WAAM_STGNN_Materials_filled.docx", "WAAM_STGNN_Materials_FINAL.docx"]:
    p = ROOT / "report" / name
    xml = zipfile.ZipFile(p).read("word/document.xml").decode("utf8")
    n_omath = xml.count('<m:oMath>') + xml.count('<m:oMath ')
    n_hl = len(re.findall(r'<w:highlight w:val="(\w+)"', xml))
    n_cyan = xml.count('w:val="cyan"')
    has_comments = 'word/comments.xml' in zipfile.ZipFile(p).namelist()
    print("\n###", name, "oMath=", n_omath, "oMathPara=", xml.count('<m:oMathPara'), "highlight=", n_hl,
          "cyan=", n_cyan, "ins=", xml.count('<w:ins '), "del=", xml.count('<w:del '), "comments=", has_comments)

# Plain text of FINAL docx (paragraph text incl. math text runs m:t)
xml = zipfile.ZipFile(ROOT / "report/WAAM_STGNN_Materials_FINAL.docx").read("word/document.xml").decode("utf8")
paras = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
text = []
for p in paras:
    t = "".join(re.findall(r"<(?:w|m):t(?: [^>]*)?>([^<]*)</(?:w|m):t>", p))
    text.append(t)
full = "\n".join(text)
(Path(sys.argv[2]) / "final_docx_text.txt").write_text(full, encoding="utf8")
for pat in [r"λ|lambda|\\lambda", r"BCE", r"author to", r"material flag|material-flag", r"DeLong", r"0\.098", r"emissiv", r"accelerometer", r"Ra\b.*?profilom|profilomet"]:
    hits = [(i, l[:220]) for i, l in enumerate(text) if re.search(pat, l, flags=re.I)]
    print(f"\n--- /{pat}/ : {len(hits)} paragraphs")
    for i, l in hits[:8]:
        print(f"  [{i}] {l}")

# cyan-highlighted runs
for p in paras:
    if 'w:val="cyan"' in p:
        print("\nCYAN PARA:", "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p))[:400])

pdf = PdfReader(ROOT / "report/WAAM_STGNN_Materials_humanized.pdf")
print("\nPDF metadata:", dict(pdf.metadata or {}), "pages:", len(pdf.pages))
ptxt = [pg.extract_text() or "" for pg in pdf.pages]
(Path(sys.argv[2]) / "humanized_pdf_text.txt").write_text("\n\n=====PAGE=====\n".join(ptxt), encoding="utf8")
for i, t in enumerate(ptxt):
    for m in re.finditer(r".{0,60}(, , |, \.|where is|\( \)|  ,).{0,60}", t):
        print(f"  p{i+1}: ...{m.group(0)!r}")
