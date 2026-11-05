"""Small python-docx layer that writes an MDPI-style manuscript with native OMML equations.

Text strings may contain inline math between $...$; every $...$ becomes a native Word equation.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from omml import latex_to_omml

FONT = "Palatino Linotype"


class Manuscript:
    def __init__(self):
        self.doc = Document()
        self.eq_no = 0
        self.fig_no = 0
        self.tab_no = 0
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = sec.right_margin = Cm(2.0)
        sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(10)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = st.paragraph_format
        pf.space_after, pf.space_before, pf.line_spacing = Pt(3), Pt(0), 1.15
        for name, size in (("Heading 1", 11), ("Heading 2", 10), ("Heading 3", 10)):
            h = self.doc.styles[name]
            h.font.name, h.font.size, h.font.bold = FONT, Pt(size), True
            h.font.italic = name in ("Heading 2", "Heading 3")
            h.font.bold = name == "Heading 1"
            h.font.color.rgb = RGBColor(0, 0, 0)
            rf = h.element.rPr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); h.element.rPr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(a), FONT)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(9), Pt(3)

    # ------------------------------------------------------------------ text
    def _add_runs(self, p, text: str, bold=False, italic=False, size=None):
        parts = re.split(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)", text)
        for part in parts:
            if not part:
                continue
            if part.startswith("$"):
                p._p.append(latex_to_omml(part[1:-1]))
            else:
                b, i, t = bold, italic, part
                if part.startswith("**"):
                    b, t = True, part[2:-2]
                elif part.startswith("*"):
                    i, t = True, part[1:-1]
                r = p.add_run(t)
                r.bold, r.italic = b, i
                if size:
                    r.font.size = Pt(size)
        return p

    def para(self, text: str, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, bold=False, italic=False,
             indent=True):
        p = self.doc.add_paragraph(style=style)
        p.alignment = align
        if indent and style is None:
            p.paragraph_format.first_line_indent = Cm(0.5)
        return self._add_runs(p, text, bold=bold, italic=italic, size=size)

    def heading(self, text: str, level: int = 1):
        h = self.doc.add_heading(level=level)
        self._add_runs(h, text)
        return h

    def bullets(self, items: list[str]):
        for it in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._add_runs(p, it)

    def numbered(self, items: list[str]):
        for k, it in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._add_runs(p, f"({k}) " + it)

    def equation(self, latex: str) -> int:
        """Display equation, centred, numbered at the right margin."""
        self.eq_no += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.tab_stops.add_tab_stop(Cm(8.5), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(17.0), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        p._p.append(latex_to_omml(latex))
        p.add_run(f"\t({self.eq_no})")
        return self.eq_no

    # ------------------------------------------------------------------ floats
    def figure(self, path: Path, caption: str, width_cm: float = 15.0) -> int:
        self.fig_no += 1
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        c = self.doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = c.add_run(f"Figure {self.fig_no}. "); r.bold = True; r.font.size = Pt(9)
        self._add_runs(c, caption, size=9)
        return self.fig_no

    def table(self, header: list[str], rows: list[list[str]], caption: str, note: str = "",
              widths: list[float] | None = None) -> int:
        self.tab_no += 1
        c = self.doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.keep_with_next = True
        r = c.add_run(f"Table {self.tab_no}. "); r.bold = True; r.font.size = Pt(9)
        self._add_runs(c, caption, size=9)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(header):
            cell = t.rows[0].cells[j]; cell.text = ""
            self._add_runs(cell.paragraphs[0], h, bold=True, size=8.5)
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row):
                cell = t.rows[i].cells[j]; cell.text = ""
                self._add_runs(cell.paragraphs[0], str(v), size=8.5)
        self._three_line(t)
        if widths:
            t.autofit = False
            layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
            t._tbl.tblPr.append(layout)
            for j, w in enumerate(widths):
                t.columns[j].width = Cm(w)
            for row in t.rows:
                for j, w in enumerate(widths):
                    row.cells[j].width = Cm(w)
        if note:
            n = self.doc.add_paragraph(); n.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._add_runs(n, note, size=8)
        else:
            self.doc.add_paragraph()
        return self.tab_no

    @staticmethod
    def _three_line(table):
        tbl = table._tbl
        pr = tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge, val in (("top", "single"), ("bottom", "single"), ("insideH", "nil"), ("insideV", "nil"),
                          ("left", "nil"), ("right", "nil")):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val); el.set(qn("w:sz"), "8"); el.set(qn("w:color"), "000000")
            borders.append(el)
        pr.append(borders)
        for cell in table.rows[0].cells:
            tcpr = cell._tc.get_or_add_tcPr()
            b = OxmlElement("w:tcBorders"); bot = OxmlElement("w:bottom")
            bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), "6"); bot.set(qn("w:color"), "000000")
            b.append(bot); tcpr.append(b)

    def page_break(self):
        self.doc.add_page_break()

    def save(self, path: Path):
        self.doc.save(str(path))
