"""LaTeX -> MathML (latex2mathml) -> native Word OMML (Office's MML2OMML.XSL)."""
from pathlib import Path

from latex2mathml.converter import convert
from lxml import etree

_XSL_CANDIDATES = [Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
                   Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL")]
_XSL = next(p for p in _XSL_CANDIDATES if p.exists())
_TRANSFORM = etree.XSLT(etree.parse(str(_XSL)))
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def latex_to_omml(latex: str) -> etree._Element:
    """Return an <m:oMath> element for a LaTeX expression."""
    mathml = convert(latex)
    tree = etree.fromstring(mathml.encode("utf8"))
    omml = _TRANSFORM(tree).getroot()
    if omml.tag == f"{{{M_NS}}}oMathPara":
        omml = omml.find(f"{{{M_NS}}}oMath")
    assert omml is not None and omml.tag == f"{{{M_NS}}}oMath", "OMML conversion failed"
    if not "".join(omml.itertext()).strip():
        raise ValueError(f"empty OMML for {latex!r}")
    return omml
