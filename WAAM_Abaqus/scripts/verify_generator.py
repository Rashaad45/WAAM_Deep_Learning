"""Check that generate_waam_inp.py (refine=1, 12 layers) reproduces waam_al5356.inp.

Compares node coordinates, connectivity, every node/element set, and for every step the
born element set, the Goldak element fluxes, the film/radiation face lists and the time
increment line. Prints a summary and exits non-zero on any mismatch.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_waam_inp import write_deck  # noqa: E402
from inp_tools import INP_DEFAULT, read_deck  # noqa: E402


def bc_lines(step):
    return [l.strip() for l in step.lines
            if l.startswith("WAAM-1.") and (", F" in l or ", R" in l) and "BF" not in l]


def main():
    tmp = Path(tempfile.gettempdir()) / "waam_regen_check.inp"
    write_deck(tmp)
    a, b = read_deck(INP_DEFAULT), read_deck(tmp)
    bad = []
    if a.nodes.keys() != b.nodes.keys():
        bad.append("node ids differ")
    else:
        dx = max(np.abs(a.nodes[i] - b.nodes[i]).max() for i in a.nodes)
        print(f"nodes: {len(a.nodes)}  max coordinate difference {dx:.2e} m")
        if dx > 1e-9:
            bad.append("node coordinates")
    same_conn = a.elems.keys() == b.elems.keys() and all((a.elems[e] == b.elems[e]).all() for e in a.elems)
    print(f"elements: {len(a.elems)}  connectivity identical: {same_conn}")
    if not same_conn:
        bad.append("connectivity")
    for kind, A, B in [("nset", a.nsets, b.nsets), ("elset", a.elsets, b.elsets)]:
        miss = set(A) ^ set(B)
        diff = [k for k in set(A) & set(B) if sorted(A[k]) != sorted(B[k])]
        print(f"{kind}s: {len(A)} vs {len(B)}  name mismatches {len(miss)}  content mismatches {len(diff)}")
        if miss or diff:
            bad.append(f"{kind}s {sorted(miss)[:5]} {diff[:5]}")
    print(f"steps: {len(a.steps)} vs {len(b.steps)}")
    worst = 0.0
    for sa, sb in zip(a.steps, b.steps):
        if sa.name != sb.name:
            bad.append(f"step name {sa.name} vs {sb.name}")
            break
        if sa.added_sets() != sb.added_sets():
            bad.append(f"{sa.name}: born set")
        fa, fb = sa.dflux(), sb.dflux()
        if fa.keys() != fb.keys():
            bad.append(f"{sa.name}: flux element list")
        elif fa:
            worst = max(worst, max(abs(fa[e] / fb[e] - 1) for e in fa))
        if sorted(bc_lines(sa)) != sorted(bc_lines(sb)):
            bad.append(f"{sa.name}: film/radiation faces")
        if [float(v) for v in sa.lines[2].split(",")] != [float(v) for v in sb.lines[2].split(",")]:
            bad.append(f"{sa.name}: increment line {sa.lines[2]} vs {sb.lines[2]}")
    print(f"max relative difference in element body flux: {worst:.2e}")
    if bad:
        print("MISMATCHES:", *bad[:20], sep="\n  ")
        sys.exit(1)
    print("OK - the generator reproduces waam_al5356.inp")


if __name__ == "__main__":
    main()
