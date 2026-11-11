"""Small reader for the WAAM Abaqus input deck (plain Python 3, no Abaqus needed).

Parses nodes, DC3D8 elements, node/element sets and the per-step keyword blocks of
waam_al5356.inp so the other scripts can reuse the mesh and the load history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INP_DEFAULT = HERE.parent / "waam_al5356.inp"


@dataclass
class Step:
    name: str
    first_line: int
    lines: list[str] = field(default_factory=list)

    @property
    def period(self) -> float:
        return float(self.lines[2].split(",")[1])

    def dflux(self) -> dict[int, float]:
        out, on = {}, False
        for l in self.lines:
            if l.startswith("*"):
                on = l.upper().startswith("*DFLUX")
                continue
            if on and l.strip():
                a = [s.strip() for s in l.split(",")]
                out[int(a[0].split(".")[-1])] = float(a[2])
        return out

    def added_sets(self) -> list[str]:
        out, on = [], False
        for l in self.lines:
            if l.startswith("*"):
                on = l.upper().startswith("*MODEL CHANGE") and "ADD" in l.upper()
                continue
            if on and l.strip():
                out += [s.strip().split(".")[-1] for s in l.split(",") if s.strip()]
        return out


@dataclass
class Deck:
    lines: list[str]
    nodes: dict[int, np.ndarray]
    elems: dict[int, np.ndarray]
    nsets: dict[str, list[int]]
    elsets: dict[str, list[int]]
    steps: list[Step]

    def node_array(self):
        ids = np.array(sorted(self.nodes))
        return ids, np.array([self.nodes[i] for i in ids])

    def elem_centroids(self):
        eids = np.array(sorted(self.elems))
        idx = {n: k for k, n in enumerate(sorted(self.nodes))}
        X = np.array([self.nodes[i] for i in sorted(self.nodes)])
        conn = np.array([[idx[n] for n in self.elems[e]] for e in eids])
        return eids, X[conn].mean(axis=1), X[conn]


def _ints(line: str) -> list[int]:
    return [int(v) for v in line.split(",") if v.strip()]


def read_deck(path: Path | str = INP_DEFAULT) -> Deck:
    L = Path(path).read_text().splitlines()
    nodes, elems, nsets, elsets, steps = {}, {}, {}, {}, []
    i, n = 0, len(L)
    while i < n:
        l = L[i]
        u = l.upper()
        if u.startswith("*NODE") and not u.startswith("*NODE OUTPUT"):
            i += 1
            while i < n and not L[i].startswith("*"):
                a = L[i].split(",")
                nodes[int(a[0])] = np.array([float(v) for v in a[1:4]])
                i += 1
            continue
        if u.startswith("*ELEMENT,") and "TYPE=" in u:
            i += 1
            while i < n and not L[i].startswith("*"):
                a = _ints(L[i])
                elems[a[0]] = np.array(a[1:9])
                i += 1
            continue
        if u.startswith("*NSET") or u.startswith("*ELSET"):
            name = re.search(r"(?:NSET|ELSET)=([^,\s]+)", l, re.I).group(1)
            gen = "GENERATE" in u
            ids: list[int] = []
            i += 1
            while i < n and not L[i].startswith("*"):
                a = _ints(L[i])
                if gen:
                    s, e, d = (a + [1])[:3]
                    ids += list(range(s, e + 1, d))
                else:
                    ids += a
                i += 1
            (nsets if u.startswith("*NSET") else elsets)[name] = ids
            continue
        if u.startswith("*STEP"):
            name = re.search(r"name=([^,\s]+)", l, re.I).group(1)
            st = Step(name, i)
            while i < n and not L[i].upper().startswith("*END STEP"):
                st.lines.append(L[i])
                i += 1
            steps.append(st)
        i += 1
    return Deck(L, nodes, elems, nsets, elsets, steps)
