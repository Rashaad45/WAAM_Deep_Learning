"""Mesh-refinement and sensitivity table for the first four layers (plain Python 3).

Compares each verification run with the production run (identical model, so its first four
layers are the reference; truncation was shown to be exact). Writes
fea_results/verification_table.csv and .md with absolute values and % deviations.
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "fea_results"
RUNS = [("reference (production run)", "final"), ("mesh refined x2 (all edges halved)", "v_r2_L4"),
        ("arc power +10 %", "v_q110_L4"), ("arc power -10 %", "v_q090_L4"),
        ("substrate underside h = 100 W m-2 K-1", "v_hb100_L4")]
NSEG = 46


def steady(mp):
    return mp[(mp.segment >= 10) & (mp.segment <= 37)]


def metrics(run):
    lm = pd.read_csv(RES / f"{run}_layer_metrics.csv").set_index("layer")
    mp = steady(pd.read_csv(RES / f"{run}_meltpool.csv")).groupby("layer").mean(numeric_only=True)
    out = {}
    for l in (1, 2, 3, 4):
        out[f"peak T L{l} (°C)"] = lm.loc[l, "peak_T"]
    for l in (3, 4):
        out[f"pool depth L{l} (mm)"] = mp.loc[l, "depth_mm"]
        out[f"re-melt depth L{l} (mm)"] = mp.loc[l, "remelt_depth_mm"]
        out[f"pool length L{l} (mm)"] = mp.loc[l, "length_mm"]
    out["solidification rate L3 (°C/s)"] = lm.loc[3, "solidification_rate_C_s"]
    out["T at start of layer 4, L3 top (°C)"] = lm.loc[3, "interpass_T_at_next_layer"]
    return out


def main():
    rows = {}
    for label, run in RUNS:
        if (RES / f"{run}_layer_metrics.csv").exists():
            rows[label] = metrics(run)
    df = pd.DataFrame(rows)
    ref = df.iloc[:, 0]
    show = df.copy().astype(object)
    for c in df.columns[1:]:
        dev = 100 * (df[c] - ref) / ref.abs()
        show[c] = [f"{v:.3g} ({d:+.1f} %)" if "re-melt" not in i else f"{v:.2f} ({v - r:+.2f} mm)"
                   for i, v, d, r in zip(df.index, df[c], dev, ref)]
    show[df.columns[0]] = [f"{v:.3g}" for v in ref]
    df.to_csv(RES / "verification_table.csv", float_format="%.4g")
    try:
        md = show.to_markdown()
    except ImportError:                       # 'tabulate' not installed
        md = show.to_string()
    (RES / "verification_table.md").write_text(md, encoding="utf-8")
    print(show.to_string())


if __name__ == "__main__":
    main()
