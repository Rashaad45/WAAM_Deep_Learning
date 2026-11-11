"""
rev_tables.py - renders every RESULTS_SUMMARY table as Markdown directly from the revision CSVs
(AUC/AP/Brier etc. 2 decimals, p 3 decimals). Output: revision_results/summary_tables.md
"""
import numpy as np
import pandas as pd

import rev_common as C

REV = C.REV
N = C.NAMES


def f2(x):
    return "–" if pd.isna(x) else f"{x:.2f}"


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def ci(lo, hi):
    return f"{f2(lo)} to {f2(hi)}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def t_ablation():
    a = pd.read_csv(REV / "ablation_features.csv")
    p = pd.read_csv(REV / "ablation_features_paired.csv").set_index(["model", "config"])
    h = pd.read_csv(REV / "holm_adjusted.csv")
    h = h[h.family.str.startswith("(iii)")].set_index("item").p_holm
    rows = []
    for r in a.itertuples():
        k = (r.model, r.config)
        pr = p.loc[k] if k in p.index else None
        rows.append(dict(Model=N[r.model], Config=r.config, **{"n feat": r.n_features},
                         **{"AUC (95% CI)": f"{f2(r.auc)} ({ci(r.auc_ci_low, r.auc_ci_high)})",
                            "AP (95% CI)": f"{f2(r.average_precision)} ({ci(r.ap_ci_low, r.ap_ci_high)})",
                            "ΔAUC vs F0 (95% CI)": "–" if pr is None else f"{pr.delta_auc:+.2f} ({ci(pr.delta_ci_low, pr.delta_ci_high)})",
                            "paired p": "–" if pr is None else f3(pr.paired_perm_p),
                            "Holm p (per model)": "–" if pr is None else f3(h.get(f"{r.model}: {r.config} − F0", np.nan))}))
    order = {m: i for i, m in enumerate(C.MODEL_ORDER)}
    d = pd.DataFrame(rows)
    d["_o"] = a.model.map(order).to_numpy()
    return md(d.sort_values(["_o", "Config"], kind="stable").drop(columns="_o"))


def t_single():
    s = pd.read_csv(REV / "single_feature_auc.csv")
    req = s[s.requested].sort_values("auc_oriented", ascending=False)
    top = s[~s.requested].sort_values("auc_oriented", ascending=False).head(10)
    kin = s[s.feature.isin(["spindle_speed", "zone_position"])]

    def tab(x):
        return md(pd.DataFrame(dict(Feature=x.feature, Sign=x.sign_chosen,
                                    **{"AUC (oriented, 95% CI)": [f"{f2(a)} ({ci(l, h)})" for a, l, h in
                                                                  zip(x.auc_oriented, x.auc_ci_low, x.auc_ci_high)],
                                       "zone-perm p": x.zone_perm_p_two_sided.map(f3)})))
    return tab(req), tab(top), tab(kin)


def t_perm():
    p = pd.read_csv(REV / "permutation_cluster.csv")
    h = pd.read_csv(REV / "holm_adjusted.csv")
    hz = h[h.family == "(i) Table 3, zone-level permutation (recomputed this run, rng 0)"].set_index("item").p_holm
    hb = h[h.family == "(i) Table 3, trial-block permutation (this run)"].set_index("item").p_holm
    rows = []
    for (m, cfg), g in p[p.retrained == False].groupby(["model", "config"], sort=False):  # noqa: E712
        g = g.set_index("null")
        z, b, w = g.loc["zone_level"], g.loc["trial_block"], g.loc["within_trial"]
        rows.append({"Model": N[m], "Config": cfg, "AUC": f2(z.auc),
                     "zone-level null 95%": ci(z.null_q025, z.null_q975), "p zone": f3(z.p_two_sided),
                     "(a) trial-block null 95%": ci(b.null_q025, b.null_q975), "p block": f3(b.p_two_sided),
                     "p block Holm": f3(hb.get(m)) if cfg == "F0" else "–",
                     "(b) within-trial null 95%": ci(w.null_q025, w.null_q975),
                     "within-trial null mean": f2(w.null_mean), "p within (one-sided)": f3(w.p_one_sided)})
    rc = p[p.retrained == True]  # noqa: E712
    rows_c = [{"Model": N[r.model], "Config": r.config, "AUC": f2(r.auc), "permutations": int(r.n_permutations),
               "seeds per permutation": int(r.n_seeds_per_permutation) if "n_seeds_per_permutation" in rc else "",
               "null mean": f2(r.null_mean), "null 95%": ci(r.null_q025, r.null_q975),
               "p two-sided": f3(r.p_two_sided), "p one-sided": f3(r.p_one_sided)} for r in rc.itertuples()]
    return md(pd.DataFrame(rows)), (md(pd.DataFrame(rows_c)) if rows_c else "_not run_")


def t_holm():
    h = pd.read_csv(REV / "holm_adjusted.csv")
    out = []
    for fam, g in h.groupby("family", sort=False):
        d = pd.DataFrame({"Item": g["item"], "p raw": g.p_raw.map(f3), "p Holm": g.p_holm.map(f3),
                          "sig. raw": np.where(g.p_raw < .05, "yes", ""), "sig. Holm": np.where(g.p_holm < .05, "yes", "")})
        out.append(f"**{fam}** (family size {len(g)}; raw p source: {g.p_source.iloc[0]})\n\n" + md(d))
    return "\n\n".join(out)


def t_levels():
    d = pd.read_csv(REV / "discrimination_levels.csv")
    rows = pd.DataFrame({"Model": d.model.map(N), "Config": d.config, "Pooled AUC": d.pooled_auc.map(f2),
                         "Within-trial mean AUC (n trials)": [f"{f2(a)} ({n})" for a, n in zip(d.within_trial_auc_mean, d.n_trials_both_classes)],
                         "Trial-level AUC ≥1 unstable (95% CI)": [f"{f2(a)} ({ci(l, h)})" for a, l, h in zip(
                             d.trial_level_auc_any_unstable, d.trial_level_auc_ci_low, d.trial_level_auc_ci_high)],
                         "Spearman ρ (p)": [f"{f2(r)} ({f3(p)})" for r, p in zip(d.spearman_rho_score_vs_n_unstable, d.spearman_p)],
                         "Pooled AUC w/o 04, 06": d.pooled_auc_without_04_06.map(f2)})
    w = pd.read_csv(REV / "within_trial_per_trial.csv")
    piv = w.pivot_table(index=["model", "config"], columns="trial_id", values="within_trial_auc", sort=False)
    piv = piv.reset_index()
    piv.insert(0, "Model", piv.pop("model").map(N))
    lab = w.drop_duplicates("trial_id").set_index("trial_id")[["zone_labels"]]
    lab["zone_labels"] = lab.zone_labels.astype(str).str.zfill(3)
    for c in piv.columns[2:]:
        piv[c] = piv[c].map(f2)
    piv.columns = [c if not str(c).startswith("trial_") else f"{c[-2:]} [{lab.loc[c, 'zone_labels']}]" for c in piv.columns]
    return md(rows), md(piv), ", ".join(sorted(w.trial_id.unique()))


def t_cal():
    c = pd.read_csv(REV / "calibration.csv")
    return md(pd.DataFrame({"Model": c.model.map(N), "Config": c.config, "Brier": c.brier.map(f3),
                            "Brier (constant 0.52)": c.brier_constant_prevalence.map(f3),
                            "Brier skill": c.brier_skill_score.map(f2), "ECE (5 bins)": c.ece_5_equal_count_bins.map(f3),
                            "Slope (95% CI)": [f"{f2(s)} ({ci(l, h)})" for s, l, h in zip(c.calibration_slope, c.slope_ci_low, c.slope_ci_high)],
                            "Intercept": c.calibration_intercept.map(f2), "CITL": c.calibration_in_the_large.map(f2),
                            "Probability": c.probability_source}))


def t_repro():
    r = pd.read_csv(REV / "reproduction_check.csv")
    return md(pd.DataFrame({"Model": r.model.map(N), "Retrained here": r.recomputed,
                            "max |Δp| vs results/predictions.csv": r.max_abs_diff_seed_level.map(lambda v: "–" if pd.isna(v) else f"{v:.2g}"),
                            "AUC this run": r.auc_this_run.map(f2), "AUC manuscript": r.auc_manuscript.map(f2)}))


def t_importance():
    i = pd.read_csv(REV / "importance.csv")
    out = []
    for m in ("logreg", "rf"):
        g = i[(i.model == m) & (i.level == "group")]
        a = g[g.scheme == "across_oof"].set_index("feature_or_group").delta_auc_mean
        w = g[g.scheme == "within_fold"].set_index("feature_or_group").delta_auc_mean
        d = pd.DataFrame({"Group": a.index, "ΔAUC across": a.map(f3).to_numpy(), "ΔAUC within-trial": w.reindex(a.index).map(f3).to_numpy()})
        d = d.iloc[np.argsort(-a.to_numpy())]
        out.append(f"**{N[m]}** (AUC {f2(g.auc_original.iloc[0])})\n\n" + md(d))
    return "\n\n".join(out)


def t_seed_spread():
    """Max over zones of (max - min) probability across the 5 seeds, per model x config (-> seed_spread.csv)."""
    rows = []
    for f in sorted((C.RAW).glob("*_F[0-9].csv")):
        d = pd.read_csv(f)
        m, cfg = f.stem.rsplit("_", 1)
        g = d.groupby(["trial_id", "zone_id"]).probability
        rows.append(dict(model=m, config=cfg, n_seeds=d.seed.nunique(),
                         max_seed_range=float((g.max() - g.min()).max())))
    s = pd.DataFrame(rows)
    s.to_csv(REV / "seed_spread.csv", index=False)
    return md(pd.DataFrame({"Model": s.model.map(N), "Config": s.config, "seeds": s.n_seeds,
                            "max over zones of seed range of p": s.max_seed_range.map(lambda v: f"{v:.2g}")}))


def t_runtimes():
    r = pd.read_csv(REV / "run_times.csv")
    g = r.groupby("model", sort=False).agg(configs=("config", lambda x: " ".join(x)),
                                           cpu_s=("cpu_seconds_all_seeds", "sum"), last=("finished", "max")).reset_index()
    return md(pd.DataFrame({"Model": g.model.map(N), "Configs trained": g.configs,
                            "CPU time, all configs × 5 seeds (min)": (g.cpu_s / 60).map(lambda v: f"{v:.1f}"),
                            "Last finished": g["last"]}))


if __name__ == "__main__":
    parts = [("Reproduction check (reproduction_check.csv)", t_repro())]
    parts.append(("Seed dependence of the retrained probabilities (seed_spread.csv, from oof_raw/)", t_seed_spread()))
    parts.append(("Step 1: feature ablation (ablation_features.csv, ablation_features_paired.csv, holm_adjusted.csv)", t_ablation()))
    s1, s2, s3 = t_single()
    parts += [("Single-feature AUC, requested features (single_feature_auc.csv)", s1),
              ("Single-feature AUC, ten highest of the other features (single_feature_auc.csv)", s2),
              ("Single-feature AUC, kinematic inputs (single_feature_auc.csv)", s3)]
    pa, pc = t_perm()
    parts += [("Step 2 (a, b): permutation nulls with scores fixed (permutation_cluster.csv, holm_adjusted.csv)", pa),
              ("Step 2 (c): retraining trial-block permutation (permutation_cluster.csv)", pc)]
    parts.append(("Step 3: Holm–Bonferroni (holm_adjusted.csv)", t_holm()))
    l1, l2, used = t_levels()
    parts += [("Step 4: discrimination levels (discrimination_levels.csv)", l1),
              (f"Step 4: within-trial AUC per trial (within_trial_per_trial.csv); trials used: {used}; column header = trial [zone labels 1-2-3]", l2)]
    parts.append(("Step 5: calibration (calibration.csv)", t_cal()))
    try:
        parts.append(("Permutation importance, feature family × modality (importance.csv)", t_importance()))
    except FileNotFoundError:
        pass
    parts.append(("Run times (run_times.csv; CPU seconds summed over seeds, jobs ran in parallel)", t_runtimes()))
    (REV / "summary_tables.md").write_text("\n\n".join(f"### {t}\n\n{b}" for t, b in parts) + "\n", encoding="utf-8")
    print("summary_tables.md written")
