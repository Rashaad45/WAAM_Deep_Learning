"""
rev_analyse.py - Steps 1 (statistics), 2(a,b), 3, 4 and 5 from saved out-of-fold predictions.

    python revision_results/scripts/rev_analyse.py

Inputs : revision_results/oof_raw/<model>_<config>.csv (this run), results/predictions.csv (manuscript run,
         used for Table 3 models not retrained here; reproduction checked in reproduction_check.csv).
Outputs: ablation_features.csv, ablation_features_paired.csv, single_feature_auc.csv,
         permutation_cluster.csv (a, b; c is added by rev_perm_retrain.py), holm_adjusted.csv,
         discrimination_levels.csv, within_trial_per_trial.csv, calibration.csv, calibration_bins.csv,
         reproduction_check.csv.
Every bootstrap / permutation uses a fresh np.random.default_rng(0) per (analysis, model, config).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import rev_common as C

warnings.filterwarnings("ignore")
REV = C.REV
PERMS = 2000
DEEP_ABL = ["bilstm", "fusion_mlp", "gat"]


# ------------------------------------------------------------------------------------------------
def table3_zone(model):
    """F0 zone scores for Table-3 models: this run's retrain if present, else results/predictions.csv."""
    z = C.zone_table(model, "F0")
    return z if z is not None else C.original_zone_table(model)


def reproduction_check():
    rows = []
    orig = pd.read_csv(C.ROOT / "results" / "predictions.csv")
    for m in C.MODEL_ORDER:
        z = C.load_preds(m, "F0")
        o = orig[(orig.experiment == "main_15db") & (orig.model == m) & (orig.variant == "full")]
        o = o[(o.experiment == "main_15db") & (o.model == m) & (o.variant == "full")]
        if z is None:
            rows.append(dict(model=m, recomputed=False, max_abs_diff_seed_level=np.nan, auc_this_run=np.nan,
                             auc_manuscript=C.auc(*_yp(C.zones(o)))))
            continue
        mm = z.merge(o, on=["seed", "trial_id", "zone_id"], suffixes=("", "_o"))
        rows.append(dict(model=m, recomputed=True, n_rows_matched=len(mm),
                         max_abs_diff_seed_level=float((mm.probability - mm.probability_o).abs().max()),
                         auc_this_run=C.auc(*_yp(C.zones(z))), auc_manuscript=C.auc(*_yp(C.zones(o)))))
    pd.DataFrame(rows).to_csv(REV / "reproduction_check.csv", index=False)


def _yp(z):
    return z.y.to_numpy(int), z.probability.to_numpy()


# ------------------------------------------------------------------------------------------------
# Step 1
# ------------------------------------------------------------------------------------------------
def step1():
    rows, paired = [], []
    for m in C.MODEL_ORDER:
        for cfg, (cols, desc) in C.CONFIGS.items():
            z = C.zone_table(m, cfg)
            if z is None:
                continue
            s = C.EV.summarise(z, C.rng0())
            rows.append(dict(model=m, config=cfg, description=desc, n_features=len(cols), **s))
            if cfg != "F0":
                z0, src = C.zone_table(m, "F0"), "this run (oof_raw)"
                if z0 is None:   # F0 not retrained here (small GNN, ST-GCN): main-benchmark OOF predictions
                    z0, src = C.original_zone_table(m), "results/predictions.csv (main_15db, full)"
                if z0 is not None:
                    t = C.EV.paired_test(z, z0, C.rng0())   # delta = AUC(cfg) - AUC(F0)
                    paired.append(dict(model=m, config=cfg, auc_config=s["auc"],
                                       auc_F0=C.auc(*_yp(z0)), F0_source=src, **t))
    a = pd.DataFrame(rows)
    a = a.rename(columns={"null_auc_q025": "zone_perm_null_q025", "null_auc_q975": "zone_perm_null_q975",
                          "perm_p_auc": "zone_perm_p"})
    a.to_csv(REV / "ablation_features.csv", index=False)
    p = pd.DataFrame(paired).rename(columns={"perm_p": "paired_perm_p"})
    p.to_csv(REV / "ablation_features_paired.csv", index=False)

    # single-feature AUC: zone-mean (over the 16 windows) of one raw feature used as the score
    gs = C.load_graphs()
    X = np.concatenate([g["x"].mean(dim=1).numpy() for g in gs])
    y = np.concatenate([g["y"].numpy() for g in gs]).astype(int)
    trials = np.concatenate([[g["trial_id"]] * len(g["y"]) for g in gs])
    requested = set(C.idx(f=C.CI)) | set(C.idx(C.FORCE, [4, 5]))
    sf = []
    for j, name in enumerate(C.FEATURE_NAMES):
        rng = C.rng0()
        a_raw = C.auc(y, X[:, j])
        sign = 1 if a_raw >= 0.5 else -1
        sc = sign * X[:, j]
        boot = np.array([C.auc(y[i], sc[i]) for i in C.EV.bootstrap_indices(trials, rng)])
        boot = boot[np.isfinite(boot)]
        null = np.array([C.auc(rng.permutation(y), sc) for _ in range(PERMS)])
        a = C.auc(y, sc)
        sf.append(dict(feature_index=j, feature=name, requested=j in requested,
                       auc_raw_direction=a_raw, sign_chosen=("+" if sign > 0 else "-"), sign_chosen_post_hoc=True,
                       auc_oriented=a, auc_ci_low=np.quantile(boot, .025), auc_ci_high=np.quantile(boot, .975),
                       zone_perm_p_two_sided=(1 + np.sum(np.abs(null - .5) >= abs(a - .5))) / (PERMS + 1)))
    pd.DataFrame(sf).sort_values(["requested", "auc_oriented"], ascending=[False, False]).to_csv(
        REV / "single_feature_auc.csv", index=False)


# ------------------------------------------------------------------------------------------------
# Step 2 (a), (b) + the existing zone-level null, all with scores fixed
# ------------------------------------------------------------------------------------------------
def perm_nulls(z, rng_seed=C.STAT_SEED):
    z = z.sort_values(["trial_id", "zone_id"]).reset_index(drop=True)
    y, p = _yp(z)
    a = C.auc(y, p)
    trials = z.trial_id.to_numpy()
    groups = [np.flatnonzero(trials == t) for t in pd.unique(trials)]
    out = {}
    for kind in ("zone_level", "trial_block", "within_trial"):
        rng = np.random.default_rng(rng_seed)
        null = np.empty(PERMS)
        for b in range(PERMS):
            if kind == "zone_level":
                yy = rng.permutation(y)
            elif kind == "trial_block":
                perm = rng.permutation(len(groups))
                yy = np.concatenate([y[groups[k]] for k in perm])     # groups are contiguous & ordered
            else:
                yy = y.copy()
                for g in groups:
                    yy[g] = y[rng.permutation(g)]
            null[b] = C.auc(yy, p)
        out[kind] = dict(auc=a, null_mean=null.mean(), null_q025=np.quantile(null, .025),
                         null_q975=np.quantile(null, .975),
                         p_two_sided=(1 + np.sum(np.abs(null - .5) >= abs(a - .5) - 1e-12)) / (PERMS + 1),
                         p_one_sided=(1 + np.sum(null >= a - 1e-12)) / (PERMS + 1),
                         p_one_sided_vs_null_mean_note="within_trial null is not centred at 0.5" if kind == "within_trial" else "")
    return out


def step2():
    rows = []
    sets = [(m, "F0", table3_zone(m), "results/predictions.csv" if C.zone_table(m, "F0") is None else "this run")
            for m in C.MODEL_ORDER]
    sets += [(m, "F1", C.zone_table(m, "F1"), "this run") for m in ("logreg", "rf", "gat")]
    for m, cfg, z, src in sets:
        if z is None:
            continue
        for kind, r in perm_nulls(z).items():
            rows.append(dict(model=m, config=cfg, score_source=src, null=kind, retrained=False,
                             n_permutations=PERMS, rng_seed=C.STAT_SEED, **r))
    old = REV / "permutation_cluster.csv"
    df = pd.DataFrame(rows)
    if old.exists():   # keep step 2(c) rows written by rev_perm_retrain.py
        prev = pd.read_csv(old)
        df = pd.concat([df, prev[prev.retrained == True]], ignore_index=True)  # noqa: E712
    df.to_csv(old, index=False)
    return df


# ------------------------------------------------------------------------------------------------
# Step 3 Holm
# ------------------------------------------------------------------------------------------------
def holm(df, pcol, family):
    d = df.copy()
    d["p_holm"] = multipletests(d[pcol].to_numpy(), method="holm")[1]
    d["family"] = family
    d["family_size"] = len(d)
    return d.rename(columns={pcol: "p_raw"})


def step3(perm):
    out = []
    met = pd.read_csv(C.ROOT / "results" / "metrics.csv")
    t3 = met[(met.experiment == "main_15db") & (met.variant == "full")].set_index("model").reindex(C.MODEL_ORDER)
    d = pd.DataFrame(dict(item=t3.index, p=t3.perm_p_auc.to_numpy(),
                          p_source="results/metrics.csv perm_p_auc (manuscript Table 3, zone-level null)"))
    out.append(holm(d, "p", "(i) Table 3, zone-level permutation (manuscript p)"))
    for kind, lab in (("zone_level", "(i) Table 3, zone-level permutation (recomputed this run, rng 0)"),
                      ("trial_block", "(i) Table 3, trial-block permutation (this run)")):
        q = perm[(perm.config == "F0") & (perm.null == kind) & (perm.retrained == False)].set_index("model")  # noqa
        q = q.reindex(C.MODEL_ORDER)
        d = pd.DataFrame(dict(item=q.index, p=q.p_two_sided.to_numpy(),
                              p_source=f"permutation_cluster.csv {kind} p_two_sided"))
        out.append(holm(d, "p", lab))
    pt = pd.read_csv(C.ROOT / "results" / "paired_tests.csv")
    t4 = pt[pt.experiment_a == "main_15db"].copy()
    assert len(t4) == 18, len(t4)
    t4["item"] = t4.model_a + ":" + t4.variant_a + " − " + t4.model_b + ":" + t4.variant_b + " (" + t4.experiment_b + ")"
    d = pd.DataFrame(dict(item=t4["item"], p=t4.perm_p, delta_auc=t4.delta_auc,
                          p_source="results/paired_tests.csv perm_p (manuscript Table 4)"))
    out.append(holm(d, "p", "(ii) Table 4, 18 paired comparisons"))
    ab = pd.read_csv(REV / "ablation_features_paired.csv")
    for m, g in ab.groupby("model", sort=False):
        d = pd.DataFrame(dict(item=m + ": " + g.config + " − F0", p=g.paired_perm_p, delta_auc=g.delta_auc,
                              p_source="ablation_features_paired.csv paired_perm_p"))
        out.append(holm(d, "p", f"(iii) Step 1 ablation vs F0, {m}"))
    h = pd.concat(out, ignore_index=True)
    h["significant_raw_0.05"] = h.p_raw < 0.05
    h["significant_holm_0.05"] = h.p_holm < 0.05
    h.to_csv(REV / "holm_adjusted.csv", index=False)


# ------------------------------------------------------------------------------------------------
# Step 4 levels
# ------------------------------------------------------------------------------------------------
def step4():
    rows, per = [], []
    sets = [(m, "F0", table3_zone(m)) for m in C.MODEL_ORDER]
    sets += [(m, "F1", C.zone_table(m, "F1")) for m in ("logreg", "rf", "gat")]
    for m, cfg, z in sets:
        if z is None:
            continue
        y, p = _yp(z)
        w = []
        for t, g in z.groupby("trial_id"):
            a = C.auc(*_yp(g))
            if np.isfinite(a):
                w.append(a)
                per.append(dict(model=m, config=cfg, trial_id=t, n_unstable=int(g.y.sum()),
                                within_trial_auc=a, zone_scores=";".join(f"{v:.3f}" for v in g.probability),
                                zone_labels="".join(str(v) for v in g.y)))
        tr = z.groupby("trial_id").agg(score=("probability", "mean"), n_unstable=("y", "sum"))
        any_u = (tr.n_unstable >= 1).astype(int).to_numpy()
        rho, rho_p = spearmanr(tr.score, tr.n_unstable)
        rng = C.rng0()
        tb = np.array([C.auc(any_u[i], tr.score.to_numpy()[i])
                       for i in [rng.integers(0, len(tr), len(tr)) for _ in range(C.EV.DRAWS)]])
        tb = tb[np.isfinite(tb)]
        keep = ~z.trial_id.isin(["trial_04", "trial_06"])
        used = sorted(set(r["trial_id"] for r in per if r["model"] == m and r["config"] == cfg))
        rows.append(dict(model=m, config=cfg, pooled_auc=C.auc(y, p),
                         within_trial_auc_mean=np.mean(w) if w else np.nan, n_trials_both_classes=len(w),
                         trials_both_classes=";".join(used),
                         trial_level_auc_any_unstable=C.auc(any_u, tr.score.to_numpy()),
                         trial_level_auc_ci_low=np.quantile(tb, .025), trial_level_auc_ci_high=np.quantile(tb, .975),
                         n_trials_any_unstable=int(any_u.sum()), n_trials_none=int((1 - any_u).sum()),
                         spearman_rho_score_vs_n_unstable=rho, spearman_p=rho_p,
                         pooled_auc_without_04_06=C.auc(*_yp(z[keep])),
                         note_without_04_06="post hoc removal from the same OOF predictions (no retraining)"))
    pd.DataFrame(rows).to_csv(REV / "discrimination_levels.csv", index=False)
    pd.DataFrame(per).to_csv(REV / "within_trial_per_trial.csv", index=False)


# ------------------------------------------------------------------------------------------------
# Step 5 calibration
# ------------------------------------------------------------------------------------------------
def equal_count_bins(p, k=5):
    order = np.argsort(p, kind="mergesort")
    return np.array_split(order, k)


def ece(y, p, k=5):
    return float(sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in equal_count_bins(p, k)))


PROB_NOTE = {"logreg": "LogisticRegression predict_proba (class_weight=balanced; not Platt-scaled)",
             "rf": "RandomForest vote fraction (class_weight=balanced; not calibrated)",
             "svm": "SVC(probability=True): libsvm internal Platt scaling with 5-fold CV on the training trials",
             "bilstm": "sigmoid of logit, trained with class-weighted BCE (not recalibrated)",
             "fusion_mlp": "sigmoid of logit, trained with class-weighted BCE (not recalibrated)",
             "gat": "sigmoid of logit, trained with class-weighted BCE (not recalibrated)"}


def step5():
    rows, bins = [], []
    sets = [(m, "F0", table3_zone(m)) for m in ("logreg", "rf", "svm", "bilstm", "fusion_mlp", "gat")]
    sets += [(m, "F1", C.zone_table(m, "F1")) for m in ("logreg", "rf")]
    for m, cfg, z in sets:
        if z is None:
            continue
        z = z.sort_values(["trial_id", "zone_id"]).reset_index(drop=True)
        y, p = _yp(z)
        prev = y.mean()
        brier = float(np.mean((p - y) ** 2))
        bref = float(np.mean((prev - y) ** 2))
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        try:
            fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
            a, b = fit.params
            ci = np.asarray(fit.conf_int())
            citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
            citl_v = float(citl.params[0])
            err = ""
        except Exception as e:  # pragma: no cover
            a = b = citl_v = np.nan; ci = np.full((2, 2), np.nan); err = str(e)
        rows.append(dict(model=m, config=cfg, probability_source=PROB_NOTE[m], n=len(y), prevalence=prev,
                         brier=brier, brier_constant_prevalence=bref, brier_skill_score=1 - brier / bref,
                         ece_5_equal_count_bins=ece(y, p), mean_predicted=float(p.mean()),
                         calibration_intercept=a, calibration_slope=b,
                         slope_ci_low=ci[1, 0], slope_ci_high=ci[1, 1],
                         calibration_in_the_large=citl_v, fit_error=err))
        rng = C.rng0()
        trials = z.trial_id.to_numpy()
        bs = []
        for idx in C.EV.bootstrap_indices(trials, rng):
            bb = equal_count_bins(p[idx])
            bs.append([(p[idx][q].mean(), y[idx][q].mean()) for q in bb])
        bs = np.array(bs)   # [draws, 5, 2]
        for k, q in enumerate(equal_count_bins(p)):
            bins.append(dict(model=m, config=cfg, bin=k + 1, n=len(q), mean_predicted=p[q].mean(),
                             observed_frequency=y[q].mean(),
                             boot_pred_q025=np.quantile(bs[:, k, 0], .025), boot_pred_q975=np.quantile(bs[:, k, 0], .975),
                             boot_obs_q025=np.quantile(bs[:, k, 1], .025), boot_obs_q975=np.quantile(bs[:, k, 1], .975)))
    pd.DataFrame(rows).to_csv(REV / "calibration.csv", index=False)
    pd.DataFrame(bins).to_csv(REV / "calibration_bins.csv", index=False)


if __name__ == "__main__":
    import sys
    steps = sys.argv[1:] or ["repro", "1", "2", "3", "4", "5"]
    if "repro" in steps:
        reproduction_check(); print("reproduction_check done")
    if "1" in steps:
        step1(); print("step1 done")
    perm = None
    if "2" in steps:
        perm = step2(); print("step2 done")
    if "3" in steps:
        step3(perm if perm is not None else pd.read_csv(REV / "permutation_cluster.csv")); print("step3 done")
    if "4" in steps:
        step4(); print("step4 done")
    if "5" in steps:
        step5(); print("step5 done")
