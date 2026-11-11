"""
rev_importance.py - trial-aware permutation importance (LR, RF; F0) and RF TreeSHAP on out-of-fold models.

For every seed and LOTO fold the model is refit exactly as in the benchmark; the held-out trial is then
scored with one feature (or one feature group) permuted. Two schemes, both reported:
  * within_fold  : the feature is permuted among the 3 zones of the held-out trial (as specified;
                   this only removes WITHIN-trial information - between-trial differences survive);
  * across_oof   : the feature is permuted across all 48 held-out zones (each fold model scores its own
                   trial with values drawn from the pooled held-out set); removes all information.
Per-fold AUC is undefined for 8/16 single-class folds, so dAUC is computed on the POOLED out-of-fold,
seed-averaged scores: dAUC = AUC(original) - AUC(permuted), averaged over R = 20 permutation repeats.
Permutations come from np.random.default_rng(0) and are shared across seeds.

Writes importance.csv (individual + grouped) and shap_rf_oof.csv (48 x 62 seed-averaged SHAP values).
"""
import numpy as np
import pandas as pd

import rev_common as C

R = 20


def main():
    gs = C.load_graphs()
    trials = np.concatenate([[g["trial_id"]] * len(g["y"]) for g in gs])
    X = np.concatenate([C.RE.zone_feature_matrix(g) for g in gs])
    y = np.concatenate([g["y"].numpy() for g in gs]).astype(int)
    rows_of = [np.flatnonzero(trials == g["trial_id"]) for g in gs]
    n = len(y)
    rng = C.rng0()
    perm_within = np.stack([np.concatenate([r[rng.permutation(len(r))] for r in rows_of]) for _ in range(R)])
    perm_across = np.stack([rng.permutation(n) for _ in range(R)])

    groups = {}
    for name, j in zip(C.FEATURE_NAMES, range(62)):
        groups[name] = [j]
    for fam in ["statistical", "mode_energy", "chatter_indicator", "wavelet", "band_centroid"]:
        for mod, chs in (("force", C.FORCE), ("acc", C.ACC)):
            groups[f"[{mod}] {fam}"] = [c * 10 + k for c in chs for k, f in enumerate(C.FEATS) if C.FAMILY[f] == fam]
    gnames = list(groups)

    out = []
    shap_sum = np.zeros((n, 62))
    for model in ("logreg", "rf"):
        P = {s: np.zeros((len(C.SEEDS), len(gnames), R, n)) for s in ("within_fold", "across_oof")}
        base = np.zeros((len(C.SEEDS), n))
        for si, seed in enumerate(C.SEEDS):
            for k, rows in enumerate(rows_of):
                tr = np.setdiff1d(np.arange(n), rows)
                m = C.RE.classical_model(model, seed).fit(X[tr], y[tr])
                base[si, rows] = m.predict_proba(X[rows])[:, 1]
                for scheme, perms in (("within_fold", perm_within), ("across_oof", perm_across)):
                    big = []
                    for gi, g in enumerate(gnames):
                        for r in range(R):
                            xt = X[rows].copy()
                            xt[:, groups[g]] = X[perms[r][rows]][:, groups[g]]
                            big.append(xt)
                    pr = m.predict_proba(np.concatenate(big))[:, 1].reshape(len(gnames), R, len(rows))
                    P[scheme][si][:, :, rows] = pr
                if model == "rf":
                    import shap
                    sv = shap.TreeExplainer(m).shap_values(X[rows])
                    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
                    shap_sum[rows] += sv / len(C.SEEDS)
            print(model, "seed", seed, flush=True)
        a0 = C.auc(y, base.mean(0))
        for scheme in P:
            pm = P[scheme].mean(0)   # seed-average -> [groups, R, n]
            for gi, g in enumerate(gnames):
                d = np.array([a0 - C.auc(y, pm[gi, r]) for r in range(R)])
                out.append(dict(model=model, scheme=scheme, feature_or_group=g,
                                level="individual" if len(groups[g]) == 1 and not g.startswith("[") else "group",
                                n_columns=len(groups[g]), auc_original=a0, delta_auc_mean=d.mean(),
                                delta_auc_sd_over_repeats=d.std(ddof=1), repeats=R, seeds=len(C.SEEDS)))
    pd.DataFrame(out).to_csv(C.REV / "importance.csv", index=False)
    s = pd.DataFrame(shap_sum, columns=C.FEATURE_NAMES)
    s.insert(0, "trial_id", trials); s.insert(1, "y", y)
    s.to_csv(C.REV / "shap_rf_oof.csv", index=False)
    pd.DataFrame(X, columns=C.FEATURE_NAMES).assign(trial_id=trials, y=y).to_csv(
        C.REV / "zone_mean_features.csv", index=False)


if __name__ == "__main__":
    main()
