"""Read-only audit analysis of existing outputs (no pipeline changes)."""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = sys.argv[1]
pred = pd.read_csv(f"{ROOT}/outputs/predictions_oof.csv")
abl = pd.read_csv(f"{ROOT}/outputs/predictions_ablation.csv")
labels = pd.read_csv(f"{ROOT}/data/zone_labels.csv")
rng = np.random.default_rng(0)

print("=== positives per trial ===")
pos = labels.groupby("trial_id").chatter_label.sum()
print(pos[pos > 0].to_dict(), "total", int(pos.sum()), "of", len(labels))

# ---- 3(a) label alignment: y_class in predictions vs zone_labels -----------------
m = pred.merge(labels[["trial_id", "zone_id", "chatter_label", "Ra_um"]], on=["trial_id", "zone_id"])
print("\n=== 3a label alignment: mismatched y_class rows:",
      int((m.y_class != m.chatter_label).sum()), "/", len(m),
      " mismatched y_reg rows:", int((~np.isclose(m.y_reg, m.Ra_um)).sum()))
print("fold -> trial uniqueness:", pred.groupby("fold").trial_id.nunique().max(), "trial(s) per fold")


def zones(frame):
    return frame.groupby(["trial_id", "material_type", "zone_id"], as_index=False).agg(
        y_class=("y_class", "first"), y_reg=("y_reg", "first"),
        probability=("probability", "mean"), ra_pred=("ra_pred", "mean"))


def perm_null(y, p, n=5000):
    return np.array([roc_auc_score(rng.permutation(y), p) for _ in range(n)])


print("\n=== 3: AUC pooled / per-fold / within-trial-rank / permutation null ===")
rows = []
for model, f in pred[pred.probability.notna()].groupby("model"):
    z = zones(f)
    y, p = z.y_class.to_numpy(int), z.probability.to_numpy()
    auc = roc_auc_score(y, p)
    auc_inv = roc_auc_score(y, 1 - p)
    null = perm_null(y, p)
    # per-fold AUC: each fold = 1 trial of 6 zones; defined only if both classes present
    per_fold = {}
    for t, g in z.groupby("trial_id"):
        if g.y_class.nunique() == 2:
            per_fold[t] = roc_auc_score(g.y_class, g.probability)
    # mean predicted prob of held-out trial vs training prevalence of that fold
    z["train_prev"] = z.trial_id.map(lambda t: (pos.sum() - pos[t]) / (len(labels) - 6))
    r_prior = np.corrcoef(z.groupby("trial_id").train_prev.first(),
                          z.groupby("trial_id").probability.mean())[0, 1]
    # within-material AUC
    w = z[z.material_type == "WAAM"]
    auc_waam = roc_auc_score(w.y_class, w.probability)
    # within-trial standardised scores (removes per-fold calibration offset)
    z["p_c"] = z.probability - z.groupby("trial_id").probability.transform("mean")
    auc_centered = roc_auc_score(y, z.p_c)
    rows.append(dict(model=model, auc_pooled=auc, auc_if_inverted=auc_inv,
                     null_mean=null.mean(), null_q025=np.quantile(null, .025), null_q975=np.quantile(null, .975),
                     perm_p_two_sided=np.mean(np.abs(null - 0.5) >= abs(auc - 0.5)),
                     auc_waam_only=auc_waam, auc_within_trial_centred=auc_centered,
                     r_trainprev_vs_mean_prob=r_prior,
                     per_fold=";".join(f"{k[-2:]}:{v:.2f}" for k, v in per_fold.items())))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.round(3).to_string(index=False))

print("\n=== mean predicted probability per held-out trial (deep models, seeds averaged) ===")
deep = pred[pred.model.isin(["cnn1d", "bilstm", "fusion_mlp", "gcn", "gat", "Logistic regression"])]
tab = deep.groupby(["trial_id", "model"]).probability.mean().unstack().round(3)
tab["n_pos"] = pos
print(tab.to_string())

print("\n=== 7: epochs trained (early stopping) ===")
e = pred[pred.epochs_trained.notna()].groupby(["model", "fold", "seed"]).epochs_trained.first()
print(e.groupby("model").describe().round(1).to_string())

print("\n=== 2/9: Ra regression diagnostics ===")
for (model, var), f in pd.concat([pred.assign(variant=pred.variant.fillna("full")), abl]).groupby(["model", "variant"]):
    if f.ra_pred.isna().all() or (model == "gat" and var == "full" and f is None):
        continue
    z = zones(f)
    err = z.ra_pred - z.y_reg
    by_mat = z.assign(se=err ** 2).groupby("material_type").se.mean() ** .5
    print(f"{model:<35}{var:<32} RMSE={np.sqrt((err**2).mean()):.3f}  "
          f"RMSE_WAAM={by_mat.get('WAAM', np.nan):.3f} RMSE_WROUGHT={by_mat.get('WROUGHT', np.nan):.3f}  "
          f"pred mean WAAM={z[z.material_type=='WAAM'].ra_pred.mean():.3f} WROUGHT={z[z.material_type=='WROUGHT'].ra_pred.mean():.3f}")

# Fair reference: predict the training mean of the held-out trial's own material
print("\n=== 9: 'predict training mean' vs 'predict training mean of same material' ===")
lab = labels.copy()
ref_all, ref_mat = [], []
for t in lab.trial_id.unique():
    tr = lab[lab.trial_id != t]; te = lab[lab.trial_id == t]
    ref_all += list(te.Ra_um - tr.Ra_um.mean())
    ref_mat += list(te.Ra_um - tr[tr.material_type == te.material_type.iloc[0]].Ra_um.mean())
print("predict-global-train-mean RMSE", np.sqrt(np.mean(np.square(ref_all))).round(4),
      " predict-material-train-mean RMSE", np.sqrt(np.mean(np.square(ref_mat))).round(4))
