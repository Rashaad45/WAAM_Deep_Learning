"""Read-only audit checks that import the pipeline code but change nothing on disk."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from config import CONFIG  # noqa: E402
from graph_data import TrialGraphBuilder, AblationVariant  # noqa: E402
from models import ModelFactory  # noqa: E402
from train_evaluate import FoldSplitter, Standardiser  # noqa: E402

graphs = TrialGraphBuilder(CONFIG).load_all()
print("graphs:", len(graphs), "x shape", tuple(graphs[0]["x"].shape),
      "force_raw", tuple(graphs[0]["force_raw"].shape))

# --- item 8: adjacency -----------------------------------------------------------
adj = TrialGraphBuilder(CONFIG).build_adjacency().numpy()
np.set_printoptions(precision=3, suppress=True)
print("\nalpha =", CONFIG.graph.thermal_diffusivity_m2_s)
print("A_kin=\n", adj[0]); print("A_th=\n", adj[1])
print("all trials share identical adj:", all(torch.equal(g["adj"], graphs[0]["adj"]) for g in graphs))
v = CONFIG.process.feed_rate_mm_min / 60 / 1000
print("diffusion length sqrt(4 alpha dt) for dt=3.75 s: %.1f mm" % (1000 * np.sqrt(4 * CONFIG.graph.thermal_diffusivity_m2_s * 3.75)))

# --- item 2: material-flag plumbing ---------------------------------------------
rng = np.random.default_rng(0)
folds = FoldSplitter.make_folds(graphs, "loto")
val_per_fold = [FoldSplitter.pick_validation([i for i in range(len(graphs)) if i not in t], graphs, rng) for t in folds]
flag_ok = []
for fold, test in enumerate(folds):
    tr = [i for i in range(len(graphs)) if i not in test and i not in val_per_fold[fold]]
    mod = [AblationVariant.apply(g, "material_flag") for g in graphs]
    sc = Standardiser([mod[i] for i in tr])
    te = sc.transform(mod[test[0]])
    col = te["x"][..., -1]
    flag_ok.append((graphs[test[0]]["trial_id"], graphs[test[0]]["material_type"], float(col.unique()[0]) if col.unique().numel() == 1 else "varies",
                    float(sc.x_std[-1])))
print("\nmaterial flag after standardisation (test trial, value, train std):")
for r in flag_ok: print("  ", r)

# --- item 7: validation sets ------------------------------------------------------
print("\nvalidation trials per fold and their positives:")
nopos = 0
for fold, (test, val) in enumerate(zip(folds, val_per_fold)):
    npos = int(sum(graphs[i]["y_class"].sum() for i in val))
    nopos += npos == 0
    print(f"  fold {fold:2d} test={graphs[test[0]]['trial_id']} val={[graphs[i]['trial_id'] for i in val]} val_pos={npos}")
print(f"  folds whose validation set has ZERO positives: {nopos}/{len(folds)}")

# --- item 10: parameter counts ------------------------------------------------------
F = graphs[0]["x"].shape[-1]
print("\nparameter counts:")
for k in ["cnn1d", "bilstm", "fusion_mlp", "gcn", "gat"]:
    m = ModelFactory.build(k, F, CONFIG)
    print(f"  {k:<11}{sum(p.numel() for p in m.parameters()):>8d}")
print("  logistic regression ~", F + 1, "(features + intercept)")

# --- item 11: chance of a random peak landing within +/-tol of a spindle harmonic ---
for rpm in (1000, 1400):
    f0 = rpm / 60
    for tol in (0.5, 1, 2):
        print(f"  P(random peak within +/-{tol} Hz of {f0:.2f} Hz harmonic) = {2 * tol / f0:.3f}")
