"""
rev_common.py - shared code for the faculty-revision analyses (revision_results/).

Reuses the ORIGINAL pipeline unchanged (src/config.py, src/graph_data.py, src/models.py,
src/run_experiments.py, src/evaluate.py). The only new thing is the feature-column selection
(Step 1), implemented by DROPPING columns of x before any standardisation.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from config import CONFIG  # noqa: E402
from graph_data import TrialGraphBuilder  # noqa: E402
from models import build_model  # noqa: E402
import run_experiments as RE  # noqa: E402
import evaluate as EV  # noqa: E402

REV = ROOT / "revision_results"
RAW = REV / "oof_raw"          # long-format predictions written by rev_train.py
FIG = REV / "figures"
for d in (REV, RAW, FIG):
    d.mkdir(parents=True, exist_ok=True)

THRESHOLD_DB = 15
STAT_SEED = 0                  # every bootstrap / permutation RNG: np.random.default_rng(0) per analysis unit
SEEDS = CONFIG.train.seeds     # (42, 43, 44, 45, 46)

# ---------------------------------------------------------------------------------------------
# Feature map (confirmed from src/prepare_dataset.py L203-205, src/signal_features.py L89-94,
# src/graph_data.py L49-50): x[..., ch*10 + f] for ch in (Fx, Fy, Fz, Acc-X, Acc-Y, Acc-Z),
# then 60 = spindle rpm_measured/1000, 61 = zone-centre position (fraction of the cut).
# ---------------------------------------------------------------------------------------------
CHANNELS = ["Fx", "Fy", "Fz", "AccX", "AccY", "AccZ"]
FEATS = ["rms", "p2p", "kurtosis", "crest", "mode1_energy_3.5kHz", "mode2_energy_4.3kHz",
         "chatter_indicator", "wavelet_entropy", "log_wavelet_energy", "band_centroid"]
FAMILY = {"rms": "statistical", "p2p": "statistical", "kurtosis": "statistical", "crest": "statistical",
          "mode1_energy_3.5kHz": "mode_energy", "mode2_energy_4.3kHz": "mode_energy",
          "chatter_indicator": "chatter_indicator", "wavelet_entropy": "wavelet", "log_wavelet_energy": "wavelet",
          "band_centroid": "band_centroid"}
FEATURE_NAMES = [f"{c}_{f}" for c in CHANNELS for f in FEATS] + ["spindle_speed", "zone_position"]
N_FEAT = len(FEATURE_NAMES)
assert N_FEAT == 62


def idx(ch=None, f=None):
    chs = range(6) if ch is None else ch
    fs = range(10) if f is None else f
    return [c * 10 + k for c in chs for k in fs]


FORCE, ACC, KIN = [0, 1, 2], [3, 4, 5], [60, 61]
STAT, SPEC, CI = [0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [6]
ALL = list(range(62))
CONFIGS = {
    "F0": (ALL, "All 62 features"),
    "F1": ([i for i in ALL if i not in idx(f=CI)], "All except chatter indicator (6 channels)"),
    "F2": ([i for i in ALL if i not in idx(FORCE, [4, 5, 6, 9])],
           "All except force chatter-band features (force CI, mode1, mode2, centroid)"),
    "F3": (idx(f=STAT) + KIN, "Statistical only (rms, p2p, kurtosis, crest; 6 ch) + speed + position"),
    "F4": (idx(f=SPEC), "Spectral only (mode1, mode2, CI, wavelet entropy, wavelet log-energy, centroid; 6 ch)"),
    "F5": (idx(f=[4, 5, 7, 8, 9]), "Spectral only without CI"),
    "F6": (idx(FORCE) + KIN, "Force channels only (30) + speed + position"),
    "F7": (idx(ACC) + KIN, "Acceleration channels only (30) + speed + position"),
    "F8": ([i for i in idx(ACC) if i % 10 != 6] + KIN, "Acceleration channels only without CI + speed + position"),
}
for k, (cols, _) in CONFIGS.items():
    assert sorted(set(cols)) == sorted(cols)

NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "svm": "SVM (RBF)", "cnn1d": "1D-CNN",
         "bilstm": "BiLSTM", "fusion_mlp": "Fusion MLP", "small_gnn": "Small GNN", "gcn": "ST-GCN", "gat": "ST-GAT"}
MODEL_ORDER = list(NAMES)
CLASSICAL = RE.CLASSICAL

_GRAPHS = None


def load_graphs(exclude=()):
    global _GRAPHS
    if _GRAPHS is None:
        _GRAPHS = TrialGraphBuilder(CONFIG, THRESHOLD_DB).load_all()
        for g in _GRAPHS:
            assert g["x"].shape[-1] == N_FEAT
    return [g for g in _GRAPHS if g["trial_id"] not in exclude]


def select(g: dict, cols) -> dict:
    s = dict(g)
    s["x"] = g["x"][..., list(cols)].contiguous()
    return s


def with_labels(graphs, ymap: dict):
    """Replace labels (permutation tests). ymap: trial_id -> label tensor."""
    out = []
    for g in graphs:
        s = dict(g); s["y"] = ymap[g["trial_id"]]; out.append(s)
    return out


def run_loto(model: str, graphs: list[dict], seed: int, cfg_id: str = "F0", record_losses=False):
    """Mirror of src/run_experiments.run_job for one seed; identical training code, but on the
    already column-selected graphs. Returns a list of prediction dicts."""
    torch.set_num_threads(CONFIG.train.torch_threads)
    preds, losses = [], []
    for fold, (train_idx, k) in enumerate(RE.loto_folds(graphs)):
        test = graphs[k]
        ytr = torch.cat([graphs[i]["y"] for i in train_idx])
        n_pos = int(ytr.sum())
        flag = n_pos == 0
        if flag:
            prob = np.full(len(test["y"]), float(ytr.mean()))
        elif model in CLASSICAL:
            xtr = np.concatenate([RE.zone_feature_matrix(graphs[i]) for i in train_idx])
            prob = RE.classical_model(model, seed).fit(xtr, ytr.numpy()).predict_proba(
                RE.zone_feature_matrix(test))[:, 1]
        else:
            RE.seed_everything(seed)
            scaler = RE.Standardiser([graphs[i] for i in train_idx])
            train = [scaler(graphs[i]) for i in train_idx]
            net = build_model(model, train[0]["x"].shape[-1], CONFIG)
            opt = torch.optim.Adam(net.parameters(), lr=CONFIG.train.learning_rate,
                                   weight_decay=CONFIG.train.weight_decay)
            pw = torch.tensor(float((ytr == 0).sum()) / n_pos)
            for epoch in range(CONFIG.train.epochs):
                net.train()
                total = 0.0
                for i in np.random.permutation(len(train)):
                    opt.zero_grad()
                    loss = RE.bce_loss(net(train[i]), train[i]["y"], pw)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    total += loss.item()
                if record_losses:
                    losses.append(dict(model=model, config=cfg_id, seed=seed, fold=fold, epoch=epoch + 1,
                                       train_loss=total / len(train)))
            net.eval()
            with torch.no_grad():
                prob = torch.sigmoid(net(scaler(test))).numpy()
        for z in range(len(test["y"])):
            preds.append(dict(model=model, config=cfg_id, seed=seed, fold=fold, trial_id=test["trial_id"],
                              zone_id=z, y=int(test["y"][z]), probability=float(prob[z]),
                              no_positive_training=flag))
    return (preds, losses) if record_losses else preds


def job(args):
    """(model, cfg_id, seed) -> (preds, losses, seconds). Worker entry point."""
    model, cfg_id, seed = args
    t0 = time.time()
    graphs = [select(g, CONFIGS[cfg_id][0]) for g in load_graphs()]
    p, l = run_loto(model, graphs, seed, cfg_id, record_losses=True)
    return p, l, time.time() - t0


# ---------------------------------------------------------------------------------------------
# Statistics helpers (reuse evaluate.py; fresh rng(0) per unit so every number is reproducible
# independently of the order in which analyses are run)
# ---------------------------------------------------------------------------------------------
auc = EV.auc
zones = EV.zones


def rng0():
    return np.random.default_rng(STAT_SEED)


def load_preds(model, cfg_id):
    f = RAW / f"{model}_{cfg_id}.csv"
    return pd.read_csv(f) if f.exists() else None


def zone_table(model, cfg_id):
    p = load_preds(model, cfg_id)
    return None if p is None else zones(p)


def original_zone_table(model, experiment="main_15db", variant="full"):
    """Seed-averaged zone scores from the manuscript's results/predictions.csv."""
    global _ORIG
    try:
        _ORIG
    except NameError:
        _ORIG = pd.read_csv(ROOT / "results" / "predictions.csv")
    f = _ORIG[(_ORIG.experiment == experiment) & (_ORIG.model == model) & (_ORIG.variant == variant)]
    return zones(f) if len(f) else None


def block_perm_labels(z: pd.DataFrame, rng):
    """Trial-block permutation: whole 3-zone label vectors are moved between trials."""
    trials = z.trial_id.unique()
    vec = {t: z[z.trial_id == t].sort_values("zone_id").y.to_numpy() for t in trials}
    perm = rng.permutation(len(trials))
    return {t: vec[trials[perm[i]]] for i, t in enumerate(trials)}


def fmt(x, d=2):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"
