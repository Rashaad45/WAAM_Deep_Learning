"""Unit tests for the post-audit pipeline (run:  python -m pytest tests -q)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import CONFIG  # noqa: E402
from signal_features import (harmonic_mask, zone_spectrum_peak, chatter_label,  # noqa: E402
                             WindowFeatureExtractor)
from prepare_dataset import zone_spans, window_starts, measure_spindle_hz  # noqa: E402
from graph_data import kinematic_adjacency, apply_variant  # noqa: E402
from models import build_model, window_mean, DEEP_MODELS, count_parameters  # noqa: E402
from run_experiments import loto_folds, bce_loss, Standardiser, zone_feature_matrix  # noqa: E402

FS = 25_000.0
F0 = 1031.0 / 60.0          # spindle frequency (Hz)
RNG = np.random.default_rng(0)


def synthetic_zone(tone_hz: float, seconds: float = 4.0, amp: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    noise = RNG.normal(0, 1, (3, t.size))
    return noise + amp * np.sin(2 * np.pi * tone_hz * t)[None, :]


# ---------------------------------------------------------------- label rule
def test_harmonic_mask():
    assert harmonic_mask(np.array([200 * F0]), F0, 2.0)[0]
    assert not harmonic_mask(np.array([200 * F0 + 3.0]), F0, 2.0)[0]


@pytest.mark.parametrize("offset_hz, expected", [(0.0, 0), (3.0, 1)])
def test_label_rule_peak_on_and_off_harmonic(offset_hz, expected):
    """A strong peak exactly on a spindle harmonic is forced (stable); 3 Hz off it is chatter (unstable)."""
    tone = 200 * F0 + offset_hz                         # ~3437 Hz, inside the chatter band
    peak, prom = zone_spectrum_peak(synthetic_zone(tone), FS, CONFIG.signal.chatter_band_hz,
                                    CONFIG.labels.label_nperseg)
    assert abs(peak - tone) < 1.0 and prom > 25
    assert chatter_label(peak, prom, F0, CONFIG.signal.harmonic_tolerance_hz, 20.0) == expected


def test_label_rule_threshold():
    midway = 174.5 * F0                                  # exactly between two spindle harmonics
    assert chatter_label(midway, 19.9, F0, 2.0, 20.0) == 0
    assert chatter_label(midway, 20.0, F0, 2.0, 20.0) == 1
    assert chatter_label(175 * F0 + 1.9, 30.0, F0, 2.0, 20.0) == 0   # within tolerance -> forced


def test_chatter_indicator_uses_spindle_harmonics():
    fx = WindowFeatureExtractor(CONFIG.signal)
    t = np.arange(int(0.5 * FS)) / FS
    on = np.sin(2 * np.pi * 200 * F0 * t)
    off = np.sin(2 * np.pi * (200 * F0 + F0 / 2) * t)
    assert fx.chatter_indicator(on, FS, F0) < 0.1
    assert fx.chatter_indicator(off, FS, F0) > 0.9


def test_measured_spindle_frequency():
    t = np.arange(int(12 * FS)) / FS
    x = sum(np.sin(2 * np.pi * n * F0 * t + n) / n for n in range(1, 40))
    force = np.stack([x, 0.5 * x, 0.2 * x], axis=1) + RNG.normal(0, 0.05, (t.size, 3))
    f0, contrast = measure_spindle_hz(force, FS)
    assert abs(f0 - F0) < 0.01 and contrast > 3


# ---------------------------------------------------------------- zones
def test_zone_segmentation():
    start, end = 2.0, 23.4
    spans = zone_spans(start, end, CONFIG)
    excl = CONFIG.process.end_exclusion_s
    assert len(spans) == 3
    assert spans[0][0] == pytest.approx(start + excl) and spans[-1][1] == pytest.approx(end - excl)
    for (a, b), (c, d) in zip(spans[:-1], spans[1:]):
        assert b == pytest.approx(c)                     # contiguous, no overlap
    third = (end - start) / 3
    assert spans[1][1] - spans[1][0] == pytest.approx(third)
    w = CONFIG.signal.window_ms / 1000
    for span in spans:
        st, ctx = window_starts(span, CONFIG)
        assert len(st) == CONFIG.signal.windows_per_zone
        assert st.min() >= span[0] - 1e-9 and st.max() + w <= span[1] + 1e-9
        assert ctx.min() >= span[0] - 1e-9 and ctx.max() + CONFIG.signal.context_s <= span[1] + 1e-9


# ---------------------------------------------------------------- graphs / splits
def dummy_graph(trial_id: str, n=3, t=4, f=62, length=64, source="WAAM_LAB", y=None):
    return {"x": torch.randn(n, t, f), "force_raw": torch.randn(n, t, 3, length),
            "acceleration_raw": torch.randn(n, t, 3, length), "adj": kinematic_adjacency(n),
            "y": torch.tensor(y if y is not None else [0.0, 1.0, 0.0]), "trial_id": trial_id, "source": source}


def test_kinematic_adjacency_only():
    a = kinematic_adjacency(3)
    assert torch.equal(a, torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]))
    assert torch.equal(apply_variant(dummy_graph("t"), "no_edges")["adj"], torch.zeros(3, 3))


def test_loto_split_has_no_overlap():
    graphs = [dummy_graph(f"trial_{i:02d}") for i in range(6)]
    folds = loto_folds(graphs)
    assert sorted(k for _, k in folds) == list(range(6))
    for train, k in folds:
        assert k not in train
        assert graphs[k]["trial_id"] not in {graphs[i]["trial_id"] for i in train}


def test_external_samples_cannot_enter_waam_splits():
    graphs = [dummy_graph("trial_01"), dummy_graph("turning_5.08cm_320rpm", source="EXTERNAL_TURNING")]
    with pytest.raises(AssertionError):
        loto_folds(graphs)


# ---------------------------------------------------------------- aggregation
@pytest.mark.parametrize("kind", ["fusion_mlp", "cnn1d", "small_gnn"])
def test_window_mean_aggregation_is_order_invariant(kind):
    torch.manual_seed(0)
    g = dummy_graph("t", length=1250)
    m = build_model(kind, g["x"].shape[-1], CONFIG).eval()
    perm = torch.randperm(g["x"].shape[1])
    gp = {**g, "x": g["x"][:, perm], "force_raw": g["force_raw"][:, perm],
          "acceleration_raw": g["acceleration_raw"][:, perm]}
    with torch.no_grad():
        assert torch.allclose(m(g), m(gp), atol=1e-5)


@pytest.mark.parametrize("kind", DEEP_MODELS)
def test_every_model_uses_window_mean(kind, monkeypatch):
    import models
    calls = []
    real = models.window_mean
    monkeypatch.setattr(models, "window_mean", lambda h: calls.append(h.shape) or real(h))
    g = dummy_graph("t", length=1250)
    out = build_model(kind, g["x"].shape[-1], CONFIG).eval()(g)
    assert calls and out.shape == (3,)                    # one logit per zone, no Ra head


def test_classical_aggregation_is_window_mean():
    g = dummy_graph("t")
    assert np.allclose(zone_feature_matrix(g), g["x"].mean(1).numpy())
    assert torch.allclose(window_mean(g["x"]), g["x"].mean(1))


# ---------------------------------------------------------------- feature plumbing (replaces the material flag)
def test_appended_feature_survives_standardisation_and_reaches_network():
    graphs = []
    for i in range(4):
        g = dummy_graph(f"trial_{i:02d}")
        flag = torch.full((3, 4, 1), float(i % 2))
        g["x"] = torch.cat([g["x"], flag], -1)
        graphs.append(g)
    s = Standardiser(graphs)
    assert s.std[-1] > 0.1
    g = s(graphs[1])
    assert torch.allclose(g["x"][..., -1], g["x"][0, 0, -1].expand(3, 4))
    m = build_model("small_gnn", g["x"].shape[-1], CONFIG)
    x = g["x"].clone().requires_grad_(True)
    m({**g, "x": x}).sum().backward()
    assert x.grad[..., -1].abs().sum() > 0


# ---------------------------------------------------------------- loss
def test_loss_is_weighted_bce_only():
    logits, y, pw = torch.tensor([0.3, -1.2, 2.0]), torch.tensor([1.0, 0.0, 1.0]), torch.tensor(2.5)
    p = torch.sigmoid(logits)
    manual = -(pw * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    assert torch.allclose(bce_loss(logits, y, pw), manual, atol=1e-6)


def test_small_gnn_capacity():
    assert count_parameters(build_model("small_gnn", 62, CONFIG)) <= 5000
