"""
config.py - every setting of the (post-audit) chatter-detection pipeline in one place.

Tags:  [paper] from the cutting plan of ref. [1] (doi:10.3390/ma19122653)
       [data]  measured from the raw files
       [choice] modelling decision, fixed BEFORE the post-audit rerun (see DECISION.md)

Scope after the audit (AUDIT.md section H and DECISION.md):
  * no thermal data, no thermal edges, no Ra head;
  * 3 zones per wall, 5 mm (one tool radius) removed at each end of the detected cut;
  * spindle-rotation harmonics everywhere;
  * fixed epoch count, no early stopping, window averaging for every model.
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "WAAM"
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "paper" / "figures"


@dataclass
class ProcessConfig:
    feed_rate_mm_min: float = 80.0           # [paper] constant feed
    tooth_count: int = 4                     # [paper] 4-flute carbide end mill
    tool_diameter_mm: float = 10.0           # [paper] D = 10 mm
    wall_path_length_mm: float = 30.0        # [paper] wall width crossed by the tool
    zone_count: int = 3                      # [choice] AUDIT item 12: zones ~ one tool diameter wide
    end_exclusion_mm: float = 5.0            # [choice] one tool radius removed at entry and exit

    @property
    def feed_mm_s(self) -> float:
        return self.feed_rate_mm_min / 60.0

    @property
    def end_exclusion_s(self) -> float:
        return self.end_exclusion_mm / self.feed_mm_s


@dataclass
class SignalConfig:
    force_fs_hz: float = 25_000.0            # [paper]+[data] Kistler 9257B
    acceleration_fs_hz: float = 25_000.0     # [data]
    window_ms: float = 100.0                 # [choice] analysis window
    windows_per_zone: int = 16               # [choice] (compute budget, fixed before any model result) T, spread evenly over the retained zone span
    context_s: float = 0.5                   # [choice] AUDIT item 5: chatter-indicator context (2 Hz bins)
    encoder_decimation: int = 2              # [choice] 25 kHz -> 12.5 kHz for the CNN encoders
    structural_modes_hz: tuple = (3500.0, 4300.0)   # [paper] dominant feed-force peaks of ref. [1]
    mode_half_width_hz: float = 150.0        # [choice]
    chatter_band_hz: tuple = (2500.0, 5500.0)        # [choice]
    harmonic_tolerance_hz: float = 2.0       # [choice] +/- around SPINDLE-ROTATION harmonics (label and feature)
    welch_segment: int = 1024                # [choice] window-level PSD for the mode energies
    wavelet: str = "cmor1.5-1.0"             # [choice]
    wavelet_scales: int = 24                 # [choice]


@dataclass
class CutDetectionConfig:
    envelope_bin_s: float = 0.01
    baseline_s: float = 1.0
    threshold_fraction: float = 0.3
    max_gap_s: float = 1.0
    min_cut_s: float = 15.0
    max_align_lag_s: float = 6.0
    alignment_ok_r: float = 0.7              # [choice] AUDIT item 13: r < 0.7 -> flagged, sensitivity run without


@dataclass
class LabelConfig:
    thresholds_db: tuple = (15.0, 20.0, 25.0)   # [choice] main results reported at all three
    primary_threshold_db: float = 20.0          # threshold used for ablations / sensitivity runs
    label_nperseg: int = 32768                  # 0.76 Hz bins


@dataclass
class TrainConfig:
    epochs: int = 30                         # [choice] FIXED a priori (compute budget), no early stopping (AUDIT item 7)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4               # decoupled from the loss (Adam weight decay)
    hidden_dim: int = 64
    small_hidden_dim: int = 16
    attention_heads: int = 4
    temporal_kernel: int = 3
    dropout: float = 0.25
    seeds: tuple = (42, 43, 44, 45, 46)
    bootstrap_draws: int = 2000
    permutation_draws: int = 2000
    torch_threads: int = 1                   # one thread per worker process


@dataclass
class PipelineConfig:
    process: ProcessConfig = field(default_factory=ProcessConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    cut: CutDetectionConfig = field(default_factory=CutDetectionConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = PipelineConfig()
