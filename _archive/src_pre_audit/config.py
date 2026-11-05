"""
config.py
=========
ONE place for EVERY setting of the WAAM ST-GNN pipeline.

If you want to change a number (window length, learning rate, number of
seeds, ...) change it HERE and nowhere else. All other files import the
object `CONFIG` that is created at the bottom of this file.

Every value has a short note that says where it comes from:
  [paper]   - stated in the earlier study (Materials 2025, ref. [1] of the draft)
  [data]    - measured directly from the raw files in the WAAM/ folder
  [lit]     - taken from published literature / alloy datasheets
  [choice]  - a modelling decision made for this study (report it as such)

Python ideas used in this file
------------------------------
* @dataclass  : a decorator that turns a class into a simple "box of values".
                Python writes the __init__ method for us, so we only list the
                fields and their default values.
* field(default_factory=...) : needed when the default value is a dict or a
                list, so that every object gets its OWN copy of it.
* @property   : a method that is read like a normal attribute, e.g.
                CONFIG.zone_duration_s  (no brackets), but is calculated
                every time from other values.
"""
from dataclasses import dataclass, field   # tools to build simple "settings" classes
from pathlib import Path                    # Path objects make file paths work on every OS


# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------
# __file__ is the path of THIS file (.../waam_gnn_pipeline/src/config.py).
# .resolve() makes it an absolute path, .parent goes one folder up (src),
# and the second .parent goes up again to the project folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_DIR / "WAAM"                 # untouched raw exports from the lab
DATA_DIR = PROJECT_DIR / "data"                     # manifests and label tables
CACHE_DIR = PROJECT_DIR / "data" / "processed"      # feature files written by Step 1
OUTPUT_DIR = PROJECT_DIR / "outputs"                # results written by Step 2
REPORT_DIR = PROJECT_DIR / "report"                 # manuscript draft + filled copy


# ---------------------------------------------------------------------------
# 1. The machining experiment
# ---------------------------------------------------------------------------
@dataclass
class ProcessConfig:
    """How the thin walls were milled."""
    feed_rate_mm_min: float = 80.0           # [paper] Table 3: the tool moves 80 mm per minute
    tooth_count: int = 4                     # [paper] 4-flute carbide end mill, D = 10 mm
    wall_path_length_mm: float = 30.0        # [paper] length of the wall after pre-machining
    zone_count: int = 6                      # [choice] the wall is split into 6 zones ...
    zone_length_mm: float = 5.0              # [choice] ... of 5 mm each (6 x 5 = 30 mm)


# ---------------------------------------------------------------------------
# 2. Signals and the hand-crafted features
# ---------------------------------------------------------------------------
@dataclass
class SignalConfig:
    """Sampling rates and feature-extraction settings."""
    force_fs_hz: float = 25_000.0            # [paper]+[data] Kistler 9257B dynamometer, 25 000 samples/s
    acceleration_fs_hz: float = 25_000.0     # [data] read from the time column of the accelerometer CSV
    window_ms: float = 100.0                 # [choice] one analysis window = 100 ms
                                             #          (6.7 tooth periods at 1000 rpm, 9.3 at 1400 rpm)
    overlap_fraction: float = 0.5            # [choice] consecutive windows overlap by 50 %
    encoder_decimation: int = 2              # [choice] keep every 2nd sample for the CNN encoders
                                             #          (25 kHz -> 12.5 kHz, still sees both modes)
    structural_modes_hz: tuple = (3500.0, 4300.0)   # [paper] the two dominant feed-force peaks
    mode_half_width_hz: float = 150.0        # [choice] +/- 150 Hz band around each mode
    chatter_band_hz: tuple = (2500.0, 5500.0)       # [choice] frequency band where chatter is searched
    harmonic_tolerance_hz: float = 2.0       # [choice] a peak within +/- 2 Hz of a spindle harmonic
                                             #          counts as "forced vibration", not chatter
    welch_segment: int = 1024                # [choice] segment length of the Welch PSD (window features)
    wavelet: str = "cmor1.5-1.0"             # [choice] complex Morlet wavelet for the CWT
    wavelet_scales: int = 24                 # [choice] number of wavelet scales J


# ---------------------------------------------------------------------------
# 3. Finding the cut inside each 40 s force recording
# ---------------------------------------------------------------------------
@dataclass
class CutDetectionConfig:
    """Settings for locating the cut and aligning the two sensors in time."""
    envelope_bin_s: float = 0.01             # [choice] the RMS envelope uses 10 ms bins
    baseline_s: float = 1.0                  # [choice] first second = spindle running in air (no cut)
    threshold_fraction: float = 0.3          # [choice] cut starts when RMS > baseline + 30 % of the rise
    max_gap_s: float = 1.0                   # [choice] dips shorter than 1 s do not end the cut
    min_cut_s: float = 15.0                  # [choice] a "cut" shorter than 15 s is rejected
    max_align_lag_s: float = 6.0             # [choice] search +/- 6 s when aligning force and acceleration
    min_alignment_r: float = 0.3             # [choice] correlation below 0.3 = alignment flagged as WEAK


# ---------------------------------------------------------------------------
# 4. Zone labels (what the model has to predict)
# ---------------------------------------------------------------------------
@dataclass
class LabelConfig:
    """Rules for the chatter label and the Ra (roughness) target."""
    # Chatter rule (fixed BEFORE training, see report Section 2.3):
    # a zone is UNSTABLE when, in the high-resolution PSD of the full zone
    # record (3.75 s, ~0.76 Hz resolution), the highest peak inside the
    # chatter band is NOT a spindle harmonic AND rises at least
    # `peak_prominence_db` above the median PSD level of that band.
    peak_prominence_db: float = 20.0         # [choice]
    label_nperseg: int = 32768               # [choice] long Welch segment = fine frequency resolution

    # Ra means of the earlier study. They are used ONLY when no zone-level
    # measurement is given in data/ra_measurements.csv.
    ra_material_mean_um: dict = field(default_factory=lambda: {"WAAM": 1.22, "WROUGHT": 0.83})   # [paper]


# ---------------------------------------------------------------------------
# 5. Graph construction
# ---------------------------------------------------------------------------
@dataclass
class GraphConfig:
    """Settings for the graph edges (Section 2.5 of the draft)."""
    # Thermal properties of Al 5356 from the datasheet  [lit]
    conductivity_w_mk: float = 116.0         # k   (W / m / K)
    density_kg_m3: float = 2640.0            # rho (kg / m^3)
    specific_heat_j_kgk: float = 900.0       # c   (J / kg / K)
    thermal_edge_cutoff: float = 0.05        # [choice] edge weights below 0.05 are set to 0
    use_thermal_edges: bool = True           # heat-kernel edges only need geometry + alpha,
                                             # NOT thermal images, so they are available

    @property
    def thermal_diffusivity_m2_s(self) -> float:
        """alpha = k / (rho * c)   -> about 4.9e-5 m^2/s for Al 5356."""
        return self.conductivity_w_mk / (self.density_kg_m3 * self.specific_heat_j_kgk)


# ---------------------------------------------------------------------------
# 6. Neural-network training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Optimiser, network size and evaluation settings."""
    epochs: int = 60                         # maximum number of passes over the training trials
    patience: int = 10                       # stop early after 10 epochs without improvement
    learning_rate: float = 1e-3              # Adam step size
    weight_decay: float = 1e-4               # L2 penalty on the weights
    hidden_dim: int = 64                     # size of the hidden vector of every zone
    attention_heads: int = 4                 # number of heads in the graph-attention layer
    temporal_kernel: int = 3                 # kernel size of the temporal convolution
    dropout: float = 0.25                    # fraction of activations set to 0 while training
    regression_weight: float = 1.0           # lambda in Eq. (14): loss = BCE + lambda * MSE
    seeds: tuple = (42,)                     # use (42, 43, 44, 45, 46) for the final paper run
    bootstrap_draws: int = 2000              # resamples used for the 95 % confidence intervals
    torch_threads: int = 6                   # CPU threads PyTorch may use


# ---------------------------------------------------------------------------
# 7. Everything together
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Groups all the settings above into one object."""
    process: ProcessConfig = field(default_factory=ProcessConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    cut: CutDetectionConfig = field(default_factory=CutDetectionConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def zone_duration_s(self) -> float:
        """Time the tool needs to cross one zone: 5 mm at 80 mm/min = 3.75 s."""
        feed_mm_per_second = self.process.feed_rate_mm_min / 60.0
        return self.process.zone_length_mm / feed_mm_per_second

    @property
    def hop_s(self) -> float:
        """Time between the starts of two consecutive windows (50 ms here)."""
        window_s = self.signal.window_ms / 1000.0
        return window_s * (1.0 - self.signal.overlap_fraction)

    @property
    def windows_per_zone(self) -> int:
        """How many overlapping windows fit inside one zone (T in the paper)."""
        window_s = self.signal.window_ms / 1000.0
        # "//" is integer division: how many full hops fit after the first window.
        return int((self.zone_duration_s - window_s) // self.hop_s) + 1


# The one settings object that every other file imports:
#     from config import CONFIG
CONFIG = PipelineConfig()
