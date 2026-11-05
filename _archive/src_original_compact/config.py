"""Single source of truth for every setting used by the WAAM ST-GNN pipeline.

Each value carries a short note on where it comes from:
  [paper]   - stated in the earlier study (Materials 2025, ref. [1] of the draft)
  [data]    - measured directly from the raw files in WAAM/
  [lit]     - taken from published literature / alloy datasheets
  [choice]  - a modelling decision made for this study (report it as such)

Change values here, never inside the other modules.
"""
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Folder layout (all paths are relative to the project root).
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "WAAM"                      # untouched raw exports
DATA_DIR = PROJECT_DIR / "data"                          # curated manifests and labels
CACHE_DIR = PROJECT_DIR / "data" / "processed"           # generated feature caches
OUTPUT_DIR = PROJECT_DIR / "outputs"                     # generated results
REPORT_DIR = PROJECT_DIR / "report"                      # manuscript draft + filled copy


@dataclass
class ProcessConfig:
    """Machining set-up of the experiments."""
    feed_rate_mm_min: float = 80.0           # [paper] Table 3
    tooth_count: int = 4                     # [paper] 4-flute WC end mill, D = 10 mm
    wall_path_length_mm: float = 30.0        # [paper] wall length after pre-machining
    zone_count: int = 6                      # [choice] 6 zones of 5 mm
    zone_length_mm: float = 5.0              # [choice]


@dataclass
class SignalConfig:
    """Signal acquisition and feature extraction."""
    force_fs_hz: float = 25_000.0            # [paper]+[data] Kistler 9257B, 25 kHz
    acceleration_fs_hz: float = 25_000.0     # [data] measured from the CSV time column (trials 4-21)
    window_ms: float = 100.0                 # [choice] 6.7 tooth periods at 1000 rpm, 9.3 at 1400 rpm
    overlap_fraction: float = 0.5            # [choice] 50 % overlap
    encoder_decimation: int = 2              # [choice] raw input to the CNN encoders at 12.5 kHz
                                             #          (Nyquist 6.25 kHz keeps both modes)
    structural_modes_hz: tuple = (3500.0, 4300.0)   # [paper] dominant feed-force peaks
    mode_half_width_hz: float = 150.0        # [choice] band half-width Δf around each mode
    chatter_band_hz: tuple = (2500.0, 5500.0)       # [choice] analysis band B of Eq. (7)
    harmonic_tolerance_hz: float = 2.0       # [choice] ±δ around spindle harmonics (labels)
    welch_segment: int = 1024                # [choice] Welch segment for window features
    wavelet: str = "cmor1.5-1.0"             # [choice] analytic Morlet
    wavelet_scales: int = 24                 # [choice] number of CWT scales J


@dataclass
class CutDetectionConfig:
    """Automatic location of the cut inside each force recording."""
    envelope_bin_s: float = 0.01             # [choice] 10 ms RMS bins
    baseline_s: float = 1.0                  # [choice] first second = spindle running, no cut
    threshold_fraction: float = 0.3          # [choice] onset when RMS > baseline + 30 % of plateau rise
    max_gap_s: float = 1.0                   # [choice] dips shorter than this do not end the cut
    min_cut_s: float = 15.0                  # [choice] reject bursts shorter than this
    max_align_lag_s: float = 6.0             # [choice] search range for force/accel alignment
    min_alignment_r: float = 0.3             # [choice] below this the alignment is flagged as weak


@dataclass
class LabelConfig:
    """Zone-level labels."""
    # Chatter rule (decided before training, see report Section 2.3):
    # a zone is UNSTABLE when, in the high-resolution PSD of the full zone
    # record (3.75 s, ~0.76 Hz resolution), the dominant peak inside the
    # chatter band is NOT a spindle harmonic AND rises at least
    # `peak_prominence_db` above the band's median PSD level.
    peak_prominence_db: float = 20.0         # [choice]
    label_nperseg: int = 32768               # [choice] PSD resolution for labelling
    # Material-level Ra means from the earlier study, used ONLY when no
    # zone-level measurement is supplied in data/ra_measurements.csv.
    ra_material_mean_um: dict = field(default_factory=lambda: {"WAAM": 1.22, "WROUGHT": 0.83})   # [paper]


@dataclass
class GraphConfig:
    """Spatio-temporal graph construction (Section 2.5 of the draft)."""
    # Thermal diffusivity of Al 5356 from datasheet values  [lit]
    # k = 116 W/m/K, rho = 2640 kg/m3, c = 900 J/kg/K  ->  alpha = k / (rho c)
    conductivity_w_mk: float = 116.0
    density_kg_m3: float = 2640.0
    specific_heat_j_kgk: float = 900.0
    thermal_edge_cutoff: float = 0.05        # [choice] weights below ε are set to 0 (ε = 0.05 in the draft)
    use_thermal_edges: bool = True           # heat-kernel edges need only geometry + alpha,
                                             # not thermal images, so they are available

    @property
    def thermal_diffusivity_m2_s(self) -> float:
        return self.conductivity_w_mk / (self.density_kg_m3 * self.specific_heat_j_kgk)


@dataclass
class TrainConfig:
    """Optimisation and evaluation."""
    epochs: int = 60
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    attention_heads: int = 4
    temporal_kernel: int = 3
    dropout: float = 0.25
    regression_weight: float = 1.0           # λ in Eq. (14); fixed-weight loss retained
    seeds: tuple = (42,)                     # use (42, 43, 44, 45, 46) for the final paper run
    bootstrap_draws: int = 2000
    torch_threads: int = 6


@dataclass
class PipelineConfig:
    process: ProcessConfig = field(default_factory=ProcessConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    cut: CutDetectionConfig = field(default_factory=CutDetectionConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def zone_duration_s(self) -> float:
        """Time the tool needs to cross one zone (5 mm at 80 mm/min = 3.75 s)."""
        return self.process.zone_length_mm / (self.process.feed_rate_mm_min / 60.0)

    @property
    def hop_s(self) -> float:
        return self.signal.window_ms / 1000.0 * (1.0 - self.signal.overlap_fraction)

    @property
    def windows_per_zone(self) -> int:
        width = self.signal.window_ms / 1000.0
        return int((self.zone_duration_s - width) // self.hop_s) + 1


CONFIG = PipelineConfig()
