"""Explicit, paper-aligned settings for the thin-wall milling experiments.

Values left unresolved in the supplied manuscript remain unset and must be
provided from the acquisition protocol or material datasheet before training.
"""
from dataclasses import dataclass, field
from pathlib import Path
import torch

PIPELINE_DIR = Path(__file__).resolve().parent


@dataclass
class SignalConfig:
    force_sampling_rate_hz: float = 25_000.0
    acceleration_sampling_rate_hz: float | None = None
    thermal_frame_rate_hz: float | None = None
    window_ms: float | None = None
    overlap_fraction: float = 0.5
    welch_segment_size: int = 4096
    structural_modes_hz: tuple[float, float] = (3500.0, 4300.0)
    mode_half_width_hz: float = 150.0
    chatter_analysis_band_hz: tuple[float, float] = (2500.0, 5500.0)
    wavelet: str = "cmor1.5-1.0"
    wavelet_scales: int = 48
    tooth_count: int = 4


@dataclass
class VisionConfig:
    include_thermal: bool = False
    roi_xywh: tuple[int, int, int, int] | None = None
    thermal_is_radiometric: bool = False
    temperature_scale: float | None = None
    temperature_offset_c: float | None = None
    ambient_temperature_c: float | None = None
    encoder_channels: int = 16


@dataclass
class GraphConfig:
    zone_count: int = 6
    zone_length_mm: float = 5.0
    wall_path_length_mm: float = 30.0
    clamp_position_mm: float = 0.0
    feed_rate_mm_min: float = 80.0
    thermal_diffusivity_m2_s: float | None = None
    thermal_edge_cutoff: float = 0.01
    kinematic_edge_weight: float = 1.0
    thermal_edge_weight: float = 1.0
    include_thermal_edges: bool = False


@dataclass
class TrainConfig:
    outer_cv: str = "leave-one-trial-out"
    inner_validation_fraction: float = 0.15
    epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    attention_heads: int = 4
    dropout: float = 0.25
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    regression_loss: str = "masked-mse"
    regression_weight: float = 1.0


@dataclass
class PipelineConfig:
    data_root: Path = PIPELINE_DIR.parent / "WAAM"
    output_dir: Path = PIPELINE_DIR / "outputs"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    signal: SignalConfig = field(default_factory=SignalConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = PipelineConfig()
