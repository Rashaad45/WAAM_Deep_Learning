"""Radiometric IR loading, interface ROIs, and the manuscript's small 2D CNN."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from config import VisionConfig
from utils_signal import thermal_summary


def load_radiometric_array(path: str | Path, config: VisionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load calibrated temperature frames from NPY/NPZ; never treat visible MP4 as temperature."""
    source = Path(path)
    if source.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("Thermal input must be calibrated temperature .npy/.npz; visible MP4 or proprietary IRGD is not radiometric data")
    if not config.thermal_is_radiometric:
        raise ValueError("Set thermal_is_radiometric=True only after confirming the exported array contains calibrated temperatures")
    if source.suffix.lower() == ".npz":
        with np.load(source) as archive:
            key = "temperature_c" if "temperature_c" in archive else "temperature"
            if key not in archive:
                raise ValueError(f"{source} needs a 'temperature_c' or 'temperature' array")
            cube = archive[key].astype(np.float32)
            times = archive["time_s"].astype(float) if "time_s" in archive else np.arange(len(cube), dtype=float)
    else:
        cube = np.load(source).astype(np.float32)
        times = np.arange(len(cube), dtype=float)
    if config.temperature_scale is not None:
        cube = cube * config.temperature_scale
    if config.temperature_offset_c is not None:
        cube = cube + config.temperature_offset_c
    if cube.ndim != 3:
        raise ValueError("Radiometric cube must have shape [frames, height, width]")
    if len(times) != len(cube) or not np.all(np.diff(times) >= 0):
        raise ValueError("Thermal timestamps must be monotonic and match the frame count")
    return cube, times


def track_interface_roi(frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Apply a calibrated fixed ROI or estimate the strongest thermal-gradient patch."""
    h, w = frame.shape[:2]
    if roi is None:
        grad_y, grad_x = np.gradient(frame.astype(np.float32))
        saliency = cv2.GaussianBlur(np.hypot(grad_x, grad_y), (5, 5), 0)
        _, _, _, point = cv2.minMaxLoc(saliency)
        rw, rh = max(8, w // 4), max(8, h // 4)
        x, y = max(0, min(point[0] - rw // 2, w - rw)), max(0, min(point[1] - rh // 2, h - rh))
    else:
        x, y, rw, rh = roi
    x, y = max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))
    rw, rh = max(1, min(int(rw), w - x)), max(1, min(int(rh), h - y))
    return frame[y:y + rh, x:x + rw], (x, y, rw, rh)


class ThermalCNN(nn.Module):
    """Small two-dimensional encoder shared across zones and time windows."""
    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(8, output_dim, 3, padding=1), nn.BatchNorm2d(output_dim), nn.ReLU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image).flatten(1)


@torch.inference_mode()
def encode_temperature_windows(cube: np.ndarray, times: np.ndarray, windows: np.ndarray,
                               zone_rois: list[tuple[int, int, int, int]], encoder: ThermalCNN,
                               config: VisionConfig, device: str = "cpu", ambient_c: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Return [zones, windows, (peak, mean, gradient, heat, CNN...)] and aligned temperature summaries."""
    encoder = encoder.to(device).eval()
    zone_features, thermal_images = [], []
    for roi in zone_rois:
        window_features, zone_images = [], []
        for start_s, end_s in windows:
            indices = np.flatnonzero((times >= start_s) & (times < end_s))
            if indices.size == 0:
                nearest = int(np.argmin(np.abs(times - (start_s + end_s) / 2)))
                indices = np.asarray([nearest])
            frames = []
            for index in indices:
                crop, _ = track_interface_roi(cube[index], roi or config.roi_xywh)
                frames.append(crop)
            image = np.mean(frames, axis=0).astype(np.float32)
            scaled = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(scaled).float()[None, None].to(device)
            cnn = encoder(tensor).squeeze(0).cpu().numpy()
            summary = thermal_summary(image, ambient_c, end_s - start_s)
            window_features.append(np.concatenate((summary, cnn)).astype(np.float32))
            zone_images.append(image)
        zone_features.append(window_features)
        thermal_images.append(zone_images)
    return np.asarray(zone_features, dtype=np.float32), np.asarray(thermal_images, dtype=object)
