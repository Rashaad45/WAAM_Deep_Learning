"""Time-frequency feature extraction used by the manuscript's node features."""
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats
from config import SignalConfig


def load_numeric_csv(path: str | Path, skip_unit_row: bool = False) -> tuple[np.ndarray, list[str]]:
    """Read numeric CSV exports, including the two-header-row accelerometer format."""
    frame = pd.read_csv(path, skiprows=[1] if skip_unit_row else None, on_bad_lines="skip")
    names = [str(name) for name in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"No numeric data found in {path}")
    frame = frame.interpolate(limit_direction="both").fillna(0.0)
    return frame.to_numpy(dtype=np.float64), names


def bandpass_filter(data: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError(f"Bandpass must lie below Nyquist ({fs / 2:g} Hz)")
    return signal.sosfiltfilt(signal.butter(order, [low_hz, high_hz], fs=fs, btype="bandpass", output="sos"), data, axis=0)


def fft_spectrum(data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("FFT requires at least two samples")
    return np.fft.rfftfreq(x.size, 1 / fs), np.abs(np.fft.rfft(x - x.mean())) / x.size


def welch_psd(data: np.ndarray, fs: float, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("PSD requires at least two samples")
    return signal.welch(x, fs=fs, nperseg=min(nperseg, x.size))


def _integrated_band(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq <= high)
    return float(np.trapz(psd[mask], freq[mask])) if mask.sum() > 1 else 0.0


def chatter_indicator(freq: np.ndarray, psd: np.ndarray, spindle_rpm: float, config: SignalConfig) -> float:
    """Non-synchronous energy fraction in the 2.5–5.5 kHz analysis band."""
    low, high = config.chatter_analysis_band_hz
    mask = (freq >= low) & (freq <= high)
    band_energy = float(np.trapz(psd[mask], freq[mask])) if mask.sum() > 1 else 0.0
    if band_energy <= 0:
        return 0.0
    tooth_hz = spindle_rpm * config.tooth_count / 60.0
    harmonics = np.arange(max(1, int(low / tooth_hz)), int(high / tooth_hz) + 1) * tooth_hz
    synchronous = np.zeros_like(freq, dtype=bool)
    resolution = float(np.median(np.diff(freq))) if len(freq) > 1 else config.mode_half_width_hz
    half_width = max(resolution, config.mode_half_width_hz / 4)
    for harmonic in harmonics:
        synchronous |= np.abs(freq - harmonic) <= half_width
    nonsync_mask = mask & ~synchronous
    nonsync = float(np.trapz(psd[nonsync_mask], freq[nonsync_mask])) if nonsync_mask.sum() > 1 else 0.0
    return nonsync / band_energy


def wavelet_energy_entropy(data: np.ndarray, fs: float, scales_count: int = 48, wavelet: str = "cmor1.5-1.0") -> tuple[np.ndarray, float]:
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 8:
        return np.zeros(scales_count, dtype=np.float32), 0.0
    scales = np.geomspace(1.0, min(128.0, x.size / 4), scales_count)
    coefficients, _ = pywt.cwt(x, scales, wavelet, sampling_period=1 / fs)
    energy = np.mean(np.abs(coefficients) ** 2, axis=1)
    prob = energy / max(float(energy.sum()), 1e-12)
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(max(2, len(prob))))
    return energy.astype(np.float32), entropy


def channel_features(data: np.ndarray, fs: float, spindle_rpm: float, config: SignalConfig) -> np.ndarray:
    """RMS, peak-to-peak, kurtosis, crest, mode-band powers, CI and CWT summary."""
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 8:
        raise ValueError("A signal window is too short for feature extraction")
    centered = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x)))
    crest = float(np.max(np.abs(centered)) / max(rms, 1e-12))
    freq, psd = welch_psd(centered, fs, config.welch_segment_size)
    mode_powers = [_integrated_band(freq, psd, mode - config.mode_half_width_hz, mode + config.mode_half_width_hz)
                   for mode in config.structural_modes_hz]
    chatter = chatter_indicator(freq, psd, spindle_rpm, config)
    wavelet_energy, entropy = wavelet_energy_entropy(centered, fs, config.wavelet_scales, config.wavelet)
    summary = np.array([rms, float(np.ptp(x)), float(stats.kurtosis(x, fisher=False, bias=False)) if x.size > 3 else 0.0,
                        crest, *mode_powers, chatter, entropy, float(np.log1p(wavelet_energy.sum()))], dtype=np.float32)
    return summary


def multichannel_features(data: np.ndarray, fs: float, spindle_rpm: float, config: SignalConfig) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.concatenate([channel_features(x[:, i], fs, spindle_rpm, config) for i in range(x.shape[1])])


def thermal_summary(temperature_c: np.ndarray, ambient_c: float, window_seconds: float) -> np.ndarray:
    """Peak, mean, spatial gradient and cumulative heat index from calibrated IR pixels."""
    image = np.asarray(temperature_c, dtype=float)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Thermal ROI must be a non-empty 2D temperature image")
    mean_temp = float(np.mean(image))
    gradient = float(np.mean(np.hypot(*np.gradient(image))))
    heat_index = float(np.mean(np.maximum(image - ambient_c, 0.0)) * window_seconds)
    return np.asarray([float(np.max(image)), mean_temp, gradient, heat_index], dtype=np.float32)
