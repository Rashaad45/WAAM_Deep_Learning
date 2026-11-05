"""
signal_features.py - hand-crafted window features and the zone chatter label.

Both use the SAME definition of "forced vibration": a frequency within +/- tol Hz of a
SPINDLE-ROTATION harmonic n * (rpm / 60)  (AUDIT item 5). With cutter runout a 4-flute tool
forces vibration at every spindle harmonic, of which tooth-passing harmonics are a subset.

Window features (10 per channel; index = position in FEATURE_NAMES):
  0 rms, 1 p2p, 2 kurtosis, 3 crest                 - 100 ms window
  4 mode1_energy, 5 mode2_energy (log, +/-150 Hz)   - 100 ms window, 1024-pt Welch
  6 chatter_indicator                               - 0.5 s context centred on the window
                                                      (2 Hz bins, spindle-harmonic mask)
  7 wavelet_entropy, 8 log_wavelet_energy           - 100 ms window
  9 band_centroid_khz                               - 100 ms window
"""
import numpy as np
import pywt
from scipy import signal, stats

from config import SignalConfig

FEATURES_PER_CHANNEL = 10
FEATURE_NAMES = ["rms", "p2p", "kurtosis", "crest", "mode1_energy", "mode2_energy",
                 "chatter_indicator", "wavelet_entropy", "log_wavelet_energy", "band_centroid"]


def harmonic_mask(freq: np.ndarray, base_hz: float, tolerance_hz: float) -> np.ndarray:
    """True where freq lies within +/- tolerance_hz of a positive integer multiple of base_hz."""
    freq = np.asarray(freq, dtype=float)
    n = np.maximum(np.round(freq / base_hz), 1.0)
    return np.abs(freq - n * base_hz) <= tolerance_hz


def non_synchronous_fraction(freq, psd, band, spindle_hz, tolerance_hz) -> float:
    """Share of PSD power in `band` that is NOT within tolerance of a spindle harmonic."""
    in_band = (freq >= band[0]) & (freq <= band[1])
    total = float(psd[in_band].sum())
    if total <= 0:
        return 0.0
    sync = harmonic_mask(freq, spindle_hz, tolerance_hz)
    return float(psd[in_band & ~sync].sum() / total)


class WindowFeatureExtractor:
    def __init__(self, settings: SignalConfig):
        self.s = settings

    @staticmethod
    def band_energy(freq, psd, low, high) -> float:
        inside = (freq >= low) & (freq <= high)
        if inside.sum() <= 1:
            return 0.0
        return float(np.trapezoid(psd[inside], freq[inside]))

    def chatter_indicator(self, context: np.ndarray, fs: float, spindle_hz: float) -> float:
        """Eq. (CI): non-synchronous share of chatter-band power on the 0.5 s context (one PSD segment)."""
        c = context - context.mean()
        freq, psd = signal.periodogram(c, fs=fs, window="hann")
        return non_synchronous_fraction(freq, psd, self.s.chatter_band_hz, spindle_hz, self.s.harmonic_tolerance_hz)

    def wavelet_features(self, x: np.ndarray, fs: float) -> tuple[float, float]:
        cf = pywt.central_frequency(self.s.wavelet)
        target = np.geomspace(500.0, fs / 2.2, self.s.wavelet_scales)
        coef, _ = pywt.cwt(x, cf * fs / target, self.s.wavelet, sampling_period=1.0 / fs, method="fft")
        e = np.mean(np.abs(coef) ** 2, axis=1)
        p = e / max(float(e.sum()), 1e-20)
        return float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(p))), float(np.log1p(e.sum()))

    def band_centroid_khz(self, x: np.ndarray, fs: float) -> float:
        f, _, z = signal.stft(x, fs=fs, nperseg=256)
        lo, hi = self.s.chatter_band_hz
        m = (f >= lo) & (f <= hi)
        pw = np.abs(z[m]) ** 2
        return float(np.sum(f[m, None] * pw) / max(pw.sum(), 1e-20)) / 1000.0

    def channel_features(self, x: np.ndarray, context: np.ndarray, fs: float, spindle_hz: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        c = x - x.mean()
        rms = float(np.sqrt(np.mean(c ** 2)))
        crest = float(np.max(np.abs(c)) / max(rms, 1e-12))
        kurt = float(stats.kurtosis(c, fisher=False, bias=False))
        p2p = float(np.ptp(c))
        freq, psd = signal.welch(c, fs=fs, nperseg=min(self.s.welch_segment, len(c)))
        modes = [self.band_energy(freq, psd, m - self.s.mode_half_width_hz, m + self.s.mode_half_width_hz)
                 for m in self.s.structural_modes_hz]
        ci = self.chatter_indicator(np.asarray(context, dtype=np.float64), fs, spindle_hz)
        ent, loge = self.wavelet_features(c, fs)
        cen = self.band_centroid_khz(c, fs)
        return np.array([rms, p2p, kurt, crest, np.log1p(modes[0]), np.log1p(modes[1]), ci, ent, loge, cen],
                        dtype=np.float32)

    def window_features(self, window: np.ndarray, context: np.ndarray, fs: float, spindle_hz: float) -> np.ndarray:
        """window [channels, samples], context [channels, context_samples] -> channels * 10 features."""
        return np.concatenate([self.channel_features(w, c, fs, spindle_hz) for w, c in zip(window, context)])


def zone_spectrum_peak(zone_signal: np.ndarray, fs: float, band: tuple, nperseg: int) -> tuple[float, float]:
    """Highest peak of the 3-axis-summed Welch PSD inside `band`: returns (peak_hz, prominence_db over band median)."""
    c = zone_signal - zone_signal.mean(axis=1, keepdims=True)
    freq, psd = signal.welch(c, fs=fs, nperseg=min(nperseg, c.shape[1]), axis=1)
    psd = psd.sum(axis=0)
    m = (freq >= band[0]) & (freq <= band[1])
    bf, bp = freq[m], psd[m]
    k = int(np.argmax(bp))
    return float(bf[k]), float(10 * np.log10(bp[k] / max(np.median(bp), 1e-30)))


def chatter_label(peak_hz: float, prominence_db: float, spindle_hz: float, tolerance_hz: float,
                  threshold_db: float) -> int:
    """THE RULE: unstable (1) if the dominant chatter-band peak is NOT a spindle harmonic and
    rises at least `threshold_db` above the band median."""
    on_harmonic = bool(harmonic_mask(np.array([peak_hz]), spindle_hz, tolerance_hz)[0])
    return int((not on_harmonic) and prominence_db >= threshold_db)
