"""Time-frequency node features (Section 2.4 of the draft) and the chatter label rule.

Per channel and per window we compute 10 numbers:
    0 RMS, 1 peak-to-peak, 2 kurtosis, 3 crest factor        (time domain)
    4 band energy at mode 1 (3.5 kHz), 5 band energy at mode 2 (4.3 kHz)   (Eq. 5)
    6 chatter indicator CI = non-synchronous share of band energy          (Eq. 7)
    7 wavelet energy entropy H_w                                            (Eq. 6)
    8 log total wavelet energy
    9 STFT spectral centroid inside the chatter band (time-localised content)
"""
import numpy as np
import pywt
from scipy import signal, stats

from config import SignalConfig, LabelConfig

FEATURES_PER_CHANNEL = 10
FEATURE_NAMES = ["rms", "p2p", "kurtosis", "crest", "mode1_energy", "mode2_energy",
                 "chatter_indicator", "wavelet_entropy", "log_wavelet_energy", "band_centroid"]


def _band_energy(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integrate a PSD between two frequencies (trapezoidal rule)."""
    mask = (freq >= low) & (freq <= high)
    return float(np.trapezoid(psd[mask], freq[mask])) if mask.sum() > 1 else 0.0


def synchronous_mask(freq: np.ndarray, base_hz: float, tolerance_hz: float) -> np.ndarray:
    """True for PSD bins that lie on a harmonic n * base_hz (forced vibration).

    For labelling, base_hz is the spindle frequency (rpm/60): its harmonics
    contain every tooth-passing harmonic plus run-out, so everything that repeats
    once per revolution counts as 'synchronous', not chatter.
    """
    nearest = np.round(freq / base_hz) * base_hz
    return np.abs(freq - nearest) <= tolerance_hz


def chatter_indicator(freq: np.ndarray, psd: np.ndarray, spindle_rpm: float, cfg: SignalConfig,
                      tooth_count: int = 4) -> float:
    """Eq. (7): share of chatter-band energy NOT at tooth-passing harmonics.

    Window-level PSDs are coarse (~24 Hz bins at 100 ms), so harmonics of the
    tooth-passing frequency (66.7 / 93.3 Hz) are used with a half-bin tolerance.
    """
    low, high = cfg.chatter_band_hz
    band = (freq >= low) & (freq <= high)
    total = float(psd[band].sum())
    if total <= 0:
        return 0.0
    resolution = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
    tooth_hz = spindle_rpm * tooth_count / 60.0
    sync = synchronous_mask(freq, tooth_hz, resolution / 2)
    return float(psd[band & ~sync].sum() / total)


def wavelet_features(x: np.ndarray, fs: float, cfg: SignalConfig) -> tuple[float, float]:
    """Eq. (6): wavelet energy entropy (normalised to 0..1) and log total CWT energy."""
    # Scales chosen so the centre frequencies span roughly 500 Hz .. fs/2.
    centre = pywt.central_frequency(cfg.wavelet)
    freqs = np.geomspace(500.0, fs / 2.2, cfg.wavelet_scales)
    scales = centre * fs / freqs
    coefficients, _ = pywt.cwt(x, scales, cfg.wavelet, sampling_period=1.0 / fs, method="fft")
    energy = np.mean(np.abs(coefficients) ** 2, axis=1)
    p = energy / max(float(energy.sum()), 1e-20)
    entropy = float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(p)))
    return entropy, float(np.log1p(energy.sum()))


def channel_features(x: np.ndarray, fs: float, spindle_rpm: float, cfg: SignalConfig) -> np.ndarray:
    """Return the 10 features listed in the module docstring for one channel."""
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean()
    rms = float(np.sqrt(np.mean(centred ** 2)))
    crest = float(np.max(np.abs(centred)) / max(rms, 1e-12))
    kurt = float(stats.kurtosis(centred, fisher=False, bias=False))
    freq, psd = signal.welch(centred, fs=fs, nperseg=min(cfg.welch_segment, len(centred)))
    modes = [_band_energy(freq, psd, m - cfg.mode_half_width_hz, m + cfg.mode_half_width_hz)
             for m in cfg.structural_modes_hz]
    ci = chatter_indicator(freq, psd, spindle_rpm, cfg)
    entropy, log_energy = wavelet_features(centred, fs, cfg)
    # STFT: average spectral centroid of the chatter band over short sub-frames.
    f_s, _, z = signal.stft(centred, fs=fs, nperseg=256)
    band = (f_s >= cfg.chatter_band_hz[0]) & (f_s <= cfg.chatter_band_hz[1])
    mag = np.abs(z[band]) ** 2
    centroid = float(np.sum(f_s[band, None] * mag) / max(mag.sum(), 1e-20)) / 1000.0  # kHz
    return np.array([rms, float(np.ptp(centred)), kurt, crest, np.log1p(modes[0]), np.log1p(modes[1]),
                     ci, entropy, log_energy, centroid], dtype=np.float32)


def window_features(window: np.ndarray, fs: float, spindle_rpm: float, cfg: SignalConfig) -> np.ndarray:
    """Features for a [channels, samples] window -> [channels * 10]."""
    return np.concatenate([channel_features(ch, fs, spindle_rpm, cfg) for ch in window])


# --------------------------------------------------------------------------
# Labelling rule (computed once per zone on the FULL zone record, with a much
# finer frequency resolution than the 100 ms model windows can see).
# --------------------------------------------------------------------------
def zone_chatter_label(zone_signal: np.ndarray, fs: float, spindle_rpm: float,
                       sig_cfg: SignalConfig, lab_cfg: LabelConfig) -> dict:
    """Apply the pre-registered chatter rule to one zone.

    zone_signal: [channels, samples] force record covering the whole zone.
    Returns the label (1 = unstable, 0 = stable) and the quantities behind it.
    """
    centred = zone_signal - zone_signal.mean(axis=1, keepdims=True)
    nperseg = min(lab_cfg.label_nperseg, centred.shape[1])
    freq, psd = signal.welch(centred, fs=fs, nperseg=nperseg, axis=1)
    psd = psd.sum(axis=0)                                  # combine the three force axes
    low, high = sig_cfg.chatter_band_hz
    band = (freq >= low) & (freq <= high)
    f_band, p_band = freq[band], psd[band]
    peak = int(np.argmax(p_band))
    peak_hz = float(f_band[peak])
    prominence_db = float(10 * np.log10(p_band[peak] / max(np.median(p_band), 1e-30)))
    spindle_hz = spindle_rpm / 60.0
    is_sync = bool(synchronous_mask(np.array([peak_hz]), spindle_hz, sig_cfg.harmonic_tolerance_hz)[0])
    unstable = (not is_sync) and prominence_db >= lab_cfg.peak_prominence_db
    sync = synchronous_mask(f_band, spindle_hz, sig_cfg.harmonic_tolerance_hz)
    ci_zone = float(p_band[~sync].sum() / max(p_band.sum(), 1e-30))
    return {"chatter_label": int(unstable), "peak_hz": peak_hz, "peak_prominence_db": prominence_db,
            "peak_is_spindle_harmonic": is_sync, "ci_zone": ci_zone}
