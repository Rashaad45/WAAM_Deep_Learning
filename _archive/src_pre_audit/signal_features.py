"""
signal_features.py
==================
Turns raw sensor signals into numbers (features) that a model can learn from,
and decides whether a zone shows chatter (the label).

Two classes live in this file:

1. WindowFeatureExtractor  (Section 2.4 of the draft)
   For ONE window of ONE sensor channel it computes 10 numbers:
       0  rms                 - size of the signal (root mean square)
       1  p2p                 - peak-to-peak range (max - min)
       2  kurtosis            - how "spiky" the signal is
       3  crest               - highest peak divided by the RMS
       4  mode1_energy        - energy around structural mode 1 (3.5 kHz)    Eq. (5)
       5  mode2_energy        - energy around structural mode 2 (4.3 kHz)    Eq. (5)
       6  chatter_indicator   - share of band energy NOT at tooth harmonics  Eq. (7)
       7  wavelet_entropy     - how spread out the energy is over scales     Eq. (6)
       8  log_wavelet_energy  - log of the total wavelet energy
       9  band_centroid       - "centre of mass" frequency in the chatter band (kHz)

2. ZoneChatterLabeler
   Applies the chatter rule (fixed before training) to the FULL 3.75 s record of
   one zone and returns 1 (unstable) or 0 (stable).

Words you will meet
-------------------
* PSD  (power spectral density): how much signal power sits at each frequency.
  scipy.signal.welch() estimates it.
* Harmonic: a whole-number multiple of a base frequency (1x, 2x, 3x, ...).
  Vibration at spindle harmonics is "forced" by the rotating tool; chatter
  appears at OTHER frequencies.
"""
import numpy as np                  # arrays and maths
import pywt                         # PyWavelets: continuous wavelet transform
from scipy import signal, stats     # signal: PSD/STFT, stats: kurtosis

from config import SignalConfig, LabelConfig

# How many features one channel produces, and their names (in order).
FEATURES_PER_CHANNEL = 10
FEATURE_NAMES = ["rms", "p2p", "kurtosis", "crest", "mode1_energy", "mode2_energy",
                 "chatter_indicator", "wavelet_entropy", "log_wavelet_energy", "band_centroid"]


def harmonic_mask(freq: np.ndarray, base_hz: float, tolerance_hz: float) -> np.ndarray:
    """Return True for every frequency that lies on a harmonic n * base_hz.

    Example: base_hz = 100, tolerance = 2  ->  198 Hz is True (close to 200),
             250 Hz is False (50 Hz away from both 200 and 300).
    """
    # Round every frequency to the nearest multiple of base_hz ...
    nearest_harmonic = np.round(freq / base_hz) * base_hz
    # ... and keep it if it is close enough to that multiple.
    return np.abs(freq - nearest_harmonic) <= tolerance_hz


# ===========================================================================
# 1. Features of one short window
# ===========================================================================
class WindowFeatureExtractor:
    """Computes the 10 hand-crafted features of every channel of a window."""

    def __init__(self, settings: SignalConfig, tooth_count: int = 4):
        self.settings = settings            # all signal settings from config.py
        self.tooth_count = tooth_count      # number of cutting edges on the tool

    # -----------------------------------------------------------------------
    # Small helper calculations
    # -----------------------------------------------------------------------
    @staticmethod
    def band_energy(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
        """Area under the PSD between `low` and `high` Hz (trapezoidal rule)."""
        inside = (freq >= low) & (freq <= high)          # True for bins inside the band
        if inside.sum() <= 1:                            # need at least 2 points for an area
            return 0.0
        return float(np.trapezoid(psd[inside], freq[inside]))

    def chatter_indicator(self, freq: np.ndarray, psd: np.ndarray, spindle_rpm: float) -> float:
        """Eq. (7): fraction of the chatter-band energy that is NOT at tooth-passing harmonics.

        0 = all energy is forced vibration, 1 = all energy is at "free" frequencies.
        """
        low, high = self.settings.chatter_band_hz
        in_band = (freq >= low) & (freq <= high)
        total_energy = float(psd[in_band].sum())
        if total_energy <= 0:
            return 0.0

        # Width of one PSD bin (about 24 Hz for a 1024-point Welch segment).
        resolution = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
        # Tooth-passing frequency = revolutions per second * number of teeth.
        tooth_hz = spindle_rpm * self.tooth_count / 60.0
        # Bins within half a bin of a tooth harmonic count as "synchronous".
        synchronous = harmonic_mask(freq, tooth_hz, resolution / 2)

        non_synchronous_energy = psd[in_band & ~synchronous].sum()   # "~" means NOT
        return float(non_synchronous_energy / total_energy)

    def wavelet_features(self, x: np.ndarray, fs: float) -> tuple[float, float]:
        """Eq. (6): wavelet energy entropy (0..1) and log of the total CWT energy."""
        # Pick wavelet scales whose centre frequencies run from 500 Hz to about fs/2.
        centre_frequency = pywt.central_frequency(self.settings.wavelet)
        target_freqs = np.geomspace(500.0, fs / 2.2, self.settings.wavelet_scales)
        scales = centre_frequency * fs / target_freqs

        # Continuous wavelet transform -> one row of coefficients per scale.
        coefficients, _ = pywt.cwt(x, scales, self.settings.wavelet,
                                   sampling_period=1.0 / fs, method="fft")

        # Mean energy of every scale.
        energy_per_scale = np.mean(np.abs(coefficients) ** 2, axis=1)
        # Turn energies into probabilities that sum to 1.
        p = energy_per_scale / max(float(energy_per_scale.sum()), 1e-20)
        # Shannon entropy, divided by log(J) so the result lies between 0 and 1.
        entropy = float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(p)))
        log_energy = float(np.log1p(energy_per_scale.sum()))
        return entropy, log_energy

    def band_centroid_khz(self, centred: np.ndarray, fs: float) -> float:
        """Average frequency (kHz) of the chatter band, weighted by STFT power."""
        # Short-time Fourier transform: many short spectra along the window.
        f_stft, _, z = signal.stft(centred, fs=fs, nperseg=256)
        low, high = self.settings.chatter_band_hz
        in_band = (f_stft >= low) & (f_stft <= high)
        power = np.abs(z[in_band]) ** 2                      # [band bins, time frames]
        # Weighted mean frequency: sum(f * power) / sum(power).
        # f_stft[in_band, None] adds a second axis so it multiplies every time frame.
        centroid_hz = float(np.sum(f_stft[in_band, None] * power) / max(power.sum(), 1e-20))
        return centroid_hz / 1000.0

    # -----------------------------------------------------------------------
    # Main methods
    # -----------------------------------------------------------------------
    def channel_features(self, x: np.ndarray, fs: float, spindle_rpm: float) -> np.ndarray:
        """Return the 10 features (listed at the top of this file) of one channel."""
        x = np.asarray(x, dtype=np.float64)
        centred = x - x.mean()                               # remove the DC offset

        # --- time domain -----------------------------------------------------
        rms = float(np.sqrt(np.mean(centred ** 2)))
        crest = float(np.max(np.abs(centred)) / max(rms, 1e-12))
        kurt = float(stats.kurtosis(centred, fisher=False, bias=False))
        peak_to_peak = float(np.ptp(centred))

        # --- frequency domain (Welch PSD) ------------------------------------
        segment = min(self.settings.welch_segment, len(centred))
        freq, psd = signal.welch(centred, fs=fs, nperseg=segment)

        # Energy in a +/- 150 Hz band around each structural mode.
        mode_energies = []
        for mode_hz in self.settings.structural_modes_hz:
            low = mode_hz - self.settings.mode_half_width_hz
            high = mode_hz + self.settings.mode_half_width_hz
            mode_energies.append(self.band_energy(freq, psd, low, high))

        ci = self.chatter_indicator(freq, psd, spindle_rpm)

        # --- time-frequency domain --------------------------------------------
        entropy, log_energy = self.wavelet_features(centred, fs)
        centroid = self.band_centroid_khz(centred, fs)

        # Same order as FEATURE_NAMES.
        return np.array([rms, peak_to_peak, kurt, crest,
                         np.log1p(mode_energies[0]), np.log1p(mode_energies[1]),
                         ci, entropy, log_energy, centroid], dtype=np.float32)

    def window_features(self, window: np.ndarray, fs: float, spindle_rpm: float) -> np.ndarray:
        """Features of a [channels, samples] window -> one vector of channels * 10 numbers."""
        per_channel = [self.channel_features(channel, fs, spindle_rpm) for channel in window]
        return np.concatenate(per_channel)


# ===========================================================================
# 2. The chatter label of one zone
# ===========================================================================
class ZoneChatterLabeler:
    """Applies the pre-registered chatter rule to the full force record of a zone.

    The rule uses a MUCH finer frequency resolution (~0.76 Hz) than the 100 ms
    model windows can see (~24 Hz). This keeps the label from being a simple
    copy of the model inputs.
    """

    def __init__(self, signal_settings: SignalConfig, label_settings: LabelConfig):
        self.signal_settings = signal_settings
        self.label_settings = label_settings

    def label_zone(self, zone_signal: np.ndarray, fs: float, spindle_rpm: float) -> dict:
        """zone_signal is [3 force axes, samples]. Returns the label and the numbers behind it."""
        # 1) Remove the mean of every axis.
        centred = zone_signal - zone_signal.mean(axis=1, keepdims=True)

        # 2) High-resolution PSD of every axis, then add the three axes together.
        segment = min(self.label_settings.label_nperseg, centred.shape[1])
        freq, psd = signal.welch(centred, fs=fs, nperseg=segment, axis=1)
        psd = psd.sum(axis=0)

        # 3) Keep only the chatter band.
        low, high = self.signal_settings.chatter_band_hz
        in_band = (freq >= low) & (freq <= high)
        band_freq, band_psd = freq[in_band], psd[in_band]

        # 4) Find the highest peak in the band and how far it rises above the median.
        peak_index = int(np.argmax(band_psd))
        peak_hz = float(band_freq[peak_index])
        prominence_db = float(10 * np.log10(band_psd[peak_index] / max(np.median(band_psd), 1e-30)))

        # 5) Is that peak a spindle harmonic (forced vibration)?
        spindle_hz = spindle_rpm / 60.0
        tolerance = self.signal_settings.harmonic_tolerance_hz
        peak_is_harmonic = bool(harmonic_mask(np.array([peak_hz]), spindle_hz, tolerance)[0])

        # 6) THE RULE: unstable = not a harmonic AND at least 20 dB above the median.
        unstable = (not peak_is_harmonic) and prominence_db >= self.label_settings.peak_prominence_db

        # Extra information (stored in zone_labels.csv): share of non-harmonic energy.
        synchronous = harmonic_mask(band_freq, spindle_hz, tolerance)
        ci_zone = float(band_psd[~synchronous].sum() / max(band_psd.sum(), 1e-30))

        return {"chatter_label": int(unstable),
                "peak_hz": peak_hz,
                "peak_prominence_db": prominence_db,
                "peak_is_spindle_harmonic": peak_is_harmonic,
                "ci_zone": ci_zone}
