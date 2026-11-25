"""Small DSP helpers shared by the analysis code (numpy/scipy only)."""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12


def db(x, floor: float = -150.0):
    """Power -> dB."""
    return np.maximum(10.0 * np.log10(np.maximum(x, EPS)), floor)


def amp_db(x, floor: float = -150.0):
    """Amplitude -> dB."""
    return np.maximum(20.0 * np.log10(np.maximum(np.abs(x), EPS)), floor)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Return shape (n, 2)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=1)


def frame_rms(x: np.ndarray, sr: float, win_s: float, hop_s: float) -> np.ndarray:
    """RMS (power, not dB) of mono x in windows of win_s seconds every hop_s seconds.
    Frame i covers [i*hop, i*hop+win)."""
    win = max(1, int(round(win_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    if len(x) < win:
        return np.array([np.mean(x ** 2)]) if len(x) else np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - win + 1, hop)
    return (c[starts + win] - c[starts]) / win


def biquad_highshelf(f0: float, gain_db: float, sr: float, q: float = 0.707):
    a = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * f0 / sr
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    b0 = a * ((a + 1) + (a - 1) * cw + 2 * np.sqrt(a) * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cw)
    b2 = a * ((a + 1) + (a - 1) * cw - 2 * np.sqrt(a) * alpha)
    a0 = (a + 1) - (a - 1) * cw + 2 * np.sqrt(a) * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cw)
    a2 = (a + 1) - (a - 1) * cw - 2 * np.sqrt(a) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def k_weight(x: np.ndarray, sr: float) -> np.ndarray:
    """ITU-R BS.1770 style K-weighting (shelf + high-pass), good enough for
    relative loudness comparisons between vocal and music."""
    b, a = biquad_highshelf(1681.0, 4.0, sr)
    y = signal.lfilter(b, a, x, axis=0)
    sos = signal.butter(2, 38.0, "highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, y, axis=0)


def third_octave_centres(lo: float = 50.0, hi: float = 16000.0) -> np.ndarray:
    n = np.arange(-20, 14)
    f = 1000.0 * 2.0 ** (n / 3.0)
    return f[(f >= lo * 0.99) & (f <= hi * 1.01)]


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    return float(np.sum(psd[m])) if np.any(m) else 0.0


def runs(mask: np.ndarray):
    """(start, end) index pairs of consecutive True runs (end exclusive)."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
