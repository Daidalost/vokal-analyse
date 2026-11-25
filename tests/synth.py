"""Synthetic test material: a sung vocal, a backing track and typical vocal
processing with known settings, so the analysis can be checked against facts."""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 44100
HP_SOS = signal.butter(2, 3000, "highpass", fs=SR, output="sos")

VOWELS = {  # formant frequencies (Hz) and bandwidths for sung vowels
    "a": ([800, 1150, 2900, 3900], [80, 90, 120, 130]),
    "e": ([400, 1700, 2600, 3200], [70, 80, 100, 120]),
    "i": ([290, 2250, 2900, 3600], [50, 90, 100, 120]),
    "o": ([450, 800, 2830, 3500], [70, 80, 100, 130]),
    "u": ([325, 700, 2530, 3500], [50, 60, 170, 180]),
}


def _resonator(f, bw, sr):
    """Klatt-style two-pole resonator with unity gain at DC."""
    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * f / sr
    a = [1.0, -2 * r * np.cos(th), r * r]
    b = [sum(a)]
    return b, a


def _formants(x, vowel, sr):
    """Cascade of the four formants (natural relative formant levels)."""
    fs, bws = VOWELS[vowel]
    y = x
    for f, bw in zip(fs, bws):
        b, a = _resonator(f, bw, sr)
        y = signal.lfilter(b, a, y)
    return y


def sung_vocal(seconds=24.0, f0_base=140.0, seed=1, sr=SR, s_level_db=-6.0,
               bpm=100.0, timing_jitter=0.0, detune_cents=0.0):
    """Mono sung line with phrases, vibrato, level variation and S-sounds."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(seconds * sr)
    out = np.zeros(n)
    t = 0.5
    scale = [0, 2, 4, 5, 7, 9, 7, 4]
    rng_t = np.random.default_rng(seed + 1000)  # timing variation for a double take
    while t < seconds - 3.0:
        n_syl = rng.integers(4, 8)
        phrase_gain = 10 ** (rng.uniform(-4, 4) / 20)
        for _ in range(n_syl):
            dur = beat * rng.choice([0.5, 1.0, 1.0, 1.5])
            vowel = rng.choice(list(VOWELS))
            semis = rng.choice(scale)
            f0 = f0_base * 2 ** (semis / 12) * 2 ** (detune_cents / 1200)
            gain = phrase_gain * 10 ** (rng.uniform(-6, 6) / 20)
            has_s = rng.random() < 0.3
            start = t + timing_jitter * rng_t.uniform(-1, 1)
            s0 = int(start * sr)
            m = int(dur * sr)
            if s0 + m >= n:
                break
            tt = np.arange(m) / sr
            vib = 0.4 * np.minimum(tt / 0.3, 1.0) * np.sin(2 * np.pi * 5.5 * tt + rng.uniform(0, 6))
            ff = f0 * 2 ** (vib / 12) * (1 + 0.003 * rng.standard_normal(m).cumsum() / np.sqrt(np.arange(1, m + 1)))
            ph = 2 * np.pi * np.cumsum(ff) / sr
            src = np.zeros(m)
            kmax = int((sr / 2 - 500) / (f0 * 1.05))
            for k in range(1, kmax + 1):
                src += np.sin(k * ph) / k  # -6 dB/oct source + radiation
            src += 0.02 * rng.standard_normal(m)  # aspiration noise
            voice = _formants(src, vowel, sr)
            # higher formants / presence: real voices keep energy above F4
            voice += 0.25 * signal.sosfilt(HP_SOS, src)
            env = np.ones(m)
            a_n, r_n = int(0.03 * sr), int(0.07 * sr)
            env[:a_n] = np.linspace(0, 1, a_n)
            env[-r_n:] = np.linspace(1, 0, r_n)
            seg = voice * env * gain
            seg /= 1e-9 + np.sqrt(np.mean(voice ** 2))
            out[s0:s0 + m] += 0.1 * seg
            if has_s:  # an S just before the vowel
                sl = int(rng.uniform(0.08, 0.13) * sr)
                st = max(0, s0 - sl)
                nz = rng.standard_normal(sl)
                sos = signal.butter(4, [5000, 9500], "bandpass", fs=sr, output="sos")
                nz = signal.sosfilt(sos, nz)
                nz *= np.hanning(sl)
                nz /= np.sqrt(np.mean(nz ** 2)) + 1e-9
                out[st:st + sl] += 0.1 * gain * 10 ** (s_level_db / 20) * nz
            t += dur + beat * 0.05
        t += beat * rng.choice([2.0, 3.0, 4.0])
    return out


def backing_track(seconds=24.0, bpm=100.0, seed=7, sr=SR):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros((n, 2))
    beat = 60.0 / bpm
    nb = int(seconds / beat)
    for i in range(nb):
        s = int(i * beat * sr)
        # kick
        m = int(0.25 * sr)
        if s + m < n:
            tt = np.arange(m) / sr
            k = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
            out[s:s + m] += 0.5 * k[:, None]
        # snare on 2 and 4
        if i % 2 == 1 and s + m < n:
            nz = rng.standard_normal(m) * np.exp(-np.arange(m) / sr * 20)
            nz = signal.sosfilt(signal.butter(2, [200, 6000], "bandpass", fs=sr, output="sos"), nz)
            out[s:s + m] += 0.25 * np.stack([nz, nz * 0.9], 1)
        # hats on eighths
        for h in (0, 0.5):
            hs = int((i + h) * beat * sr)
            hm = int(0.05 * sr)
            if hs + hm < n:
                nz = rng.standard_normal(hm) * np.exp(-np.arange(hm) / sr * 80)
                nz = signal.sosfilt(signal.butter(2, 8000, "highpass", fs=sr, output="sos"), nz)
                out[hs:hs + hm] += 0.08 * np.stack([nz * 0.7, nz], 1)
    # bass + pad, one chord per bar
    roots = [55.0, 43.65, 65.41, 49.0]
    bar = 4 * beat
    for j in range(int(seconds / bar) + 1):
        s, m = int(j * bar * sr), int(bar * sr)
        m = min(m, n - s)
        if m <= 0:
            break
        tt = np.arange(m) / sr
        r = roots[j % 4]
        bass = signal.sawtooth(2 * np.pi * r * tt)
        bass = signal.sosfilt(signal.butter(2, 300, fs=sr, output="sos"), bass)
        pad = sum(np.sin(2 * np.pi * r * 4 * 2 ** (iv / 12) * tt + rng.uniform(0, 6)) for iv in (0, 4, 7))
        env = np.minimum(1, np.minimum(tt / 0.05, (m / sr - tt) / 0.05))
        out[s:s + m, 0] += env * (0.3 * bass + 0.04 * pad)
        out[s:s + m, 1] += env * (0.3 * bass + 0.04 * np.roll(pad, 300))
    return out


# ---------------------------------------------------------------- processing

def tilt_eq(x, db_per_oct, pivot=1000.0, sr=SR):
    X = np.fft.rfft(x, axis=0)
    f = np.maximum(np.fft.rfftfreq(len(x), 1 / sr), 40.0)
    g = 10 ** (db_per_oct * np.log2(f / pivot) / 20)
    if X.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(X * g, len(x), axis=0)


def band_gain(x, lo, hi, gain_db, sr=SR):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    g = np.where((f >= lo) & (f < hi), 10 ** (gain_db / 20), 1.0)
    if X.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(X * g, len(x), axis=0)


def compress(x, threshold_db=-30.0, ratio=6.0, attack=0.005, release=0.08, sr=SR):
    xm = np.abs(x if x.ndim == 1 else x.max(axis=1))
    # peak-ish envelope: fast attack via max filter, then release smoothing
    a = np.exp(-1 / (release * sr))
    env = signal.lfilter([1 - a], [1, -a], xm)
    env = np.maximum(env, signal.lfilter([1 - np.exp(-1 / (attack * sr))], [1, -np.exp(-1 / (attack * sr))], xm))
    lvl = 20 * np.log10(env + 1e-9)
    over = np.maximum(lvl - threshold_db, 0)
    gr = over * (1 - 1 / ratio)
    g = 10 ** (-gr / 20)
    y = x * (g if x.ndim == 1 else g[:, None])
    # make-up so loud parts land where they were
    return y * 10 ** (np.percentile(gr[xm > 1e-3], 90) / 20 * 0.0)


def reverb(x, rt60=1.8, wet_db=-8.0, predelay=0.02, seed=3, sr=SR):
    """Stereo output: dry in the centre plus decorrelated exponential tail."""
    rng = np.random.default_rng(seed)
    m = int(rt60 * 1.2 * sr)
    tt = np.arange(m) / sr
    decay = 10 ** (-3 * tt / rt60)
    pd = int(predelay * sr)
    xm = x if x.ndim == 1 else x.mean(axis=1)
    outs = []
    for ch in range(2):
        ir = rng.standard_normal(m) * decay
        ir = signal.sosfilt(signal.butter(1, 6000, fs=sr, output="sos"), ir)
        ir /= np.sqrt(np.sum(ir ** 2))
        ir = np.concatenate([np.zeros(pd), ir])
        outs.append(signal.fftconvolve(xm, ir)[:len(xm)])
    wet = np.stack(outs, 1) * 10 ** (wet_db / 20)
    dry = np.stack([xm, xm], 1) if x.ndim == 1 else x
    return dry + wet


def delay(x, ms=375.0, feedback=0.35, level_db=-9.0, repeats=4, sr=SR):
    d = int(ms / 1000 * sr)
    y = np.array(x, dtype=float, copy=True)
    g = 10 ** (level_db / 20)
    for k in range(1, repeats + 1):
        if k * d >= len(x):
            break
        y[k * d:] += g * feedback ** (k - 1) * x[:-k * d]
    return y


def double_track(f0_base=140.0, seed=1, seconds=24.0, bpm=100.0, sr=SR):
    """Lead in the centre and two extra takes panned hard left/right."""
    lead = sung_vocal(seconds, f0_base, seed, sr, bpm=bpm)
    l = sung_vocal(seconds, f0_base, seed, sr, bpm=bpm, timing_jitter=0.02, detune_cents=-8)
    r = sung_vocal(seconds, f0_base, seed, sr, bpm=bpm, timing_jitter=0.02, detune_cents=9)
    # different takes have different random jitter in the source: change the seed for breath only
    g = 10 ** (-4 / 20)
    return np.stack([lead + g * l, lead + g * np.roll(r, 37)], 1)
