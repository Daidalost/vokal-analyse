"""Measurements on a separated vocal stem (and the music behind it).

Everything here is plain numpy/scipy so it can be tested without the
separation model. The input is the vocal stem and, when available, the
instrumental stem of the same track, both at the same sample rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy import signal

from .dsp import (EPS, db, to_stereo, mono, frame_rms, k_weight, third_octave_centres,
                  band_power, runs)

NFFT = 2048

# Named regions used in the report (Hz).
REGIONS = {
    "low": (80.0, 200.0),         # chest / proximity effect
    "low_mid": (200.0, 500.0),    # warmth -> boxiness / mud
    "mid": (500.0, 2000.0),       # body of the vowels, nasal / honky
    "presence": (2000.0, 5000.0),  # intelligibility, forwardness, harshness
    "sibilance": (5000.0, 10000.0),
    "air": (10000.0, 16000.0),
}


@dataclass
class Measure:
    value: float
    confidence: float  # 0..1, how much to trust the number

    def __float__(self):
        return float(self.value)


@dataclass
class VocalProfile:
    ok: bool = False
    error: str = ""
    sample_rate: float = 0.0
    seconds: float = 0.0
    active_seconds: float = 0.0

    # Tone
    third_oct_hz: list = field(default_factory=list)
    third_oct_db: list = field(default_factory=list)   # relative, mean of 250 Hz-4 kHz = 0
    regions_db: dict = field(default_factory=dict)      # name -> relative dB per 1/3 octave
    tilt_db_per_oct: Optional[Measure] = None
    f0_median: Optional[Measure] = None
    f0_range: tuple = (0.0, 0.0)

    # Dynamics (dB)
    syllable_range_db: Optional[Measure] = None   # P95-P10 of 50 ms levels while singing
    phrase_range_db: Optional[Measure] = None     # P95-P10 of 400 ms levels
    crest_db: Optional[Measure] = None            # peak to RMS within 400 ms windows

    # Sibilance
    sibilant_share: float = 0.0                   # share of singing time that is S-like
    sib_vs_vowel_db: Optional[Measure] = None     # S peak level minus vowel peak level
    sib_freq_hz: Optional[Measure] = None

    # Harmonic clarity (saturation proxy)
    harmonic_share: Optional[Measure] = None      # 0..1, energy on harmonics 1-4 kHz

    # Space
    side_db: Optional[Measure] = None             # side/mid energy while singing
    onset_side_db: Optional[Measure] = None       # side/mid at note onsets (doubling)
    tail_side_db: Optional[Measure] = None        # side/mid in the tails (stereo reverb)
    tail_level_db: Optional[Measure] = None       # level 100-250 ms after phrase end re phrase
    rt60_s: Optional[Measure] = None              # decay of the tail
    delay_ms: Optional[Measure] = None            # discrete echo, if any
    delay_strength: float = 0.0

    # Mix
    bpm: Optional[Measure] = None
    vocal_vs_music_db: Optional[Measure] = None   # K-weighted, while singing

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- helpers

def _stft_power(x: np.ndarray, sr: float, nfft: int = NFFT, hop: int = NFFT // 4):
    f, t, z = signal.stft(x, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - hop,
                          boundary=None, padded=False, axis=0)
    p = np.abs(z) ** 2
    if p.ndim == 3:  # (freq, ch, time) for multichannel input
        p = np.moveaxis(p, 1, 0)
    return f, t, p


def _activity(frame_db: np.ndarray, below_peak: float) -> tuple[np.ndarray, float]:
    """Frames that hold voice: within below_peak dB of the loud end."""
    if frame_db.size == 0:
        return np.zeros(0, bool), -150.0
    ref = float(np.percentile(frame_db, 95))
    thr = max(ref - below_peak, -80.0)
    m = frame_db > thr
    # close tiny holes (consonant gaps) of one or two frames
    m2 = m.copy()
    for k in (1, 2):
        m2[k:-k] |= m[:-2 * k] & m[2 * k:]
    return m2, ref


def _band_spectrum(f, psd):
    centres = third_octave_centres(50.0, 16000.0)
    out = []
    for c in centres:
        out.append(band_power(f, psd, c / 2 ** (1 / 6), c * 2 ** (1 / 6)))
    return centres, db(np.array(out))


def _region_db(f, psd, lo, hi, norm):
    p = band_power(f, psd, lo, hi)
    thirds = 3.0 * np.log2(hi / lo)
    return float(db(p / thirds) - norm)


def _f0_track(x: np.ndarray, sr: float, frame_starts: np.ndarray, n: int = 2048):
    """Normalised autocorrelation pitch on selected frames. Returns f0 (0 = unvoiced)."""
    lag_lo = int(sr / 1000.0)
    lag_hi = int(sr / 70.0)
    n = max(n, 2 * lag_hi + 16)
    win = np.hanning(n)
    out = np.zeros(len(frame_starts))
    for i, s in enumerate(frame_starts):
        seg = x[s:s + n]
        if len(seg) < n:
            continue
        seg = (seg - seg.mean()) * win
        spec = np.fft.rfft(seg, 2 * n)
        ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
        if ac[0] <= EPS:
            continue
        # compensate the window's own autocorrelation taper
        ac = ac / ac[0]
        wac = np.correlate(win, win, "full")[n - 1:]
        wac = wac / wac[0]
        acn = ac[:lag_hi + 2] / np.maximum(wac[:lag_hi + 2], 1e-3)
        seg_ac = acn[lag_lo:lag_hi]
        if seg_ac.size == 0:
            continue
        best = float(seg_ac.max())
        if best < 0.55:
            continue
        # first local peak that reaches 85 % of the best one (avoids octave-down errors)
        cand = np.flatnonzero((seg_ac[1:-1] >= seg_ac[:-2]) & (seg_ac[1:-1] >= seg_ac[2:])
                              & (seg_ac[1:-1] >= 0.85 * best)) + 1
        k = int(cand[0]) if cand.size else int(np.argmax(seg_ac))
        lag = k + lag_lo
        # parabolic refinement
        if 1 <= k < len(seg_ac) - 1:
            a, b, c = seg_ac[k - 1], seg_ac[k], seg_ac[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                lag += 0.5 * (a - c) / den
        out[i] = sr / lag
    return out


def _harmonic_share(x, sr, starts, f0s, n=4096, lo=1000.0, hi=4000.0):
    win = np.hanning(n)
    shares = []
    for s, f0 in zip(starts, f0s):
        if f0 <= 0:
            continue
        seg = x[s:s + n]
        if len(seg) < n:
            continue
        p = np.abs(np.fft.rfft(seg * win)) ** 2
        freqs = np.fft.rfftfreq(n, 1 / sr)
        band = (freqs >= lo) & (freqs < hi)
        tot = p[band].sum()
        if tot <= EPS:
            continue
        df = sr / n
        h = np.arange(np.ceil(lo / f0), np.floor(hi / f0) + 1) * f0
        on = np.zeros_like(band)
        tol = max(1.5 * df, 0.02 * f0)  # allow some vibrato
        for fh in h:
            on |= np.abs(freqs - fh) <= tol
        on &= band
        # expected share for pure noise = fraction of bins that count as "on"
        noise_share = on.sum() / max(band.sum(), 1)
        share = p[on].sum() / tot
        if noise_share < 0.6:
            shares.append((share - noise_share) / (1 - noise_share))
    return shares


def estimate_bpm(x: np.ndarray, sr: float) -> Measure:
    """Tempo from the onset envelope (spectral flux) autocorrelation."""
    xm = mono(x)
    if len(xm) < sr * 4:
        return Measure(0.0, 0.0)
    f, t, p = _stft_power(xm, sr, 2048, 512)
    hop_s = 512 / sr
    edges = np.geomspace(40, min(sr / 2, 12000), 25)
    bands = np.stack([p[(f >= a) & (f < b)].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    lb = np.log1p(1000 * bands / (bands.max() + EPS))
    flux = np.maximum(np.diff(lb, axis=1), 0).sum(axis=0)
    flux = flux - signal.medfilt(flux, 15)
    flux = np.maximum(flux, 0)
    flux -= flux.mean()
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    ac /= ac[0] + EPS
    best, best_bpm = -1.0, 0.0
    lags = np.arange(int(60 / 200 / hop_s), int(60 / 60 / hop_s) + 1)
    scores = []
    for lag in lags:
        bpm = 60.0 / (lag * hop_s)
        prior = np.exp(-0.5 * (np.log2(bpm / 115.0) / 0.9) ** 2)
        # reward lags whose double also correlates (metrical structure)
        s = max(ac[lag], 0.0)
        scores.append(s * prior)
    scores = np.array(scores)
    i = int(np.argmax(scores))
    lag = lags[i]
    # parabolic refine
    if 0 < i < len(scores) - 1:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        den = a - 2 * b + c
        lag = lag + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0)
    best_bpm = 60.0 / (lag * hop_s)
    conf = float(np.clip(ac[lags[i]] * 2.0, 0, 1))
    # half/double tempo: songs are written mostly between ~80 and ~160 BPM, and a
    # kick/snare pattern often makes the half tempo correlate best
    li = int(round(lag))
    if best_bpm < 85 and li // 2 > 0 and ac[li // 2] > 0.5 * ac[li]:
        best_bpm *= 2
    elif best_bpm > 170 and 2 * li < len(ac) and ac[2 * li] > 0.5 * ac[li]:
        best_bpm /= 2
    return Measure(round(float(best_bpm), 1), conf)


# ---------------------------------------------------------------- main

def analyse_vocal(vocal: np.ndarray, sr: float, music: Optional[np.ndarray] = None) -> VocalProfile:
    prof = VocalProfile(sample_rate=sr)
    v = to_stereo(vocal)
    prof.seconds = len(v) / sr
    if len(v) < sr * 2:
        prof.error = "The vocal is shorter than 2 seconds."
        return prof

    mid = 0.5 * (v[:, 0] + v[:, 1])
    side = 0.5 * (v[:, 0] - v[:, 1])
    hop = NFFT // 4
    hop_s = hop / sr
    f, t, pm = _stft_power(mid, sr, NFFT, hop)
    _, _, ps = _stft_power(side, sr, NFFT, hop)
    frame_pow = pm.sum(axis=0) / NFFT
    frame_db = db(frame_pow)
    active, ref = _activity(frame_db, 35.0)
    if active.sum() * hop_s < 1.0:
        prof.error = "Hardly any vocal was found in the track."
        return prof
    prof.active_seconds = float(active.sum() * hop_s)
    starts = (np.arange(pm.shape[1]) * hop).astype(int)

    # ---- sibilant vs tonal frames
    hf = (f >= 4500) & (f < 11000)
    sib_frac = pm[hf].sum(axis=0) / (pm.sum(axis=0) + EPS)
    sibilant = active & (sib_frac > 0.5)
    tonal = active & ~sibilant

    # ---- pitch (subset of tonal frames)
    ton_idx = np.flatnonzero(tonal)
    sel = ton_idx[np.linspace(0, len(ton_idx) - 1, min(len(ton_idx), 1500)).astype(int)] if len(ton_idx) else ton_idx
    f0 = _f0_track(mid, sr, starts[sel])
    voiced = f0[f0 > 0]
    if voiced.size >= 20:
        prof.f0_median = Measure(round(float(np.median(voiced)), 1),
                                 float(np.clip(voiced.size / max(len(sel), 1) * 1.5, 0, 1)))
        prof.f0_range = (round(float(np.percentile(voiced, 5)), 1), round(float(np.percentile(voiced, 95)), 1))
        sub = np.linspace(0, len(sel) - 1, min(len(sel), 400)).astype(int)
        hs = _harmonic_share(mid, sr, starts[sel][sub], f0[sub])
        if len(hs) >= 20:
            prof.harmonic_share = Measure(round(float(np.median(hs)), 3), 0.4)

    # ---- tone (long-term spectrum of the tonal frames)
    # vowel spectrum from the frames with a clear pitch, so S-sounds and breaths
    # that slip into the tonal frames do not pull it around
    near_s = sibilant.copy()
    for k in (1, 2, 3):
        near_s[k:] |= sibilant[:-k]
        near_s[:-k] |= sibilant[k:]
    vf = sel[(f0 > 0) & ~near_s[sel]] if voiced.size >= 20 else ton_idx
    if len(vf) >= 20:
        # drop vowel frames with an S or breath riding on them (HF far above typical)
        hf_db = db(sib_frac[vf])
        vf = vf[hf_db < np.median(hf_db) + 8.0]
    psd = pm[:, vf].mean(axis=1) if len(vf) else pm[:, active].mean(axis=1)
    centres, bdb = _band_spectrum(f, psd)
    normmask = (centres >= 250) & (centres <= 4000)
    norm = float(np.mean(bdb[normmask]))
    prof.third_oct_hz = [float(c) for c in centres]
    prof.third_oct_db = [float(b - norm) for b in bdb]
    full_psd = pm[:, active].mean(axis=1)
    for name, (lo, hi) in REGIONS.items():
        if hi > sr / 2:
            continue
        src = full_psd if name == "sibilance" else psd
        prof.regions_db[name] = round(_region_db(f, src, lo, hi, norm), 2)
    tm = (centres >= 150) & (centres <= min(12000, sr / 2.2))
    slope = np.polyfit(np.log2(centres[tm]), bdb[tm], 1)[0]
    prof.tilt_db_per_oct = Measure(round(float(slope), 2), 0.9)

    # ---- sibilance
    prof.sibilant_share = round(float(sibilant.sum() / max(active.sum(), 1)), 3)
    if sibilant.sum() >= 5 and tonal.sum() >= 20:
        s_lvl = np.percentile(frame_db[sibilant], 90)
        v_lvl = np.percentile(frame_db[tonal], 90)
        prof.sib_vs_vowel_db = Measure(round(float(s_lvl - v_lvl), 2),
                                       float(np.clip(sibilant.sum() / 40, 0.2, 0.9)))
        sp = pm[:, sibilant].mean(axis=1)
        m = (f >= 4000) & (f <= 11000)
        sm = np.convolve(sp[m], np.ones(9) / 9, "same")
        prof.sib_freq_hz = Measure(round(float(f[m][np.argmax(sm)])), 0.5)

    # ---- space
    side_pow = ps.sum(axis=0) / NFFT
    prof.side_db = Measure(round(float(db(side_pow[active].sum() / (frame_pow[active].sum() + EPS))), 2), 0.8)

    phrase, _ = _activity(frame_db, 20.0)
    on_frames = np.zeros_like(phrase)
    for a, b in runs(phrase):
        if (b - a) * hop_s >= 0.15:
            on_frames[a:a + max(2, int(0.06 / hop_s))] = True
    if on_frames.sum() >= 10:
        prof.onset_side_db = Measure(round(float(db(side_pow[on_frames].sum() / (frame_pow[on_frames].sum() + EPS))), 2), 0.6)

    _space(prof, phrase, frame_db, frame_pow, side_pow, pm, f, hop_s)

    # ---- dynamics (K-weighted mid)
    kw = k_weight(mid, sr)
    l50 = db(frame_rms(kw, sr, 0.05, 0.025))
    a50, ref50 = _activity(l50, 30.0)
    # evenness of the syllables: spread of the syllable peak levels. Peaks (not all
    # frames) so a reverb tail or a breath does not count as "dynamics".
    pk, _ = signal.find_peaks(l50, distance=6, prominence=3.0, height=ref50 - 30.0)
    if prof.delay_ms is not None and len(pk):
        # echoes of a delay are not syllables: drop peaks that sit one delay time
        # after a louder peak
        dl = prof.delay_ms.value / 1000.0 / 0.025
        keep = []
        for i, k in enumerate(pk):
            src = pk[np.abs(pk - (k - dl)) <= 1.5]
            keep.append(not np.any(l50[src] > l50[k] + 2.0))
        pk = pk[np.array(keep, bool)]
    if len(pk) >= 12:
        vals = l50[pk]
        prof.syllable_range_db = Measure(round(float(np.percentile(vals, 90) - np.percentile(vals, 25)), 2),
                                         float(np.clip(len(pk) / 60, 0.3, 0.8)) * (0.6 if prof.delay_ms is not None else 1.0))
    l400 = db(frame_rms(kw, sr, 0.4, 0.1))
    a400, _ = _activity(l400, 20.0)
    if a400.sum() > 8:
        vals = l400[a400]
        prof.phrase_range_db = Measure(round(float(np.percentile(vals, 95) - np.percentile(vals, 10)), 2), 0.6)
        # crest: peak vs RMS inside the active 400 ms windows
        win, hp = int(0.4 * sr), int(0.1 * sr)
        crest = []
        for i in np.flatnonzero(a400):
            seg = kw[i * hp:i * hp + win]
            if len(seg) == win:
                crest.append(20 * np.log10(np.max(np.abs(seg)) + EPS) - 10 * np.log10(np.mean(seg ** 2) + EPS))
        if crest:
            prof.crest_db = Measure(round(float(np.median(crest)), 2), 0.6)


    # ---- mix context
    if music is not None:
        mus = to_stereo(music)
        n = min(len(mus), len(v))
        prof.bpm = estimate_bpm(mus[:n], sr)
        kv = k_weight(v[:n], sr)
        km = k_weight(mus[:n], sr)
        lv = frame_rms(kv[:, 0], sr, 0.4, 0.1) + frame_rms(kv[:, 1], sr, 0.4, 0.1)
        lm = frame_rms(km[:, 0], sr, 0.4, 0.1) + frame_rms(km[:, 1], sr, 0.4, 0.1)
        av, _ = _activity(db(lv), 15.0)
        if av.sum() > 5 and lm[av].sum() > EPS:
            prof.vocal_vs_music_db = Measure(round(float(db(lv[av].mean() / lm[av].mean())), 2), 0.7)

    prof.ok = True
    return prof


def _space(prof: VocalProfile, phrase, frame_db, frame_pow, side_pow, pm, f, hop_s):
    """Reverb tail, its width and discrete echoes, measured in the gaps after phrases."""
    floor_db = float(np.percentile(frame_db, 5))
    tails_lvl, slopes, tail_sides = [], [], []
    gaps = []
    rs = runs(phrase)
    for i, (a, b) in enumerate(rs):
        if (b - a) * hop_s < 0.25:
            continue
        nxt = rs[i + 1][0] if i + 1 < len(rs) else len(phrase)
        gap = nxt - b
        if gap * hop_s < 0.35:
            continue
        end_lvl = float(np.median(frame_db[max(a, b - int(0.2 / hop_s)):b]))
        g_end = min(nxt, b + int(1.2 / hop_s))
        gaps.append((a, b, g_end, end_lvl))
        w0, w1 = b + int(0.10 / hop_s), b + int(0.25 / hop_s)
        if w1 <= g_end:
            tails_lvl.append(float(np.mean(frame_db[w0:w1])) - end_lvl)
            tp = frame_pow[w0:w1].sum()
            if tp > EPS:
                tail_sides.append(float(db(side_pow[w0:w1].sum() / tp)))
        # decay slope from 60 ms to the point the tail meets the floor (max 800 ms)
        s0 = b + int(0.06 / hop_s)
        seg = frame_db[s0:min(g_end, b + int(0.8 / hop_s))]
        if len(seg) >= 8:
            stop = np.flatnonzero(seg < max(floor_db + 6, end_lvl - 60))
            seg = seg[:stop[0]] if stop.size else seg
            if len(seg) >= 8 and seg[0] - seg[-1] > 3:
                tt = np.arange(len(seg)) * hop_s
                slopes.append(np.polyfit(tt, seg, 1)[0])

    n = len(tails_lvl)
    if n:
        prof.tail_level_db = Measure(round(float(np.median(tails_lvl)), 2), float(np.clip(n / 8, 0.2, 0.9)))
    if tail_sides:
        prof.tail_side_db = Measure(round(float(np.median(tail_sides)), 2), float(np.clip(len(tail_sides) / 8, 0.2, 0.8)))
    if slopes and prof.tail_level_db is not None and prof.tail_level_db.value > -35:
        sl = float(np.median(slopes))
        if sl < -1:
            prof.rt60_s = Measure(round(float(np.clip(-60.0 / sl, 0.1, 8.0)), 2),
                                  float(np.clip(len(slopes) / 10, 0.2, 0.7)))

    _detect_echo(prof, pm, f, frame_db, hop_s)
    if prof.delay_ms is not None and prof.rt60_s is not None:
        # repeats of a delay look like a long tail as well: trust the decay less
        prof.rt60_s.confidence = min(prof.rt60_s.confidence, 0.15)


def _detect_echo(prof: VocalProfile, pm, f, frame_db, hop_s):
    """Discrete delay: an echo repeats each sung onset a fixed time later, with the
    same pitch and vowel (so the fine spectrum matches) but at a lower level.
    Melody notes that happen to repeat do that at varying distances and levels,
    so only a lag that keeps coming back across many onsets counts."""
    m = (f >= 150) & (f <= 5000)
    lp = np.log(pm[m] + 1e-10)
    edges = np.geomspace(150, min(8000, f[-1]), 21)
    bands = np.stack([pm[(f >= lo) & (f < hi)].sum(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    lb = np.log(bands + 1e-10)
    flux = np.maximum(np.diff(lb, axis=1), 0).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    ref = float(np.percentile(frame_db, 95))
    pk, _ = signal.find_peaks(flux, distance=max(1, int(0.06 / hop_s)), prominence=np.percentile(flux, 75))
    probe = int(0.04 / hop_s)
    pk = pk[(pk + probe < len(frame_db))]
    pk = pk[frame_db[pk + probe] > ref - 45]
    if len(pk) < 15:
        return
    look = pk + probe
    spec = lp[:, look]
    spec = spec - spec.mean(axis=0, keepdims=True)
    spec /= np.linalg.norm(spec, axis=0, keepdims=True) + EPS
    lvl = frame_db[look]
    lo_l, hi_l = int(0.08 / hop_s), int(0.9 / hop_s)
    bins = np.zeros(hi_l + 2)
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            L = pk[j] - pk[i]
            if L > hi_l:
                break
            if L < lo_l:
                continue
            diff = lvl[j] - lvl[i]
            if not (-30.0 < diff < -2.0):
                continue
            sim = float(spec[:, i] @ spec[:, j])
            if sim > 0.8:
                bins[L] += sim
    # merge neighbouring lags (onset jitter of one frame)
    sm = np.convolve(bins, [1, 1, 1], "same")
    if sm.max() <= 0:
        return
    share = sm / len(pk)
    # an echo with feedback also shows at 2L, 3L: take the shortest strong lag
    strong = np.flatnonzero(share >= max(0.1, 0.6 * share.max()))
    L = int(strong[0]) if strong.size else int(np.argmax(share))
    # centre of mass around the chosen lag for sub-frame precision
    w = bins[max(0, L - 1):L + 2]
    lags = np.arange(max(0, L - 1), L + 2)[:len(w)]
    Lc = float((w * lags).sum() / w.sum()) if w.sum() > 0 else float(L)
    background = float(np.median(share[lo_l:hi_l + 1]))
    if share[L] >= 0.1 and share[L] > 5 * background + 0.03:
        prof.delay_ms = Measure(round(Lc * hop_s * 1000.0), float(np.clip(share[L] * 3, 0.3, 0.85)))
        prof.delay_strength = round(float(share[L]), 3)
