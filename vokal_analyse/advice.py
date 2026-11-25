"""Rules that turn measurements into advice.

Two situations:
  * only a reference: describe how that vocal is made and how to get there.
  * reference + your own track: the differences, and what to change.

Every advice carries the measurement it is based on and a confidence, so the
report can be honest about guesses. Thresholds are first estimates and are
kept together in LIMITS so they can be calibrated on real productions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .features import VocalProfile, Measure, REGIONS

LIMITS = {
    "tilt_bright": -3.5,     # dB/oct above this: bright
    "tilt_dark": -6.5,       # below this: dark
    "dry_tail": -35.0,       # tail level re phrase (dB) below this: dry
    "obvious_tail": -20.0,   # above this: clearly audible reverb
    "even_syll": 5.0,        # syllable peak spread (dB) below this: heavily compressed
    "loose_syll": 9.0,       # above this: lightly compressed / natural
    "sib_hot": 0.0,          # S peaks vs vowel peaks (dB): above this the S-sounds are hot
    "sib_soft": -6.0,
    "double_onset_side": -15.0,  # side/mid at onsets above this: doubled / stereo takes
    "wide_tail_side": -6.0,
    "eq_diff": 2.0,          # region difference worth acting on (dB)
    "sib_diff": 2.0,
    "syll_diff": 2.0,
    "tail_diff": 6.0,
    "level_diff": 2.0,
}

STAGE_REC = "recording"
STAGE_MIX = "mixing"


@dataclass
class Advice:
    stage: str          # recording | mixing
    area: str           # e.g. "Microphone & distance", "EQ", "Compression"
    title: str
    text: str
    evidence: str
    confidence: str     # high | medium | low

    def to_dict(self):
        return asdict(self)


def _conf(*measures: Optional[Measure], base: float = 1.0) -> str:
    c = base
    for m in measures:
        if m is not None:
            c = min(c, m.confidence)
    return "high" if c >= 0.7 else "medium" if c >= 0.4 else "low"


def _v(m: Optional[Measure]) -> Optional[float]:
    return None if m is None else float(m.value)


REGION_TEXT = {
    "low": ("80-200 Hz", "chest and proximity effect", 120.0),
    "low_mid": ("200-500 Hz", "warmth, or boxiness when too much", 300.0),
    "mid": ("500 Hz-2 kHz", "body of the vowels, nasal or honky when too much", 1000.0),
    "presence": ("2-5 kHz", "presence and clarity, harsh when too much", 3200.0),
    "sibilance": ("5-10 kHz", "S-sounds and edge", 7000.0),
    "air": ("10-16 kHz", "air and sheen", 12000.0),
}

NOTE_VALUES = [  # (name, length in beats)
    ("1/16", 0.25), ("1/8 triplet", 1 / 3), ("1/8", 0.5), ("dotted 1/8", 0.75),
    ("1/4 triplet", 2 / 3), ("1/4", 1.0), ("dotted 1/4", 1.5), ("1/2", 2.0),
]


def note_value(ms: float, bpm: Optional[float]) -> Optional[str]:
    if not bpm or bpm <= 0:
        return None
    beat = 60000.0 / bpm
    best = min(NOTE_VALUES, key=lambda nv: abs(ms - nv[1] * beat) / (nv[1] * beat))
    if abs(ms - best[1] * beat) / (best[1] * beat) <= 0.06:
        return best[0]
    return None


# ------------------------------------------------------------ descriptions

def tone_words(p: VocalProfile) -> str:
    t = _v(p.tilt_db_per_oct)
    if t is None:
        return "unknown"
    if t > LIMITS["tilt_bright"]:
        return "bright"
    if t < LIMITS["tilt_dark"]:
        return "dark/warm"
    return "balanced"


def compression_words(p: VocalProfile) -> str:
    s = _v(p.syllable_range_db)
    if s is None:
        return "unknown"
    if s < LIMITS["even_syll"]:
        return "heavily compressed, very even"
    if s > LIMITS["loose_syll"]:
        return "light compression, natural dynamics"
    return "medium compression"


def space_words(p: VocalProfile) -> str:
    tl = _v(p.tail_level_db)
    if tl is None:
        return "unknown"
    if tl < LIMITS["dry_tail"]:
        return "dry"
    if tl < LIMITS["obvious_tail"]:
        return "a little ambience"
    return "clearly audible reverb"


def summary(p: VocalProfile) -> dict:
    """Short facts about one vocal, for the report header."""
    bpm = _v(p.bpm)
    out = {
        "Tone": tone_words(p),
        "Dynamics": compression_words(p),
        "Space": space_words(p),
        "Pitch (median)": f"{p.f0_median.value:.0f} Hz" if p.f0_median else "unknown",
        "Tempo": f"{bpm:.0f} BPM" if bpm else "unknown",
    }
    if p.delay_ms is not None:
        nv = note_value(p.delay_ms.value, bpm)
        out["Delay"] = f"{p.delay_ms.value:.0f} ms" + (f" ({nv} note)" if nv else "")
    if p.onset_side_db is not None and p.onset_side_db.value > LIMITS["double_onset_side"]:
        out["Width"] = "doubled / wide"
    elif p.side_db is not None and p.side_db.value < -30:
        out["Width"] = "mono"
    else:
        out["Width"] = "mono lead with stereo effects"
    if p.vocal_vs_music_db is not None:
        out["Vocal vs music"] = f"{p.vocal_vs_music_db.value:+.1f} dB"
    return out


# ------------------------------------------------------------ reference only

def describe(ref: VocalProfile) -> list[Advice]:
    """How to get the reference sound, without a track of your own."""
    adv: list[Advice] = []
    t = _v(ref.tilt_db_per_oct)
    reg = ref.regions_db

    # --- recording: mic and distance from tone + low end
    if t is not None:
        low = reg.get("low", -99) - reg.get("mid", 0)
        f0 = _v(ref.f0_median) or 0
        lowish = low > 3 and f0 and f0 < 220
        if t > LIMITS["tilt_bright"]:
            adv.append(Advice(STAGE_REC, "Microphone & distance", "Bright, open capture",
                "Use a condenser microphone (large diaphragm) at about 15-25 cm. A pop filter keeps plosives down at that distance. "
                "Brightness from the microphone sounds smoother than adding a lot of top end with EQ later.",
                f"Spectral tilt {t:.1f} dB/oct (bright).", _conf(ref.tilt_db_per_oct)))
        elif t < LIMITS["tilt_dark"]:
            adv.append(Advice(STAGE_REC, "Microphone & distance", "Warm, close capture",
                "A dynamic microphone (SM7B type) or a ribbon close to the mouth (5-10 cm) gives this dark, warm tone. "
                "The proximity effect adds chest; sing slightly across the capsule to avoid pops.",
                f"Spectral tilt {t:.1f} dB/oct (dark).", _conf(ref.tilt_db_per_oct)))
        else:
            adv.append(Advice(STAGE_REC, "Microphone & distance", "Balanced capture",
                "A large-diaphragm condenser at about 10-20 cm gives this balanced tone. Move closer for more chest, further back for more air.",
                f"Spectral tilt {t:.1f} dB/oct (balanced).", _conf(ref.tilt_db_per_oct)))
        if lowish:
            adv.append(Advice(STAGE_REC, "Microphone & distance", "Use the proximity effect",
                "The reference has a lot of chest (80-200 Hz) compared to the mids. Sing close to a cardioid microphone and keep the distance steady, "
                "because the low end changes a lot with small movements.",
                f"80-200 Hz is {low:+.1f} dB re 500 Hz-2 kHz.", _conf(ref.tilt_db_per_oct, base=0.6)))

    # --- room
    tl = _v(ref.tail_level_db)
    if tl is not None and tl < LIMITS["dry_tail"]:
        adv.append(Advice(STAGE_REC, "Room", "Record dry",
            "The reference is dry and up front. Record in a dead space: a closet with clothes, a reflection filter plus blankets behind you, "
            "or acoustic panels around the microphone. Any room sound in the recording cannot be removed later.",
            f"Tail level after phrases {tl:.0f} dB.", _conf(ref.tail_level_db)))
    else:
        adv.append(Advice(STAGE_REC, "Room", "Record dry, add space in the mix",
            "Even though the reference has audible space, that is almost always added in the mix. Record as dry as you can, so you control the reverb.",
            f"Tail level after phrases {tl:.0f} dB." if tl is not None else "No clear phrase endings found.", "medium"))

    # --- performance
    s = _v(ref.syllable_range_db)
    if s is not None and s < LIMITS["even_syll"]:
        adv.append(Advice(STAGE_REC, "Performance", "Sing evenly",
            "The vocal is very even. Heavy compression does a lot of that, but it sounds more natural when the performance is already steady: "
            "keep the same distance to the microphone and lean back a little on the loudest notes.",
            f"Syllable level spread {s:.1f} dB.", _conf(ref.syllable_range_db)))
    hs = _v(ref.harmonic_share)
    if hs is not None and hs < 0.3 and (tl is None or tl < LIMITS["obvious_tail"]):
        adv.append(Advice(STAGE_REC, "Performance", "Breathy or gritty delivery",
            "There is a lot of noise between the harmonics (breath, rasp or distortion). If it is breath: sing softer and closer to the microphone. "
            "If it is grit: see Saturation below.",
            f"Harmonic share 1-4 kHz {hs:.2f}.", "low"))

    # --- EQ shape
    if reg:
        notes = []
        for name in ("low", "low_mid", "presence", "air"):
            if name in reg:
                notes.append(f"{REGION_TEXT[name][0]}: {reg[name]:+.1f} dB")
        adv.append(Advice(STAGE_MIX, "EQ", f"Tone: {tone_words(ref)}",
            "Use the reference's tone curve as the target. Load your own vocal as well to get exact EQ moves; without your own track the shape is: "
            + "; ".join(notes) + " (relative to the 250 Hz-4 kHz average).",
            f"Spectral tilt {t:.1f} dB/oct." if t is not None else "", _conf(ref.tilt_db_per_oct, base=0.6)))

    adv += _compression_abs(ref)
    adv += _deess_abs(ref)
    adv += _saturation_abs(ref)
    adv += _space_abs(ref)
    adv += _level_abs(ref)
    return adv


def _compression_abs(p: VocalProfile) -> list[Advice]:
    s = _v(p.syllable_range_db)
    if s is None:
        return []
    if s < LIMITS["even_syll"]:
        txt = ("Two compressors in series: a fast one (FET/1176 style, 4:1, attack 3-10 ms, release ~50 ms) catching peaks with 4-8 dB gain reduction, "
               "then a smooth one (opto/LA-2A style) with 2-4 dB. Ride the level with automation where single words still jump out.")
        t = "Heavy compression"
    elif s < LIMITS["loose_syll"]:
        txt = "One compressor, 3:1 to 4:1, medium attack (10-20 ms), auto or ~100 ms release, about 3-6 dB gain reduction on the loud parts."
        t = "Medium compression"
    else:
        txt = "Gentle compression: 2:1, slow attack (20-30 ms), 1-3 dB gain reduction. Let the dynamics of the performance through."
        t = "Light compression"
    return [Advice(STAGE_MIX, "Compression", t, txt, f"Syllable level spread {s:.1f} dB.", _conf(p.syllable_range_db))]


def _deess_abs(p: VocalProfile) -> list[Advice]:
    sv = _v(p.sib_vs_vowel_db)
    if sv is None:
        return []
    fz = _v(p.sib_freq_hz) or 7000
    if sv > LIMITS["sib_hot"]:
        return [Advice(STAGE_MIX, "De-esser", "Bright S-sounds are part of the sound",
            f"The S-sounds are as loud as the vowels. This sound uses a lot of top end; put a de-esser around {fz/1000:.1f} kHz after the EQ boost "
            "so the S-sounds stay crisp without hurting.", f"S peaks {sv:+.1f} dB re vowel peaks.", _conf(p.sib_vs_vowel_db))]
    if sv < LIMITS["sib_soft"]:
        return [Advice(STAGE_MIX, "De-esser", "Soft S-sounds",
            f"The S-sounds are well tamed. Use a de-esser around {fz/1000:.1f} kHz with 4-8 dB reduction, or record slightly off-axis.",
            f"S peaks {sv:+.1f} dB re vowel peaks.", _conf(p.sib_vs_vowel_db))]
    return [Advice(STAGE_MIX, "De-esser", "Moderate de-essing",
        f"A de-esser around {fz/1000:.1f} kHz with 2-5 dB reduction on the S-sounds.",
        f"S peaks {sv:+.1f} dB re vowel peaks.", _conf(p.sib_vs_vowel_db))]


def _saturation_abs(p: VocalProfile) -> list[Advice]:
    hs = _v(p.harmonic_share)
    if hs is None:
        return []
    wet = _v(p.tail_level_db) is not None and p.tail_level_db.value > LIMITS["obvious_tail"]
    if hs < 0.3 and not wet:
        return [Advice(STAGE_MIX, "Saturation", "Some grit or density",
            "The vocal is dense between the harmonics. Try tape or tube saturation (parallel, 20-40 % mix) after the compressor. "
            "This measurement is uncertain: breathiness gives the same reading.", f"Harmonic share {hs:.2f}.", "low")]
    if hs > 0.55:
        return [Advice(STAGE_MIX, "Saturation", "Clean",
            "The vocal sounds clean. If you use saturation, keep it subtle (for density, not audible distortion).",
            f"Harmonic share {hs:.2f}.", "low")]
    return []


def _space_abs(p: VocalProfile) -> list[Advice]:
    adv = []
    bpm = _v(p.bpm)
    tl = _v(p.tail_level_db)
    rt = _v(p.rt60_s)
    ts = _v(p.tail_side_db)
    if tl is not None:
        if tl < LIMITS["dry_tail"]:
            adv.append(Advice(STAGE_MIX, "Reverb", "Little or no reverb",
                "The vocal is dry and intimate. If you use reverb, keep it very low or short (a small room or short plate under 1 s) so it is felt rather than heard.",
                f"Tail level {tl:.0f} dB.", _conf(p.tail_level_db)))
        else:
            kind = "a room or short plate" if (rt or 0) < 0.9 else "a plate" if (rt or 0) < 1.8 else "a hall or long plate"
            pre = ""
            if bpm:
                pre = f" Pre-delay about {60000 / bpm / 16:.0f} ms (a 1/64 note at {bpm:.0f} BPM) keeps the words clear."
            width = " The tail is wide (stereo)." if ts is not None and ts > LIMITS["wide_tail_side"] else ""
            rt_txt = f"decay about {rt:.1f} s" if rt else "decay not measurable"
            adv.append(Advice(STAGE_MIX, "Reverb", "Audible reverb",
                f"Use {kind} ({rt_txt}) on a send.{pre}{width} Cut the reverb return below ~300 Hz and above ~8 kHz so it does not muddy the vocal.",
                f"Tail level {tl:.0f} dB" + (f", decay {rt:.1f} s" if rt else "") + ".", _conf(p.tail_level_db, p.rt60_s)))
    if p.delay_ms is not None:
        ms = p.delay_ms.value
        nv = note_value(ms, bpm)
        if ms < 150:
            txt = f"A slapback delay of about {ms:.0f} ms, one repeat, mixed low. Typical for rock'n'roll and some pop/rap."
        else:
            txt = (f"A delay of about {ms:.0f} ms" + (f", a {nv} note at {bpm:.0f} BPM" if nv else "") +
                   ", with a few repeats (feedback 20-35 %). Filter the delay return (high-pass ~300 Hz, low-pass ~6 kHz).")
        adv.append(Advice(STAGE_MIX, "Delay", "Echo on the vocal", txt, f"Repeats found at {ms:.0f} ms.", _conf(p.delay_ms)))
    os_ = _v(p.onset_side_db)
    if os_ is not None and os_ > LIMITS["double_onset_side"]:
        adv.append(Advice(STAGE_REC, "Doubling", "Doubled vocal",
            "The vocal is wide already at the start of the notes, which points to extra takes panned left and right. Record 2 extra takes of the same part, "
            "tight to the lead, and pan them hard left/right 6-10 dB below the lead.",
            f"Side/mid at note starts {os_:.0f} dB.", _conf(p.onset_side_db)))
    elif p.side_db is not None and p.side_db.value < -30:
        adv.append(Advice(STAGE_MIX, "Doubling", "Mono lead",
            "The lead vocal is mono and centred with no doubles or stereo effects.", f"Side/mid {p.side_db.value:.0f} dB.", _conf(p.side_db)))
    return adv


def _level_abs(p: VocalProfile) -> list[Advice]:
    d = _v(p.vocal_vs_music_db)
    if d is None:
        return []
    if d > 2:
        t, txt = "Vocal on top", "The vocal sits clearly above the music (typical for pop and rap)."
    elif d < -3:
        t, txt = "Vocal tucked in", "The vocal sits inside the music (typical for rock and some indie)."
    else:
        t, txt = "Vocal level with the music", "The vocal and the music are about equally loud while singing."
    return [Advice(STAGE_MIX, "Level in the mix", t, txt + " Separation leaves some bleed, so take the number as a guide.",
                   f"Vocal vs music {d:+.1f} dB (K-weighted, while singing).", _conf(p.vocal_vs_music_db, base=0.6))]


# ------------------------------------------------------------ comparison

def compare(ref: VocalProfile, own: VocalProfile) -> list[Advice]:
    """What to change in your own vocal to get closer to the reference."""
    adv: list[Advice] = []

    # --- EQ, region by region
    for name, (lo, hi) in REGIONS.items():
        if name not in ref.regions_db or name not in own.regions_db:
            continue
        d = ref.regions_db[name] - own.regions_db[name]
        if abs(d) < LIMITS["eq_diff"]:
            continue
        rng, what, fc = REGION_TEXT[name]
        g = max(-6.0, min(6.0, 0.8 * d))
        if name == "sibilance" and d < 0:
            continue  # handled by the de-esser advice
        kind = "high shelf" if name == "air" else "low shelf or bell" if name == "low" else "wide bell (Q ~1)"
        verb = "Boost" if g > 0 else "Cut"
        adv.append(Advice(STAGE_MIX, "EQ", f"{verb} {rng}",
            f"{verb} about {abs(g):.1f} dB around {fc/1000:.1f} kHz ({kind}). Your vocal has {'less' if d > 0 else 'more'} {what} than the reference."
            if fc >= 1000 else
            f"{verb} about {abs(g):.1f} dB around {fc:.0f} Hz ({kind}). Your vocal has {'less' if d > 0 else 'more'} {what} than the reference.",
            f"{rng}: reference {ref.regions_db[name]:+.1f} dB, yours {own.regions_db[name]:+.1f} dB.",
            _conf(ref.tilt_db_per_oct, own.tilt_db_per_oct, base=0.8)))

    # --- recording hints from the tone difference
    tr, to = _v(ref.tilt_db_per_oct), _v(own.tilt_db_per_oct)
    if tr is not None and to is not None and abs(tr - to) > 1.5:
        if tr > to:
            adv.append(Advice(STAGE_REC, "Microphone & distance", "Capture more top end",
                "Your recording is darker than the reference overall. Before using a lot of EQ: try a brighter condenser microphone, "
                "or move back a little (15-25 cm) so the proximity effect does not thicken the low end.",
                f"Spectral tilt: reference {tr:.1f}, yours {to:.1f} dB/oct.", _conf(ref.tilt_db_per_oct)))
        else:
            adv.append(Advice(STAGE_REC, "Microphone & distance", "Capture a warmer tone",
                "Your recording is brighter than the reference overall. Try a darker microphone (dynamic or ribbon) or sing closer (5-10 cm) for more proximity effect.",
                f"Spectral tilt: reference {tr:.1f}, yours {to:.1f} dB/oct.", _conf(ref.tilt_db_per_oct)))

    # --- compression
    sr_, so = _v(ref.syllable_range_db), _v(own.syllable_range_db)
    if sr_ is not None and so is not None:
        d = so - sr_
        if d > LIMITS["syll_diff"]:
            adv.append(Advice(STAGE_MIX, "Compression", "More compression",
                f"Your syllables vary about {d:.0f} dB more than in the reference. Add roughly {min(10, 0.8 * d):.0f} dB more gain reduction on the loud parts "
                "(ratio 4:1, attack 5-15 ms), or split it over two compressors for a natural sound.",
                f"Syllable level spread: reference {sr_:.1f} dB, yours {so:.1f} dB.", _conf(ref.syllable_range_db, own.syllable_range_db)))
        elif d < -LIMITS["syll_diff"]:
            adv.append(Advice(STAGE_MIX, "Compression", "Less compression",
                f"Your vocal is about {-d:.0f} dB more squashed than the reference. Lower the ratio or raise the threshold so more of the performance's dynamics come through.",
                f"Syllable level spread: reference {sr_:.1f} dB, yours {so:.1f} dB.", _conf(ref.syllable_range_db, own.syllable_range_db)))
    else:
        adv += _compression_abs(ref)

    # --- de-esser
    sr2, so2 = _v(ref.sib_vs_vowel_db), _v(own.sib_vs_vowel_db)
    if sr2 is not None and so2 is not None:
        d = so2 - sr2
        fz = _v(own.sib_freq_hz) or 7000
        if d > LIMITS["sib_diff"]:
            adv.append(Advice(STAGE_MIX, "De-esser", "Tame the S-sounds",
                f"Your S-sounds are about {d:.0f} dB hotter (relative to the vowels) than the reference. De-ess around {fz/1000:.1f} kHz with about {d:.0f} dB reduction. "
                "When recording: a pop filter and angling the microphone slightly off the mouth also help.",
                f"S peaks re vowel peaks: reference {sr2:+.1f} dB, yours {so2:+.1f} dB.", _conf(ref.sib_vs_vowel_db, own.sib_vs_vowel_db)))
        elif d < -LIMITS["sib_diff"]:
            adv.append(Advice(STAGE_MIX, "De-esser", "Less de-essing",
                f"Your S-sounds are about {-d:.0f} dB softer than the reference. Use less de-essing, or add a high shelf above 8 kHz and de-ess only the worst S-sounds.",
                f"S peaks re vowel peaks: reference {sr2:+.1f} dB, yours {so2:+.1f} dB.", _conf(ref.sib_vs_vowel_db, own.sib_vs_vowel_db)))

    # --- saturation
    hr, ho = _v(ref.harmonic_share), _v(own.harmonic_share)
    if hr is not None and ho is not None and ho - hr > 0.15:
        adv.append(Advice(STAGE_MIX, "Saturation", "Add density",
            "The reference has more energy between the harmonics than yours. Try tape or tube saturation in parallel. "
            "Uncertain: a breathier performance or more reverb in the reference reads the same way.",
            f"Harmonic share: reference {hr:.2f}, yours {ho:.2f}.", "low"))

    # --- room / reverb
    tr_, to_ = _v(ref.tail_level_db), _v(own.tail_level_db)
    if tr_ is not None and to_ is not None:
        d = tr_ - to_
        if d > LIMITS["tail_diff"]:
            adv += [a for a in _space_abs(ref) if a.area == "Reverb"]
        elif d < -LIMITS["tail_diff"]:
            adv.append(Advice(STAGE_REC, "Room", "Less room / reverb",
                "Your vocal has more space after the phrases than the reference. If it is reverb in the mix, turn it down or shorten it. "
                "If it is the recording room, treat the room or move closer to the microphone next time.",
                f"Tail level: reference {tr_:.0f} dB, yours {to_:.0f} dB.", _conf(ref.tail_level_db, own.tail_level_db)))

    # --- delay
    if ref.delay_ms is not None and own.delay_ms is None:
        adv += [a for a in _space_abs(ref) if a.area == "Delay"]
    elif ref.delay_ms is None and own.delay_ms is not None:
        adv.append(Advice(STAGE_MIX, "Delay", "Remove or hide the delay",
            "The reference has no clear echo. Turn your delay down so it is felt rather than heard, or remove it.",
            f"Your repeats at {own.delay_ms.value:.0f} ms.", _conf(own.delay_ms)))

    # --- doubling
    dr, do = _v(ref.onset_side_db), _v(own.onset_side_db)
    lim = LIMITS["double_onset_side"]
    if dr is not None and dr > lim and (do is None or do <= lim):
        adv += [a for a in _space_abs(ref) if a.area == "Doubling"]
    elif do is not None and do > lim and (dr is None or dr <= lim):
        adv.append(Advice(STAGE_MIX, "Doubling", "Narrower lead",
            "The reference lead is centred without wide doubles. Turn your doubles down or bring them towards the centre.",
            f"Side/mid at note starts: reference {dr if dr is not None else -120:.0f} dB, yours {do:.0f} dB.", _conf(own.onset_side_db)))

    # --- level in the mix
    lr, lo = _v(ref.vocal_vs_music_db), _v(own.vocal_vs_music_db)
    if lr is not None and lo is not None and abs(lr - lo) > LIMITS["level_diff"]:
        d = lr - lo
        adv.append(Advice(STAGE_MIX, "Level in the mix", "Raise the vocal" if d > 0 else "Lower the vocal",
            f"{'Raise' if d > 0 else 'Lower'} the vocal about {abs(d):.0f} dB against the music to match the reference balance.",
            f"Vocal vs music: reference {lr:+.1f} dB, yours {lo:+.1f} dB.", _conf(ref.vocal_vs_music_db, own.vocal_vs_music_db, base=0.6)))

    if not adv:
        adv.append(Advice(STAGE_MIX, "Overall", "Already close",
            "No clear differences were found between your vocal and the reference.", "", "medium"))
    return adv
