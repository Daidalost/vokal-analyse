"""Checks the measurements and the advice against synthetic vocals processed
with known settings (dry, EQ, compression, de-essing, reverb, delay, doubling)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import synth as S  # noqa: E402
from vokal_analyse.features import analyse_vocal, estimate_bpm  # noqa: E402
from vokal_analyse.advice import describe, compare, note_value, summary  # noqa: E402

SR = S.SR


@pytest.fixture(scope="module")
def dry():
    return S.sung_vocal()


@pytest.fixture(scope="module")
def music():
    return S.backing_track()


@pytest.fixture(scope="module")
def p_dry(dry, music):
    return analyse_vocal(dry, SR, music)


def areas(advice):
    return {(a.area, a.title) for a in advice}


# ---------------------------------------------------------------- features

def test_dry_vocal_basics(p_dry):
    p = p_dry
    assert p.ok
    assert 150 < p.f0_median.value < 210            # base 140 Hz, melody up to +9 semitones
    assert p.tail_level_db.value < -35              # dry
    assert p.rt60_s is None
    assert p.delay_ms is None
    assert p.side_db.value < -60                    # mono
    assert abs(p.bpm.value - 100) < 2


def test_female_pitch(music):
    p = analyse_vocal(S.sung_vocal(f0_base=280), SR, music)
    assert 300 < p.f0_median.value < 420


@pytest.mark.parametrize("slope", [-2.0, 2.0])
def test_tilt_follows_eq(dry, p_dry, slope):
    p = analyse_vocal(S.tilt_eq(dry, slope), SR)
    d = p.tilt_db_per_oct.value - p_dry.tilt_db_per_oct.value
    assert abs(d - slope) < 0.8


def test_low_boost_shows_in_low_region(dry, p_dry):
    p = analyse_vocal(S.band_gain(dry, 80, 200, 6.0), SR)
    assert p.regions_db["low"] - p_dry.regions_db["low"] > 4.0
    assert abs(p.regions_db["presence"] - p_dry.regions_db["presence"]) < 1.5


def test_compression_reduces_spread(dry, p_dry):
    p = analyse_vocal(S.compress(dry, -32, 8), SR)
    assert p.syllable_range_db.value < p_dry.syllable_range_db.value - 3
    assert p.phrase_range_db.value < p_dry.phrase_range_db.value - 4


def test_sibilance(dry, p_dry):
    hot = analyse_vocal(S.band_gain(dry, 5000, 10000, 6.0), SR)
    soft = analyse_vocal(S.band_gain(dry, 5000, 10000, -6.0), SR)
    assert hot.sib_vs_vowel_db.value > p_dry.sib_vs_vowel_db.value + 2.5
    assert soft.sib_vs_vowel_db.value < p_dry.sib_vs_vowel_db.value - 2.5
    assert 5500 < p_dry.sib_freq_hz.value < 9500


def test_reverb(dry):
    p = analyse_vocal(S.reverb(dry, 2.0, -6.0), SR)
    assert p.tail_level_db.value > -20
    assert 1.2 < p.rt60_s.value < 3.0
    assert p.tail_side_db.value > -6            # decorrelated stereo tail
    assert p.onset_side_db.value < -15          # but the notes start in the centre
    assert p.delay_ms is None


def test_short_reverb_is_shorter(dry):
    long_ = analyse_vocal(S.reverb(dry, 2.5, -6.0), SR)
    short = analyse_vocal(S.reverb(dry, 0.6, -6.0), SR)
    assert short.rt60_s is None or short.rt60_s.value < long_.rt60_s.value


@pytest.mark.parametrize("ms", [300.0, 375.0, 450.0])
def test_delay_time(dry, ms):
    p = analyse_vocal(S.delay(dry, ms), SR)
    assert p.delay_ms is not None
    assert abs(p.delay_ms.value - ms) < 15


def test_doubling(music):
    p = analyse_vocal(S.double_track(), SR, music)
    assert p.onset_side_db.value > -15


def test_vocal_level_vs_music(dry, music, p_dry):
    p = analyse_vocal(dry * 0.5, SR, music)
    assert abs((p_dry.vocal_vs_music_db.value - p.vocal_vs_music_db.value) - 6.0) < 1.0


@pytest.mark.parametrize("bpm", [80, 96, 110, 128, 140])
def test_bpm(bpm):
    m = estimate_bpm(S.backing_track(seconds=30, bpm=bpm), SR)
    assert abs(m.value - bpm) / bpm < 0.02


def test_too_short():
    p = analyse_vocal(np.zeros(SR), SR)
    assert not p.ok and p.error


def test_silence():
    p = analyse_vocal(np.zeros(SR * 5), SR)
    assert not p.ok


# ---------------------------------------------------------------- advice

def test_note_value():
    assert note_value(375, 120) == "dotted 1/8"
    assert note_value(250, 120) == "1/8"
    assert note_value(500, 120) == "1/4"
    assert note_value(333, 120) == "1/4 triplet"
    assert note_value(167, 120) == "1/8 triplet"
    assert note_value(410, 120) is None


def test_describe_reference(dry, music):
    c = S.compress(dry, -32, 8)
    adv = describe(analyse_vocal(S.delay(c, 375), SR, music))
    a = areas(adv)
    assert ("Compression", "Heavy compression") in a
    assert any(x[0] == "Delay" for x in a)
    assert {x.stage for x in adv} == {"recording", "mixing"}

    wet = analyse_vocal(S.reverb(c, 2.0, -8.0), SR, music)
    assert any(x.area == "Reverb" and x.title == "Audible reverb" for x in describe(wet))
    assert summary(wet)["Space"] == "clearly audible reverb"
    assert summary(analyse_vocal(S.delay(c, 375), SR, music))["Space"].startswith("delay echoes")

    dry_ref = analyse_vocal(dry, SR, music)
    assert any(x.area == "Room" and x.title == "Record dry" for x in describe(dry_ref))


@pytest.mark.xfail(reason="known limit: echoes are not found under a loud reverb", strict=False)
def test_delay_under_loud_reverb(dry):
    p = analyse_vocal(S.delay(S.reverb(dry, 2.0, -6.0), 375), SR)
    assert p.delay_ms is not None


def test_compare_finds_known_differences(dry, music):
    ref = analyse_vocal(S.compress(S.tilt_eq(dry, 2.5), -32, 8), SR, music)
    own = analyse_vocal(S.reverb(dry, 1.5, -6.0), SR, music)
    adv = compare(ref, own)
    a = areas(adv)
    assert ("Compression", "More compression") in a
    assert ("Room", "Less room / reverb") in a
    assert ("Microphone & distance", "Capture more top end") in a


def test_compare_same_is_quiet(p_dry):
    adv = compare(p_dry, p_dry)
    assert [x.title for x in adv] == ["Already close"]


def test_compare_level(dry, music, p_dry):
    own = analyse_vocal(dry * 0.5, SR, music)
    adv = compare(p_dry, own)
    assert any(x.title == "Raise the vocal" for x in adv)


def test_compare_delay_and_doubling(dry, music, p_dry):
    ref = analyse_vocal(S.delay(S.double_track(), 375), SR, music)
    adv = compare(ref, p_dry)
    a = areas(adv)
    assert any(x[0] == "Delay" for x in a)
    assert any(x[0] == "Doubling" and x[1] == "Doubled vocal" for x in a)
