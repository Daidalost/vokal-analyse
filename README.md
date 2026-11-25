# Vokal Analyse

Analyses a fully mixed track (a reference, your own song, or both) and gives
concrete advice on how to get that vocal sound: recording (microphone,
distance, room, performance) and effects (EQ, compression, de-esser,
saturation, reverb, delay, doubling).

Runs locally on your PC. The report opens in your browser.

## How it works

1. The vocal is separated from the music (Mel-Band RoFormer, Demucs as fallback).
2. The vocal stem is measured: tone curve, pitch, syllable dynamics, S-sounds,
   reverb tail, echoes, stereo width / doubling, tempo and vocal level against the music.
3. Rules turn the measurements into advice. Every piece of advice shows the
   measurement behind it and how sure it is.

## Status

- [x] Measurements (`vokal_analyse/features.py`)
- [x] Advice rules (`vokal_analyse/advice.py`)
- [x] Tests on synthetic vocals with known processing (`tests/`)
- [ ] Stem separation
- [ ] Browser report + one-click start on Windows

## Development

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt pytest
    .venv/bin/python -m pytest -q tests
