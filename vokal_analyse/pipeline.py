"""Track in, report out."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .features import analyse_vocal, VocalProfile
from .advice import describe, compare
from .report import build_report


def load_audio(path) -> tuple[np.ndarray, float]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    return x, float(sr)


def analyse_stems(vocal_path, music_path=None) -> VocalProfile:
    v, sr = load_audio(vocal_path)
    m = None
    if music_path is not None:
        m, sr_m = load_audio(music_path)
        if sr_m != sr:
            raise ValueError("Vocal and music stems have different sample rates.")
    return analyse_vocal(v, sr, m)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:60] or "track"


def run(ref_path, own_path=None, out_dir="output", separator=None,
        ref_stems: Optional[tuple] = None, own_stems: Optional[tuple] = None,
        log: Callable[[str], None] = print) -> Path:
    """Analyse the reference (and optionally your own track) and write an HTML report.

    ref_stems / own_stems: (vocal, music) files to use instead of separating.
    Returns the path of the report."""
    out = Path(out_dir)
    ref_path = Path(ref_path)
    name = _slug(ref_path.stem) + ("_vs_" + _slug(Path(own_path).stem) if own_path else "")
    rdir = out / name
    rdir.mkdir(parents=True, exist_ok=True)

    def stems_for(path, given, label):
        if given is not None:
            return Path(given[0]), (Path(given[1]) if len(given) > 1 and given[1] else None)
        if separator is None:
            raise RuntimeError("No separator available and no stems given.")
        log(f"{label}: separating {Path(path).name}")
        return separator.separate(path)

    stems = {}
    rv, rm = stems_for(ref_path, ref_stems, "Reference")
    log("Reference: measuring the vocal...")
    ref = analyse_stems(rv, rm)
    if not ref.ok:
        raise RuntimeError(f"Reference: {ref.error}")
    shutil.copy(rv, rdir / "reference_vocal.wav")
    stems["Reference vocal"] = "reference_vocal.wav"

    own = None
    if own_path is not None:
        ov, om = stems_for(own_path, own_stems, "Your track")
        log("Your track: measuring the vocal...")
        own = analyse_stems(ov, om)
        if not own.ok:
            raise RuntimeError(f"Your track: {own.error}")
        shutil.copy(ov, rdir / "your_vocal.wav")
        stems["Your vocal"] = "your_vocal.wav"

    advice = compare(ref, own) if own is not None else describe(ref)
    html = build_report(ref_path.name, ref, advice, Path(own_path).name if own_path else None, own, stems)
    report = rdir / "report.html"
    report.write_text(html, encoding="utf-8")
    (rdir / "measurements.json").write_text(json.dumps(
        {"reference": ref.to_dict(), "own": own.to_dict() if own else None,
         "advice": [a.to_dict() for a in advice]}, indent=1, default=float), encoding="utf-8")
    log(f"Report written: {report}")
    return report
