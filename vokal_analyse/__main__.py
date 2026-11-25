"""Command line / launcher.

    python -m vokal_analyse                      -> window to pick the tracks
    python -m vokal_analyse REF [--own OWN]      -> analyse and open the report
    python -m vokal_analyse REF --ref-stems VOCAL MUSIC   (skip separation, for testing)
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vokal_analyse", description="Vocal sound analysis of full tracks.")
    ap.add_argument("reference", nargs="?", help="reference track (wav/flac/mp3)")
    ap.add_argument("--own", help="your own track to compare with the reference")
    ap.add_argument("--ref-stems", nargs="+", metavar="WAV", help="vocal [music] stems of the reference (no separation)")
    ap.add_argument("--own-stems", nargs="+", metavar="WAV", help="vocal [music] stems of your track (no separation)")
    ap.add_argument("--out", default=None, help="output folder for reports")
    ap.add_argument("--no-open", action="store_true", help="do not open the report in the browser")
    a = ap.parse_args(argv)

    if a.reference is None:
        from .gui import run_gui
        return run_gui()

    from .pipeline import run
    sep = None
    if not a.ref_stems or (a.own and not a.own_stems):
        from .separation import StemSeparator
        sep = StemSeparator(paths.models_dir(), paths.cache_dir())
    report = run(a.reference, a.own, a.out or paths.reports_dir(), sep,
                 tuple(a.ref_stems) if a.ref_stems else None,
                 tuple(a.own_stems) if a.own_stems else None)
    if not a.no_open:
        webbrowser.open(Path(report).resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
