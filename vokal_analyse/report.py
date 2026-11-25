"""Self-contained HTML report (no external files except the optional stems)."""
from __future__ import annotations

import datetime as _dt
import html
import math
from typing import Optional

from .features import VocalProfile
from .advice import Advice, summary

CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--ref:#c2410c;--own:#1d4ed8;
--high:#15803d;--medium:#b45309;--low:#6b7280;--chip:#f0eee9}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e21;--ink:#ececef;--muted:#a1a1a8;--line:#2e2e33;
--ref:#fb923c;--own:#60a5fa;--high:#4ade80;--medium:#fbbf24;--low:#9ca3af;--chip:#2a2a2e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 12px}
.sub{color:var(--muted);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;width:42%}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;background:var(--chip);color:var(--muted);margin-right:6px}
.conf-high{color:var(--high)}.conf-medium{color:var(--medium)}.conf-low{color:var(--low)}
.adv{margin-bottom:12px}.adv h3{font-size:16px;margin:4px 0 6px}.adv p{margin:0 0 6px}
.ev{font-size:13px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:13px;color:var(--muted)}
.legend i{width:18px;height:3px;border-radius:2px;display:inline-block}
svg text{fill:var(--muted);font-size:11px}
audio{width:100%;margin-top:6px}
.note{font-size:13px;color:var(--muted);margin-top:28px}
"""


def _e(s) -> str:
    return html.escape(str(s))


def _summary_table(title: str, p: VocalProfile, colour: str) -> str:
    rows = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in summary(p).items())
    return (f'<div class="card"><div class="tag" style="color:{colour}">{_e(title)}</div>'
            f"<table>{rows}</table></div>")


def _tone_chart(ref: VocalProfile, own: Optional[VocalProfile]) -> str:
    W, H, L, R, T, B = 900, 300, 44, 12, 12, 30
    fmin, fmax = 80.0, 16000.0
    lo, hi = -30.0, 20.0

    def x(f):
        return L + (math.log10(f) - math.log10(fmin)) / (math.log10(fmax) - math.log10(fmin)) * (W - L - R)

    def y(d):
        d = max(lo, min(hi, d))
        return T + (hi - d) / (hi - lo) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Tone curves">']
    for d in range(int(lo), int(hi) + 1, 10):
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{y(d):.1f}" y2="{y(d):.1f}" stroke="var(--line)"/>'
                     f'<text x="{L-6}" y="{y(d)+4:.1f}" text-anchor="end">{d:+d}</text>')
    for f, lab in [(100, "100"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"), (5000, "5k"), (10000, "10k")]:
        parts.append(f'<line x1="{x(f):.1f}" x2="{x(f):.1f}" y1="{T}" y2="{H-B}" stroke="var(--line)"/>'
                     f'<text x="{x(f):.1f}" y="{H-B+16}" text-anchor="middle">{lab}</text>')

    def line(p, colour):
        pts = [(x(f), y(d)) for f, d in zip(p.third_oct_hz, p.third_oct_db) if fmin <= f <= fmax]
        if not pts:
            return ""
        path = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        return f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>'

    parts.append(line(ref, "var(--ref)"))
    if own is not None:
        parts.append(line(own, "var(--own)"))
    parts.append("</svg>")
    legend = '<div class="legend"><span><i style="background:var(--ref)"></i>Reference</span>'
    if own is not None:
        legend += '<span><i style="background:var(--own)"></i>Your vocal</span>'
    legend += "</div>"
    return legend + "".join(parts)


def _advice_cards(items: list[Advice]) -> str:
    out = []
    for a in items:
        out.append(
            f'<div class="card adv"><span class="tag">{_e(a.area)}</span>'
            f'<span class="tag conf-{_e(a.confidence)}">{_e(a.confidence)} confidence</span>'
            f"<h3>{_e(a.title)}</h3><p>{_e(a.text)}</p>"
            + (f'<div class="ev">Based on: {_e(a.evidence)}</div>' if a.evidence else "")
            + "</div>")
    return "".join(out) or '<p class="sub">Nothing to add here.</p>'


def build_report(ref_name: str, ref: VocalProfile, advice: list[Advice],
                 own_name: Optional[str] = None, own: Optional[VocalProfile] = None,
                 stems: Optional[dict] = None) -> str:
    """stems: optional {label: relative path to a wav} for listening checks."""
    mode = "Comparison" if own is not None else "Reference"
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = (f"<h1>Vocal sound: {_e(ref_name)}</h1>"
            f'<p class="sub">{_e(mode)} report · {when}'
            + (f" · your track: {_e(own_name)}" if own_name else "") + "</p>")
    cards = _summary_table("Reference", ref, "var(--ref)")
    if own is not None:
        cards += _summary_table("Your vocal", own, "var(--own)")
    rec = [a for a in advice if a.stage == "recording"]
    mix = [a for a in advice if a.stage == "mixing"]
    stem_html = ""
    if stems:
        stem_html = '<h2>Separated stems</h2><p class="sub">Listen to check the separation. Bleed from the music makes the measurements less certain.</p><div class="grid">' + "".join(
            f'<div class="card"><div class="tag">{_e(k)}</div><audio controls preload="none" src="{_e(v)}"></audio></div>'
            for k, v in stems.items()) + "</div>"
    body = (head + f'<div class="grid">{cards}</div>'
            + '<h2>Tone curve</h2><div class="card">' + _tone_chart(ref, own)
            + '<p class="ev">Level per third octave of the sung vowels, relative to the 250 Hz-4 kHz average (dB).</p></div>'
            + "<h2>Recording</h2>" + _advice_cards(rec)
            + "<h2>Mixing and effects</h2>" + _advice_cards(mix)
            + stem_html
            + '<p class="note">The vocal is separated from the music by an AI model, so a little of the music remains in the measurements. '
              "Tone, S-sounds and level are measured fairly precisely. Compression, saturation and the room are informed estimates. "
              "An echo hidden under a loud reverb may not be found.</p>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Vocal sound report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")
