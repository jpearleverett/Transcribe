"""Export a diarized transcript to the formats people actually need.

Timestamp formats are the classic source of silent breakage: SRT uses a comma
before milliseconds and 1-based cue numbering, WebVTT uses a period and
requires the WEBVTT header, and both need HH:MM:SS even past 24 hours.
"""

from __future__ import annotations

import json
from typing import Optional


def _clock(seconds: float, sep: str, *, always_hours: bool = True) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    ms = ms_total % 1000
    total = ms_total // 1000
    s, m, h = total % 60, (total // 60) % 60, total // 3600
    if always_hours or h:
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
    return f"{m:02d}:{s:02d}{sep}{ms:03d}"


def srt_time(seconds: float) -> str:
    return _clock(seconds, ",")


def vtt_time(seconds: float) -> str:
    return _clock(seconds, ".")


def hms(seconds: float) -> str:
    """Human-readable [HH:]MM:SS for the reading view and text export."""
    total = int(seconds)
    s, m, h = total % 60, (total // 60) % 60, total // 3600
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _label(speaker: Optional[str], names: Optional[dict]) -> str:
    if speaker is None:
        return "Speaker"
    if names and speaker in names and names[speaker]:
        return names[speaker]
    # Engines hand us "A"/"0"/"SPEAKER_00"; normalise to something readable.
    s = str(speaker)
    if s.upper().startswith("SPEAKER_"):
        s = s.split("_", 1)[1].lstrip("0") or "0"
    return f"Speaker {s}"


def to_srt(segments: list, names: Optional[dict] = None, *, with_speaker: bool = True) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        text = seg.text
        if with_speaker and seg.speaker is not None:
            text = f"[{_label(seg.speaker, names)}] {text}"
        out.append(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n")
    return "\n".join(out)


def to_vtt(segments: list, names: Optional[dict] = None, *, with_speaker: bool = True) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        text = seg.text
        if with_speaker and seg.speaker is not None:
            # <v Name> is the standard WebVTT voice span; players that don't
            # understand it still render the text.
            text = f"<v {_label(seg.speaker, names)}>{text}"
        out.append(f"{vtt_time(seg.start)} --> {vtt_time(seg.end)}")
        out.append(text)
        out.append("")
    return "\n".join(out)


def to_text(segments: list, names: Optional[dict] = None, *, timestamps: bool = True) -> str:
    """Speaker-blocked plain text — the format people paste into notes."""
    lines = []
    last_speaker = object()
    for seg in segments:
        if seg.speaker != last_speaker:
            if lines:
                lines.append("")
            head = _label(seg.speaker, names)
            if timestamps:
                head += f"  [{hms(seg.start)}]"
            lines.append(head)
            last_speaker = seg.speaker
        lines.append(seg.text)
    return "\n".join(lines) + "\n"


def to_markdown(segments: list, names: Optional[dict] = None, *, meta: Optional[dict] = None) -> str:
    lines = []
    if meta:
        lines.append(f"# {meta.get('name', 'Transcript')}")
        lines.append("")
        bits = []
        if meta.get("duration"):
            bits.append(f"**Duration:** {hms(meta['duration'])}")
        if meta.get("engine"):
            bits.append(f"**Engine:** {meta['engine']}")
        if meta.get("model"):
            bits.append(f"**Model:** `{meta['model']}`")
        if meta.get("language"):
            bits.append(f"**Language:** {meta['language']}")
        if bits:
            lines.append(" · ".join(bits))
            lines.append("")
    last_speaker = object()
    for seg in segments:
        if seg.speaker != last_speaker:
            lines.append("")
            lines.append(f"**{_label(seg.speaker, names)}** `{hms(seg.start)}`")
            lines.append("")
            last_speaker = seg.speaker
        lines.append(seg.text)
    return "\n".join(lines).strip() + "\n"


def to_json(segments: list, names: Optional[dict] = None, *, meta: Optional[dict] = None,
            include_words: bool = True) -> str:
    payload = {
        "meta": meta or {},
        "speakers": names or {},
        "segments": [s.to_dict(include_words=include_words) for s in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_csv(segments: list, names: Optional[dict] = None) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["start", "end", "start_hms", "speaker", "text"])
    for seg in segments:
        w.writerow([
            f"{seg.start:.3f}", f"{seg.end:.3f}", hms(seg.start),
            _label(seg.speaker, names), seg.text,
        ])
    return buf.getvalue()


FORMATS = {
    "txt":  ("text/plain; charset=utf-8", "txt", to_text),
    "md":   ("text/markdown; charset=utf-8", "md", to_markdown),
    "srt":  ("application/x-subrip; charset=utf-8", "srt", to_srt),
    "vtt":  ("text/vtt; charset=utf-8", "vtt", to_vtt),
    "json": ("application/json; charset=utf-8", "json", to_json),
    "csv":  ("text/csv; charset=utf-8", "csv", to_csv),
}


def export(fmt: str, segments: list, names: Optional[dict] = None, meta: Optional[dict] = None) -> tuple:
    """Return (content_type, extension, body) for a requested format."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format: {fmt}")
    ctype, ext, fn = FORMATS[fmt]
    if fmt in ("md", "json"):
        body = fn(segments, names, meta=meta)
    else:
        body = fn(segments, names)
    return ctype, ext, body
