"""Word-level speaker attribution and readable segment construction.

This is the accuracy-critical part of the pipeline. ASR engines give us words
with timestamps; diarizers give us a timeline of speaker turns. Naively
assigning a whole ASR segment to whichever speaker "mostly" covers it loses
every short interjection and smears speaker boundaries by seconds. We instead
attribute *each word* by temporal overlap, smooth away implausible one-word
flips, and only then group words back into readable segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional


# A word shorter than this is treated as having a nominal duration when its
# engine-reported end <= start (some engines emit zero-length words).
MIN_WORD_DUR = 0.02

SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*$")


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None

    @property
    def dur(self) -> float:
        return max(self.end - self.start, MIN_WORD_DUR)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Segment:
    start: float
    end: float
    speaker: Optional[str]
    text: str
    words: list = field(default_factory=list)

    def to_dict(self, include_words: bool = True) -> dict:
        d = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "text": self.text,
        }
        if include_words:
            d["words"] = [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "text": w.text,
                    "speaker": w.speaker,
                    **({"confidence": round(w.confidence, 4)} if w.confidence is not None else {}),
                }
                for w in self.words
            ]
        return d


# --------------------------------------------------------------------------
# Step 1: attribute each word to a speaker
# --------------------------------------------------------------------------

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    words: list,
    turns: Optional[list] = None,
    *,
    nearest_tolerance: float = 2.0,
) -> list:
    """Give every word a speaker label.

    Precedence:
      1. A speaker the engine already attached to the word (AssemblyAI, Deepgram
         and friends do word-level diarization themselves and are better at it
         than we can be after the fact).
      2. Maximum temporal overlap with a diarization turn. A word that straddles
         a boundary goes to whichever side holds more of it, which is the
         behaviour you want for the very common case of one speaker starting
         before the other has finished a word.
      3. For a word landing in a gap between turns (diarizers routinely leave
         gaps over breaths and non-speech), the nearest turn within
         `nearest_tolerance` seconds, measured centre-to-interval.
      4. Carry forward the previous word's speaker.
    """
    if not words:
        return words
    if not turns:
        return words

    turns = sorted(turns, key=lambda t: (t.start, t.end))
    last = None
    # Turns are sorted, so we can advance a cursor instead of rescanning the
    # whole timeline for every word: this is O(n + m) rather than O(n*m), which
    # matters for a 3-hour file with 30k words.
    cursor = 0
    for w in words:
        if w.speaker:
            last = w.speaker
            continue

        while cursor > 0 and turns[cursor - 1].end > w.start:
            cursor -= 1
        while cursor < len(turns) and turns[cursor].end < w.start:
            cursor += 1

        best_spk, best_ov = None, 0.0
        j = cursor
        while j < len(turns) and turns[j].start < w.end:
            ov = _overlap(w.start, w.end, turns[j].start, turns[j].end)
            if ov > best_ov:
                best_ov, best_spk = ov, turns[j].speaker
            j += 1

        if best_spk is None:
            # No overlap at all -> nearest turn by distance from the word centre.
            best_dist = nearest_tolerance
            mid = w.mid
            for j in range(max(0, cursor - 2), min(len(turns), cursor + 3)):
                t = turns[j]
                dist = 0.0 if t.start <= mid <= t.end else min(abs(mid - t.start), abs(mid - t.end))
                if dist < best_dist:
                    best_dist, best_spk = dist, t.speaker

        w.speaker = best_spk or last
        if w.speaker:
            last = w.speaker

    # Any leading words we never resolved inherit the first speaker we did.
    first = next((w.speaker for w in words if w.speaker), None)
    for w in words:
        if not w.speaker:
            w.speaker = first
        else:
            break
    return words


# --------------------------------------------------------------------------
# Step 2: smooth implausible speaker flips
# --------------------------------------------------------------------------

def smooth_speakers(
    words: list,
    *,
    min_run_words: int = 2,
    min_run_dur: float = 0.40,
) -> list:
    """Absorb single-word speaker flips surrounded by one other speaker.

    A genuine one-word turn ("Right." "Exactly.") is real and worth keeping, so
    we only absorb a run when it is *both* short in words and short in time and
    the same speaker holds the floor on both sides. That combination is
    overwhelmingly a boundary error rather than a real interjection.
    """
    if len(words) < 3:
        return words

    runs = []  # (start_idx, end_idx_exclusive, speaker)
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].speaker == words[i].speaker:
            j += 1
        runs.append([i, j, words[i].speaker])
        i = j

    changed = True
    while changed:
        changed = False
        for k in range(1, len(runs) - 1):
            s, e, spk = runs[k]
            prev_spk, next_spk = runs[k - 1][2], runs[k + 1][2]
            if prev_spk != next_spk or prev_spk == spk:
                continue
            n_words = e - s
            dur = words[e - 1].end - words[s].start
            if n_words <= min_run_words and dur <= min_run_dur:
                for w in words[s:e]:
                    w.speaker = prev_spk
                runs[k][2] = prev_spk
                changed = True
        if changed:
            merged = []
            for r in runs:
                if merged and merged[-1][2] == r[2]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            runs = merged
    return words


# --------------------------------------------------------------------------
# Step 3: group words into readable segments
# --------------------------------------------------------------------------

def build_segments(
    words: list,
    *,
    max_gap: float = 1.0,
    max_dur: float = 30.0,
    max_chars: int = 320,
    soft_ratio: float = 0.5,
) -> list:
    """Group consecutive same-speaker words into caption-sized segments.

    Splits on: speaker change (always), a silence longer than `max_gap`, a
    sentence ending once the segment is past `soft_ratio` of its budget, or the
    hard `max_dur`/`max_chars` cap for speech with no usable punctuation.
    """
    segments: list = []
    cur: list = []

    def flush():
        if not cur:
            return
        text = _join_words(cur)
        segments.append(
            Segment(start=cur[0].start, end=cur[-1].end, speaker=cur[0].speaker, text=text, words=list(cur))
        )
        cur.clear()

    for w in words:
        if not cur:
            cur.append(w)
            continue

        prev = cur[-1]
        gap = w.start - prev.end
        seg_dur = prev.end - cur[0].start
        seg_chars = sum(len(x.text) + 1 for x in cur)

        if w.speaker != prev.speaker or gap > max_gap:
            flush()
        elif SENTENCE_END.search(prev.text) and (
            seg_dur >= max_dur * soft_ratio or seg_chars >= max_chars * soft_ratio
        ):
            # Past the soft target and we just finished a sentence: break here.
            # Waiting for the hard cap instead would overshoot to the middle of
            # the *next* sentence, which is exactly the mid-clause slice we are
            # trying to avoid.
            flush()
        elif seg_dur >= max_dur or seg_chars >= max_chars:
            flush()  # hard cap: a run-on with no punctuation still has to break
        cur.append(w)

    flush()
    return segments


def _join_words(words: Iterable) -> str:
    out = []
    for w in words:
        t = w.text
        if not t:
            continue
        if out and not t.startswith(("'", "’")) and t[0] not in ",.!?;:)]}%":
            out.append(" ")
        out.append(t)
    return "".join(out).strip()


def merge_adjacent(segments: list, *, max_gap: float = 0.8, max_dur: float = 60.0) -> list:
    """Merge neighbouring same-speaker segments into paragraph-sized blocks.

    Used for the reading view; captions keep the unmerged segments.
    """
    out: list = []
    for seg in segments:
        if (
            out
            and out[-1].speaker == seg.speaker
            and seg.start - out[-1].end <= max_gap
            and seg.end - out[-1].start <= max_dur
        ):
            prev = out[-1]
            prev.end = seg.end
            prev.text = (prev.text + " " + seg.text).strip()
            prev.words.extend(seg.words)
        else:
            out.append(Segment(seg.start, seg.end, seg.speaker, seg.text, list(seg.words)))
    return out


def diarize_transcript(
    words: list,
    turns: Optional[list] = None,
    **opts,
) -> list:
    """Full pipeline: attribute -> smooth -> segment."""
    words = assign_speakers(words, turns, nearest_tolerance=opts.get("nearest_tolerance", 2.0))
    words = smooth_speakers(
        words,
        min_run_words=opts.get("min_run_words", 2),
        min_run_dur=opts.get("min_run_dur", 0.40),
    )
    return build_segments(
        words,
        max_gap=opts.get("max_gap", 1.0),
        max_dur=opts.get("max_dur", 30.0),
        max_chars=opts.get("max_chars", 320),
    )


def speaker_stats(segments: list) -> list:
    """Per-speaker talk time and word counts, for the UI summary strip."""
    stats: dict = {}
    for s in segments:
        e = stats.setdefault(s.speaker, {"speaker": s.speaker, "seconds": 0.0, "words": 0, "segments": 0})
        e["seconds"] += max(0.0, s.end - s.start)
        e["words"] += len(s.words) or len(s.text.split())
        e["segments"] += 1
    total = sum(e["seconds"] for e in stats.values()) or 1.0
    for e in stats.values():
        e["seconds"] = round(e["seconds"], 2)
        e["share"] = round(e["seconds"] / total, 4)
    return sorted(stats.values(), key=lambda e: -e["seconds"])
