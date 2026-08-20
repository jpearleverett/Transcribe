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
    confidence: Optional[float] = None          # how sure the ASR is of the word
    # How sure the diarizer is of the *speaker*. A different quantity entirely,
    # and the one that matters for attribution: a perfectly transcribed word can
    # still be handed to the wrong person. Deepgram reports it per word; other
    # engines leave it None, in which case everything below treats it as 1.0.
    speaker_confidence: Optional[float] = None

    @property
    def dur(self) -> float:
        return max(self.end - self.start, MIN_WORD_DUR)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def spk_conf(self) -> float:
        """Speaker confidence, treating "unreported" as certain."""
        if self.speaker_confidence is None:
            return 1.0
        return max(0.0, min(1.0, self.speaker_confidence))


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
                    **({"speaker_confidence": round(w.speaker_confidence, 4)}
                       if w.speaker_confidence is not None else {}),
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

        # Sum overlap per speaker rather than taking the single best turn: a
        # speaker with two short turns touching one word should beat a speaker
        # with one slightly longer one.
        totals: dict = {}
        j = cursor
        while j < len(turns) and turns[j].start < w.end:
            ov = _overlap(w.start, w.end, turns[j].start, turns[j].end)
            if ov > 0:
                totals[turns[j].speaker] = totals.get(turns[j].speaker, 0.0) + ov
            j += 1
        best_spk = max(totals, key=totals.get) if totals else None

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


def resolve_low_confidence(
    words: list,
    *,
    threshold: float = 0.5,
    window: float = 4.0,
) -> list:
    """Re-decide words whose speaker the diarizer was unsure about.

    Some engines report how confident they are in each word's *speaker*
    separately from the word itself. Those low-confidence words are where
    confusion errors concentrate — a diarizer that is 45% sure is essentially
    guessing, and its guess is as likely to be wrong as right.

    Where the confident words on both sides agree, we take their answer. That
    is strictly better than a coin flip, and it leaves genuine turn changes
    alone because those have confident words disagreeing across the boundary.
    """
    if not words:
        return words
    if all(w.speaker_confidence is None for w in words):
        return words        # engine reports nothing to work with

    confident = [i for i, w in enumerate(words) if w.spk_conf >= threshold and w.speaker]
    if not confident:
        return words

    for i, w in enumerate(words):
        if w.spk_conf >= threshold or not w.speaker:
            continue
        before = _nearest_confident(words, confident, i, -1, window)
        after = _nearest_confident(words, confident, i, 1, window)
        if before is not None and after is not None and before == after:
            w.speaker = before
    return words


def _nearest_confident(words: list, confident: list, index: int, step: int,
                       window: float) -> Optional[str]:
    """The speaker of the nearest confident word within `window` seconds."""
    i = index + step
    while 0 <= i < len(words):
        gap = abs(words[i].mid - words[index].mid)
        if gap > window:
            return None
        if words[i].spk_conf >= 0.5 and words[i].speaker:
            return words[i].speaker
        i += step
    return None


# --------------------------------------------------------------------------
# Step 2: smooth implausible speaker flips
# --------------------------------------------------------------------------

def smooth_speakers(
    words: list,
    *,
    min_run_words: int = 2,
    min_run_dur: float = 0.40,
) -> list:
    """Absorb short speaker flips surrounded by one other speaker.

    A run is only absorbed when it is short in words, short in time, and has the
    same speaker on both sides — that combination is overwhelmingly a diarizer
    boundary error.

    The exception that matters: a run forming a *complete sentence* of its own
    ("Right." "Exactly.") is a real interjection, and those are frequently under
    both thresholds — "Right." is often less than 200 ms. Absorbing them would
    silently delete exactly the short turns that per-word attribution exists to
    capture, so a run bounded by sentence punctuation on both sides is kept
    regardless of how brief it is.
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
            if _is_standalone_sentence(words, s, e):
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


def smooth_sentences(
    words: list,
    *,
    majority: float = 0.6,
    max_sentence_dur: float = 15.0,
    protect_run_dur: float = 1.5,
    edge_confidence: float = 0.6,
) -> list:
    """Snap a sentence to its dominant speaker.

    Real speaker changes almost never happen mid-sentence, so when one speaker
    holds most of a sentence the stragglers are usually boundary errors from the
    diarizer. This is the single most effective correction available after
    per-word overlap assignment.

    It is applied conservatively, because the premise fails in three ways:

      * A long "sentence" is usually one the ASR failed to punctuate, and those
        genuinely do span turns — hence `max_sentence_dur`.
      * A substantial contiguous stretch by the minority speaker is a real turn
        that happens to lack punctuation around it — hence `protect_run_dur`.
      * A minority run at the *start or end* of a sentence is ambiguous: it is
        just as likely to be a real turn change that the punctuation lags by a
        word as it is to be an error. We only correct runs strictly interior to
        the sentence, where no plausible turn structure explains a speaker
        leaving and the same speaker immediately resuming. Edge runs keep
        whatever the per-word overlap decided, which is the best local evidence
        we have without reading across sentences.
    """
    if len(words) < 2:
        return words

    for start, end in _sentence_spans(words):
        span = words[start:end]
        if len(span) < 2:
            continue
        duration = span[-1].end - span[0].start
        if duration > max_sentence_dur:
            continue

        totals: dict = {}
        for w in span:
            if w.speaker is not None:
                # Weight by speaker confidence: a word the diarizer was unsure
                # about should not anchor the whole sentence against words it
                # was certain about. Engines reporting nothing weigh 1.0.
                totals[w.speaker] = totals.get(w.speaker, 0.0) + w.dur * max(w.spk_conf, 0.05)
        if len(totals) < 2:
            continue

        total = sum(totals.values()) or 1.0
        winner = max(totals, key=totals.get)
        if totals[winner] / total < majority:
            continue

        for run_start, run_end in _minority_runs(span, winner):
            run = span[run_start:run_end]
            at_edge = run_start == 0 or run_end == len(span)
            if at_edge:
                # An edge run is normally ambiguous — as likely a real turn the
                # punctuation lags by a word as an error. But when the engine
                # reports how sure it was, an edge run it was *unsure* about is
                # not ambiguous: that is the signature of a turn boundary landing
                # a few words late, which is one of the most common diarizer
                # errors. Confident edge runs are still left alone.
                if min((w.spk_conf for w in run), default=1.0) >= edge_confidence:
                    continue
            if sum(w.dur for w in run) >= protect_run_dur:
                continue
            for w in run:
                w.speaker = winner
    return words


def _sentence_spans(words: list) -> list:
    """Index ranges delimited by sentence-ending punctuation."""
    spans, start = [], 0
    for i, w in enumerate(words):
        if SENTENCE_END.search(w.text):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def _minority_runs(span: list, winner) -> list:
    """Index ranges of contiguous words not belonging to `winner`."""
    runs, start = [], None
    for i, w in enumerate(span):
        if w.speaker != winner:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(span)))
    return runs


def _is_standalone_sentence(words: list, start: int, end: int) -> bool:
    """True when words[start:end] is exactly one or more whole sentences."""
    if end <= start or not SENTENCE_END.search(words[end - 1].text):
        return False
    return start == 0 or bool(SENTENCE_END.search(words[start - 1].text))


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
    if opts.get("use_speaker_confidence", True):
        words = resolve_low_confidence(
            words,
            threshold=opts.get("speaker_confidence_threshold", 0.5),
        )
    words = smooth_speakers(
        words,
        min_run_words=opts.get("min_run_words", 2),
        min_run_dur=opts.get("min_run_dur", 0.40),
    )
    if opts.get("sentence_smoothing", True):
        words = smooth_sentences(
            words,
            majority=opts.get("sentence_majority", 0.6),
            max_sentence_dur=opts.get("max_sentence_dur", 15.0),
            protect_run_dur=opts.get("protect_run_dur", 1.5),
            edge_confidence=opts.get("edge_confidence", 0.6),
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
