"""Measures word-level speaker error on a simulated conversation.

The pipeline in align.py exists to beat naive segment-level attribution. That
claim should be measured, not asserted, so this builds a synthetic conversation
with known ground truth, runs a diarizer simulation that makes the errors real
diarizers make (jittered turn boundaries, occasional dropped short turns), and
reports WDER — the fraction of words given the wrong speaker.

Deterministic: a fixed seed, so a regression fails the build rather than
flickering.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe.align import (
    Word, Turn, assign_speakers, smooth_speakers, smooth_sentences,
)

SENTENCES = [
    "So how did the quarter actually go for you.",
    "Better than expected honestly.",
    "Revenue came in about eleven percent above plan.",
    "That is a big jump.",
    "What drove it.",
    "Mostly renewals and a couple of large expansions.",
    "Did churn move at all.",
    "It ticked down slightly which surprised everyone.",
    "Right.",
    "Exactly.",
    "And the pipeline for next quarter looks reasonable.",
    "I would still call it cautious optimism.",
]


def build_conversation(seed=7):
    """Return (words, true_turns) with ground-truth speakers on each word."""
    rng = random.Random(seed)
    words, turns = [], []
    t = 0.0
    speaker = "A"
    for i, sentence in enumerate(SENTENCES):
        turn_start = t
        for token in sentence.split():
            dur = rng.uniform(0.18, 0.42)
            words.append(Word(start=round(t, 3), end=round(t + dur, 3), text=token))
            t += dur + rng.uniform(0.01, 0.06)
        turns.append(Turn(round(turn_start, 3), round(t, 3), speaker))
        # Ground truth lives on a parallel list, since assign_speakers writes
        # to Word.speaker.
        for w in words[-len(sentence.split()):]:
            w.confidence = None
        t += rng.uniform(0.15, 0.5)          # inter-turn pause
        speaker = "B" if speaker == "A" else "A"
    truth = []
    for turn in turns:
        for w in words:
            if turn.start <= w.start < turn.end:
                truth.append(turn.speaker)
    truth = []
    for w in words:
        for turn in turns:
            if turn.start <= w.start <= turn.end:
                truth.append(turn.speaker)
                break
    assert len(truth) == len(words)
    return words, turns, truth


def noisy_turns(true_turns, seed=11, jitter=0.35, drop_short=True):
    """A diarization timeline with the errors real diarizers make."""
    rng = random.Random(seed)
    out = []
    for turn in true_turns:
        start = turn.start + rng.uniform(-jitter, jitter)
        end = turn.end + rng.uniform(-jitter, jitter)
        if end - start < 0.05:
            continue
        # Real diarizers sometimes miss a very short turn entirely, which the
        # gap-filling path then has to cover.
        if drop_short and (turn.end - turn.start) < 0.7 and rng.random() < 0.5:
            continue
        out.append(Turn(round(start, 3), round(end, 3), turn.speaker))
    return out


def wder(words, truth):
    wrong = sum(1 for w, t in zip(words, truth) if w.speaker != t)
    return wrong / max(len(words), 1)


def naive_segment_assignment(words, turns, truth):
    """Baseline: one speaker per ASR segment, by majority coverage.

    Approximates what most tools do — group words into ~10 s blocks (standing in
    for Whisper's decoder segments) and give each block a single speaker.
    """
    labelled = [Word(w.start, w.end, w.text) for w in words]
    block, blocks = [], []
    for w in labelled:
        if block and w.end - block[0].start > 10.0:
            blocks.append(block)
            block = []
        block.append(w)
    if block:
        blocks.append(block)

    for blk in blocks:
        span_start, span_end = blk[0].start, blk[-1].end
        totals = {}
        for turn in turns:
            ov = max(0.0, min(span_end, turn.end) - max(span_start, turn.start))
            if ov > 0:
                totals[turn.speaker] = totals.get(turn.speaker, 0.0) + ov
        winner = max(totals, key=totals.get) if totals else None
        for w in blk:
            w.speaker = winner
    return wder(labelled, truth)


class AccuracyTest(unittest.TestCase):
    def setUp(self):
        self.words, self.true_turns, self.truth = build_conversation()
        self.turns = noisy_turns(self.true_turns)

    def _run_pipeline(self, sentences=True):
        words = [Word(w.start, w.end, w.text) for w in self.words]
        assign_speakers(words, self.turns)
        smooth_speakers(words)
        if sentences:
            smooth_sentences(words)
        return words

    def test_pipeline_beats_naive_segment_assignment(self):
        naive = naive_segment_assignment(self.words, self.turns, self.truth)
        ours = wder(self._run_pipeline(), self.truth)
        print(f"\n  WDER naive segment-level : {naive:.1%}"
              f"\n  WDER per-word pipeline   : {ours:.1%}")
        self.assertLess(ours, naive,
                        "per-word attribution should beat segment-level majority")
        self.assertLess(ours, 0.12, "word error rate should stay under 12%")

    def test_smoothing_passes_help_not_hurt(self):
        raw = [Word(w.start, w.end, w.text) for w in self.words]
        assign_speakers(raw, self.turns)
        base = wder(raw, self.truth)
        smoothed = wder(self._run_pipeline(), self.truth)
        print(f"  WDER before smoothing    : {base:.1%}"
              f"\n  WDER after smoothing     : {smoothed:.1%}")
        self.assertLessEqual(smoothed, base,
                             "smoothing must never make attribution worse")

    def test_holds_up_under_heavier_jitter(self):
        for jitter in (0.2, 0.5, 0.8):
            turns = noisy_turns(self.true_turns, jitter=jitter)
            words = [Word(w.start, w.end, w.text) for w in self.words]
            assign_speakers(words, turns)
            smooth_speakers(words)
            smooth_sentences(words)
            score = wder(words, self.truth)
            print(f"  jitter ±{jitter:.1f}s -> WDER {score:.1%}")
            self.assertLess(score, 0.30, f"fell apart at jitter {jitter}")

    def test_perfect_turns_give_perfect_attribution(self):
        words = [Word(w.start, w.end, w.text) for w in self.words]
        assign_speakers(words, self.true_turns)
        smooth_speakers(words)
        smooth_sentences(words)
        self.assertEqual(wder(words, self.truth), 0.0,
                         "with an exact diarization timeline nothing should be misattributed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
