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

import itertools

from transcribe.align import (
    Word, Turn, assign_speakers, smooth_speakers, smooth_sentences,
    resolve_low_confidence, enforce_speaker_count,
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


def confusing_diarizer(words, truth, seed=23, error_rate=0.14, honest=True):
    """Per-word speaker labels with confusion errors, the way Deepgram reports.

    Real diarizers do not just wobble at boundaries: they hand individual words
    to the wrong speaker outright, and — crucially — they tend to be *less
    confident* when they do. `honest=False` models an engine whose confidence
    tells you nothing, to check we do not make things worse for those.
    """
    rng = random.Random(seed)
    out = []
    for w, true_speaker in zip(words, truth):
        wrong = rng.random() < error_rate
        speaker = ("B" if true_speaker == "A" else "A") if wrong else true_speaker
        if honest:
            # Errors concentrate in the low-confidence band, but not perfectly:
            # plenty of correct words are also unsure.
            conf = rng.uniform(0.2, 0.55) if wrong else rng.uniform(0.35, 0.99)
        else:
            conf = rng.uniform(0.2, 0.99)
        out.append(Word(w.start, w.end, w.text, speaker=speaker, speaker_confidence=conf))
    return out


def boundary_lag_diarizer(words, truth, seed=9, lag_words=3, rate=0.6):
    """The most common real diarizer error: a turn boundary lands late.

    The first few words of a turn keep the previous speaker, and the engine is
    typically least confident exactly there. Nothing local can detect this —
    the mislabelled words simply extend the previous run — which is why the
    confidence signal is what makes it fixable.
    """
    rng = random.Random(seed)
    starts = [i for i in range(1, len(truth)) if truth[i] != truth[i - 1]]
    out = [Word(w.start, w.end, w.text, speaker=truth[i],
                speaker_confidence=rng.uniform(0.75, 0.99))
           for i, w in enumerate(words)]
    for s in starts:
        if rng.random() > rate:
            continue
        for k in range(s, min(s + rng.randint(1, lag_words), len(out))):
            out[k].speaker = truth[s - 1]
            out[k].speaker_confidence = rng.uniform(0.25, 0.5)
    return out


class ConfidenceTest(unittest.TestCase):
    """Using the diarizer's own speaker-confidence, measured over many seeds.

    Single-seed comparisons on a 66-word conversation are far too noisy to
    judge these by — one seed showed the confidence pass making things worse
    while the mean over 200 shows it halving the error.
    """

    SEEDS = 120

    def setUp(self):
        self.words, self.true_turns, self.truth = build_conversation()

    def _mean(self, generator, resolve, edge):
        import statistics
        scores = []
        for seed in range(self.SEEDS):
            words = generator(seed)
            if resolve:
                resolve_low_confidence(words)
            smooth_speakers(words)
            smooth_sentences(words, edge_confidence=edge)
            scores.append(wder(words, self.truth))
        return statistics.mean(scores)

    def scattered(self, seed):
        return confusing_diarizer(self.words, self.truth, seed=seed)

    def lagged(self, seed):
        return boundary_lag_diarizer(self.words, self.truth, seed=seed)

    def test_edge_confidence_fixes_boundary_lag(self):
        off = self._mean(self.lagged, resolve=False, edge=0.0)
        on = self._mean(self.lagged, resolve=False, edge=0.6)
        print(f"\n  boundary lag: {off:.1%} -> {on:.1%} with edge confidence")
        self.assertLess(on, off * 0.6,
                        "confidence-aware sentence edges should cut this sharply")

    def test_confidence_pass_helps_on_scattered_errors(self):
        off = self._mean(self.scattered, resolve=False, edge=0.6)
        on = self._mean(self.scattered, resolve=True, edge=0.6)
        print(f"  scattered noise: {off:.1%} -> {on:.1%} with the confidence pass")
        self.assertLess(on, off,
                        "re-deciding low-confidence words should help on average")

    def test_full_stack_beats_the_old_behaviour(self):
        for name, gen in (("scattered", self.scattered), ("lagged", self.lagged)):
            old = self._mean(gen, resolve=False, edge=0.0)
            new = self._mean(gen, resolve=True, edge=0.6)
            print(f"  {name}: {old:.1%} -> {new:.1%}")
            self.assertLess(new, old * 0.75, f"{name} should improve materially")

    def test_engines_reporting_nothing_are_untouched(self):
        words = [Word(0.0, 0.5, "a", "A"), Word(0.6, 1.1, "b", "B")]
        before = [w.speaker for w in words]
        resolve_low_confidence(words)
        self.assertEqual([w.speaker for w in words], before)

    def test_no_confidence_means_edges_stay_protected(self):
        """Engines without a confidence signal keep the conservative behaviour."""
        words = [
            Word(0.0, 1.0, "One", "A"),
            Word(1.0, 1.4, "two", "B"), Word(1.4, 2.4, "three.", "B"),
        ]
        smooth_sentences(words, edge_confidence=0.6)
        self.assertEqual(words[0].speaker, "A",
                         "an edge run with no confidence reported is still ambiguous")

    def test_confident_edge_runs_are_still_protected(self):
        """A real one-word turn the engine was sure about must survive."""
        words = [
            Word(0.0, 0.9, "Right.", "A", speaker_confidence=0.97),
            Word(1.0, 1.4, "So", "B", speaker_confidence=0.95),
            Word(1.4, 2.4, "anyway.", "B", speaker_confidence=0.95),
        ]
        smooth_sentences(words, edge_confidence=0.6)
        self.assertEqual(words[0].speaker, "A")

    def test_only_flips_when_both_neighbours_agree(self):
        words = [
            Word(0.0, 0.4, "one", "A", speaker_confidence=0.95),
            Word(0.5, 0.9, "two", "B", speaker_confidence=0.30),
            Word(1.0, 1.4, "three", "A", speaker_confidence=0.95),
        ]
        resolve_low_confidence(words)
        self.assertEqual(words[1].speaker, "A")

        words = [
            Word(0.0, 0.4, "one", "A", speaker_confidence=0.95),
            Word(0.5, 0.9, "two", "A", speaker_confidence=0.30),
            Word(1.0, 1.4, "three", "B", speaker_confidence=0.95),
        ]
        resolve_low_confidence(words)
        self.assertEqual(words[1].speaker, "A",
                         "neighbours disagree at a real boundary; leave it")

    def test_threshold_is_honoured_by_the_neighbour_search(self):
        """Both halves of the decision must use the caller's threshold.

        The neighbour search once tested a hardcoded 0.5 while the caller
        chose which words to correct using its own threshold. At any other
        setting the two disagreed: a word could be too unsure to trust as an
        answer, yet confident enough to serve as someone else's neighbour.

        Catching that needs a specific shape — the neighbours must sit
        *between* the hardcoded cutoff and the caller's threshold, and at
        least one word must clear the threshold or an early return fires
        first and hides the difference.
        """
        def build():
            return [
                Word(0.0, 0.4, "one", "A", speaker_confidence=0.55),   # 0.5 < c < 0.8
                Word(0.5, 0.9, "two", "B", speaker_confidence=0.30),   # the candidate
                Word(1.0, 1.4, "three", "A", speaker_confidence=0.55),
                Word(1.5, 1.9, "four", "A", speaker_confidence=0.95),  # clears any threshold
            ]

        # At 0.5 the flanking words are confident, so the middle word moves.
        low = build()
        resolve_low_confidence(low, threshold=0.5)
        self.assertEqual(low[1].speaker, "A")

        # At 0.8 they are not confident enough to be trusted as an answer, so
        # the middle word must be left alone. A hardcoded 0.5 would move it.
        high = build()
        resolve_low_confidence(high, threshold=0.8)
        self.assertEqual(high[1].speaker, "B",
                         "neighbours below the caller's threshold must not decide")

    def test_distant_neighbours_are_not_consulted(self):
        words = [
            Word(0.0, 0.4, "one", "A", speaker_confidence=0.95),
            Word(30.0, 30.4, "two", "B", speaker_confidence=0.20),
            Word(60.0, 60.4, "three", "A", speaker_confidence=0.95),
        ]
        resolve_low_confidence(words, window=4.0)
        self.assertEqual(words[1].speaker, "B", "half a minute away is not context")


def wder_one_to_one(pred, truth):
    """Error rate under the best ONE-TO-ONE label mapping — the DER convention.

    Using a many-to-one mapping instead would let every spurious cluster fold
    back onto the correct speaker for free, making over-segmentation score as
    perfect. That is not what a reader experiences: an invented third speaker
    is an error even when its words are otherwise in the right place.
    """
    plabels = sorted({w.speaker for w in pred if w.speaker is not None})
    tlabels = sorted(set(truth))
    best = len(truth)
    if len(plabels) <= len(tlabels):
        pairs = ((plabels, perm) for perm in itertools.permutations(tlabels, len(plabels)))
    else:
        pairs = ((chosen, tlabels) for chosen in itertools.permutations(plabels, len(tlabels)))
    for keys, values in pairs:
        m = dict(zip(keys, values))
        best = min(best, sum(1 for w, t in zip(pred, truth) if m.get(w.speaker) != t))
    return best / max(len(truth), 1)


def oversegmented(words, truth, seed, extra=1, share=0.35, whole_turns=True):
    """A diarizer that invents extra speakers out of one real speaker's audio.

    Two shapes, because they need opposite corrections: whole turns split into
    their own cluster (bracketed by the *other* speaker), and short fragments
    split out from inside a turn (bracketed by their own speaker).
    """
    rng = random.Random(seed)
    out = [Word(w.start, w.end, w.text, speaker=truth[i], speaker_confidence=0.9)
           for i, w in enumerate(words)]
    spare = [f"S{n}" for n in range(extra)]
    runs, cur, last = [], [], None
    for i, t in enumerate(truth):
        if t != last and cur:
            runs.append(cur)
            cur = []
        cur.append(i)
        last = t
    if cur:
        runs.append(cur)
    for r in runs:
        if truth[r[0]] != "A" or rng.random() >= share:
            continue
        tag = rng.choice(spare)
        if whole_turns:
            for i in r:
                out[i].speaker = tag
        else:
            out[r[len(r) // 2]].speaker = tag
    return out


class SpeakerCountTest(unittest.TestCase):
    """Applying a known speaker count after the fact.

    Deepgram's API takes no speaker-count hint at all, so for that engine this
    is the only place the user's answer can be used. Over-segmentation reads to
    a user as ordinary misattribution: a stretch of what one person said simply
    appears under somebody else's name.
    """

    SEEDS = 100

    def setUp(self):
        self.words, self.true_turns, self.truth = build_conversation()

    def _mean(self, whole_turns, target):
        import statistics
        scores = []
        for seed in range(self.SEEDS):
            w = oversegmented(self.words, self.truth, seed, whole_turns=whole_turns)
            if target:
                enforce_speaker_count(w, target)
            smooth_speakers(w)
            smooth_sentences(w)
            scores.append(wder_one_to_one(w, self.truth))
        return statistics.mean(scores)

    def test_recovers_whole_turns_split_into_a_new_cluster(self):
        off = self._mean(whole_turns=True, target=0)
        on = self._mean(whole_turns=True, target=2)
        print(f"\n  whole turns split off: {off:.1%} -> {on:.1%} with the count set")
        self.assertLess(on, off * 0.5, "should cut this error class sharply")

    def test_recovers_fragments_split_out_of_a_turn(self):
        off = self._mean(whole_turns=False, target=0)
        on = self._mean(whole_turns=False, target=2)
        print(f"  fragments split off:   {off:.1%} -> {on:.1%}")
        self.assertLessEqual(on, off)

    def test_correct_transcript_is_untouched(self):
        words = [Word(w.start, w.end, w.text, speaker=self.truth[i])
                 for i, w in enumerate(self.words)]
        enforce_speaker_count(words, 2)
        self.assertEqual([w.speaker for w in words], self.truth,
                         "a hint matching reality must change nothing")

    def test_hint_larger_than_reality_changes_nothing(self):
        words = [Word(w.start, w.end, w.text, speaker=self.truth[i])
                 for i, w in enumerate(self.words)]
        enforce_speaker_count(words, 5)
        self.assertEqual([w.speaker for w in words], self.truth)

    def test_merges_down_to_exactly_the_requested_count(self):
        words = [Word(i * 1.0, i * 1.0 + 0.8, f"w{i}",
                      speaker="ABCD"[i % 4]) for i in range(40)]
        enforce_speaker_count(words, 2)
        self.assertEqual(len({w.speaker for w in words}), 2)

    def test_a_short_interruption_folds_into_its_host(self):
        """X X x X X — the fragment belongs to X, not to whoever is next."""
        words = [
            Word(0.0, 2.0, "aaa", "A"), Word(2.0, 4.0, "bbb", "A"),
            Word(4.0, 4.3, "hm", "S0"),
            Word(4.3, 6.0, "ccc", "A"), Word(6.0, 8.0, "ddd", "A"),
            Word(8.0, 10.0, "eee", "B"), Word(10.0, 12.0, "fff", "B"),
        ]
        enforce_speaker_count(words, 2)
        self.assertEqual(words[2].speaker, "A")

    def test_zero_or_negative_target_is_a_noop(self):
        words = [Word(0, 1, "a", "A"), Word(1, 2, "b", "B"), Word(2, 3, "c", "C")]
        before = [w.speaker for w in words]
        enforce_speaker_count(words, 0)
        enforce_speaker_count(words, -1)
        self.assertEqual([w.speaker for w in words], before)

    def test_empty_input(self):
        self.assertEqual(enforce_speaker_count([], 2), [])


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
