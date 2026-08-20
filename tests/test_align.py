import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe.align import (
    Word, Turn, assign_speakers, smooth_speakers, smooth_sentences, build_segments,
    diarize_transcript, merge_adjacent, speaker_stats, _join_words, _sentence_spans, _minority_runs,
)


def W(start, end, text, speaker=None):
    return Word(start=start, end=end, text=text, speaker=speaker)


class TestAssign(unittest.TestCase):
    def test_basic_overlap(self):
        words = [W(0.0, 0.5, "Hello"), W(0.6, 1.0, "there"), W(5.1, 5.5, "Hi")]
        turns = [Turn(0.0, 2.0, "A"), Turn(5.0, 7.0, "B")]
        assign_speakers(words, turns)
        self.assertEqual([w.speaker for w in words], ["A", "A", "B"])

    def test_word_straddling_boundary_goes_to_majority_side(self):
        # Word spans 1.8-2.4; A covers 1.8-2.0 (0.2s), B covers 2.0-2.4 (0.4s).
        words = [W(1.8, 2.4, "maybe")]
        turns = [Turn(0.0, 2.0, "A"), Turn(2.0, 4.0, "B")]
        assign_speakers(words, turns)
        self.assertEqual(words[0].speaker, "B")

        words = [W(1.5, 2.1, "maybe")]
        assign_speakers(words, [Turn(0.0, 2.0, "A"), Turn(2.0, 4.0, "B")])
        self.assertEqual(words[0].speaker, "A")

    def test_word_in_gap_uses_nearest_turn(self):
        words = [W(2.3, 2.5, "um")]
        turns = [Turn(0.0, 2.2, "A"), Turn(4.0, 6.0, "B")]
        assign_speakers(words, turns)
        self.assertEqual(words[0].speaker, "A")

    def test_word_far_outside_any_turn_carries_previous(self):
        words = [W(0.0, 0.5, "one"), W(50.0, 50.4, "two")]
        turns = [Turn(0.0, 1.0, "A")]
        assign_speakers(words, turns)
        self.assertEqual([w.speaker for w in words], ["A", "A"])

    def test_leading_unresolved_words_inherit_first_known(self):
        words = [W(0.0, 0.2, "uh"), W(30.0, 30.5, "hello")]
        turns = [Turn(29.0, 40.0, "B")]
        assign_speakers(words, turns)
        self.assertEqual([w.speaker for w in words], ["B", "B"])

    def test_engine_word_speakers_are_respected(self):
        words = [W(0.0, 0.5, "Hello", speaker="X")]
        turns = [Turn(0.0, 2.0, "A")]
        assign_speakers(words, turns)
        self.assertEqual(words[0].speaker, "X")

    def test_no_turns_is_a_noop(self):
        words = [W(0.0, 0.5, "Hello")]
        assign_speakers(words, None)
        self.assertIsNone(words[0].speaker)

    def test_overlapping_turns_pick_max_overlap(self):
        words = [W(1.0, 2.0, "crosstalk")]
        turns = [Turn(0.0, 1.3, "A"), Turn(0.9, 3.0, "B")]
        assign_speakers(words, turns)
        self.assertEqual(words[0].speaker, "B")

    def test_cursor_handles_many_words_and_turns(self):
        words, turns = [], []
        for i in range(400):
            spk = "A" if i % 2 == 0 else "B"
            turns.append(Turn(i * 2.0, i * 2.0 + 2.0, spk))
            words.append(W(i * 2.0 + 0.5, i * 2.0 + 1.5, f"w{i}"))
        assign_speakers(words, turns)
        self.assertEqual([w.speaker for w in words[:6]], ["A", "B", "A", "B", "A", "B"])

    def test_unsorted_turns_are_handled(self):
        words = [W(5.1, 5.5, "Hi"), W(0.1, 0.5, "Hello")]
        words.sort(key=lambda w: w.start)
        turns = [Turn(5.0, 7.0, "B"), Turn(0.0, 2.0, "A")]
        assign_speakers(words, turns)
        self.assertEqual([w.speaker for w in words], ["A", "B"])


class TestSmooth(unittest.TestCase):
    def test_single_short_flip_absorbed(self):
        words = [W(0.0, 0.5, "a", "A"), W(0.5, 0.7, "b", "B"), W(0.7, 1.2, "c", "A")]
        smooth_speakers(words)
        self.assertEqual([w.speaker for w in words], ["A", "A", "A"])

    def test_real_long_interjection_preserved(self):
        words = [W(0.0, 0.5, "a", "A"), W(0.5, 1.4, "Exactly", "B"), W(1.4, 2.0, "c", "A")]
        smooth_speakers(words)
        self.assertEqual([w.speaker for w in words], ["A", "B", "A"])

    def test_many_word_run_preserved(self):
        words = [W(0.0, 0.2, "a", "A")] + [
            W(0.2 + i * 0.1, 0.3 + i * 0.1, f"x{i}", "B") for i in range(3)
        ] + [W(1.0, 1.2, "z", "A")]
        smooth_speakers(words)
        self.assertEqual([w.speaker for w in words][1:4], ["B", "B", "B"])

    def test_flip_between_different_speakers_not_absorbed(self):
        words = [W(0.0, 0.5, "a", "A"), W(0.5, 0.6, "b", "B"), W(0.6, 1.2, "c", "C")]
        smooth_speakers(words)
        self.assertEqual([w.speaker for w in words], ["A", "B", "C"])

    def test_short_lists_are_safe(self):
        for n in (0, 1, 2):
            words = [W(i * 0.1, i * 0.1 + 0.05, "x", "A") for i in range(n)]
            self.assertEqual(len(smooth_speakers(words)), n)

    def test_cascading_absorb_converges(self):
        words = [
            W(0.0, 0.5, "a", "A"), W(0.5, 0.6, "b", "B"),
            W(0.6, 0.7, "c", "A"), W(0.7, 0.8, "d", "B"), W(0.8, 1.4, "e", "A"),
        ]
        smooth_speakers(words)
        self.assertEqual(set(w.speaker for w in words), {"A"})


class TestSentenceSmoothing(unittest.TestCase):
    def test_straddle_error_snapped_to_dominant_speaker(self):
        # "I think that is right." — one word wrongly given to B mid-sentence.
        words = [
            W(0.0, 0.4, "I", "A"), W(0.4, 0.9, "think", "A"),
            W(0.9, 1.2, "that", "B"),
            W(1.2, 1.6, "is", "A"), W(1.6, 2.2, "right.", "A"),
        ]
        smooth_sentences(words)
        self.assertEqual([w.speaker for w in words], ["A"] * 5)

    def test_real_interjection_is_protected(self):
        # B holds the floor for 1.8s: a real turn, even without punctuation.
        words = [
            W(0.0, 0.4, "I", "A"), W(0.4, 0.9, "think", "A"),
            W(1.0, 2.8, "absolutely", "B"),
            W(2.9, 3.3, "yes.", "A"),
        ]
        smooth_sentences(words)
        self.assertEqual(words[2].speaker, "B")

    def test_long_unpunctuated_run_left_alone(self):
        # A 20s "sentence" is unpunctuated ASR output, not one utterance.
        words = [W(i * 1.0, i * 1.0 + 0.8, f"w{i}", "A") for i in range(18)]
        words.append(W(18.0, 18.8, "no.", "B"))
        smooth_sentences(words)
        self.assertEqual(words[-1].speaker, "B")

    def test_no_majority_leaves_split_alone(self):
        words = [
            W(0.0, 1.0, "one", "A"), W(1.0, 1.4, "two", "B"), W(1.4, 2.4, "three", "A"),
            W(2.4, 3.4, "four", "B"), W(3.4, 4.4, "five.", "B"),
        ]
        before = [w.speaker for w in words]
        smooth_sentences(words)
        self.assertEqual([w.speaker for w in words], before,
                         "no speaker holds a 60% majority, so nothing moves")

    def test_single_speaker_sentence_untouched(self):
        words = [W(0.0, 0.5, "Hello", "A"), W(0.5, 1.0, "there.", "A")]
        smooth_sentences(words)
        self.assertEqual([w.speaker for w in words], ["A", "A"])

    def test_each_sentence_scored_independently(self):
        words = [
            W(0.0, 0.4, "I", "A"), W(0.4, 0.9, "agree", "B"), W(0.9, 1.4, "entirely.", "A"),
            W(2.0, 2.4, "No", "B"), W(2.4, 2.9, "you", "B"), W(2.9, 3.4, "don't.", "A"),
        ]
        smooth_sentences(words)
        # Interior flip corrected...
        self.assertEqual([w.speaker for w in words[:3]], ["A", "A", "A"])
        # ...trailing one left alone: it may be a real turn the punctuation lags.
        self.assertEqual([w.speaker for w in words[3:]], ["B", "B", "A"])

    def test_leading_minority_run_left_alone(self):
        words = [W(0.0, 1.0, "one", "A"), W(1.0, 2.0, "two", "B"), W(2.0, 3.0, "three.", "B")]
        smooth_sentences(words)
        self.assertEqual([w.speaker for w in words], ["A", "B", "B"])

    def test_minority_runs_indexing(self):
        span = [W(0, 1, "a", "A"), W(1, 2, "b", "B"), W(2, 3, "c", "B"), W(3, 4, "d", "A")]
        self.assertEqual(_minority_runs(span, "A"), [(1, 3)])
        self.assertEqual(_minority_runs(span, "B"), [(0, 1), (3, 4)])

    def test_multiple_interior_runs_all_corrected(self):
        words = [
            W(0.0, 0.6, "a", "A"), W(0.6, 0.9, "b", "B"), W(0.9, 1.5, "c", "A"),
            W(1.5, 1.8, "d", "B"), W(1.8, 2.6, "e.", "A"),
        ]
        smooth_sentences(words)
        self.assertEqual([w.speaker for w in words], ["A"] * 5)

    def test_sentence_spans(self):
        words = [W(0, 0.1, "a"), W(0.1, 0.2, "b."), W(0.2, 0.3, "c"), W(0.3, 0.4, "d?")]
        self.assertEqual(_sentence_spans(words), [(0, 2), (2, 4)])

    def test_trailing_words_without_punctuation_form_a_span(self):
        words = [W(0, 0.1, "a."), W(0.1, 0.2, "b")]
        self.assertEqual(_sentence_spans(words), [(0, 1), (1, 2)])

    def test_short_lists_safe(self):
        self.assertEqual(smooth_sentences([]), [])
        self.assertEqual(len(smooth_sentences([W(0, 1, "hi.", "A")])), 1)

    def test_pipeline_can_disable_it(self):
        # 0.7s is long enough to survive the run-length smoother, so this
        # isolates the sentence pass.
        def make():
            return [
                W(0.0, 0.4, "I", "A"), W(0.4, 0.9, "think", "A"),
                W(0.9, 1.6, "that", "B"),
                W(1.6, 2.0, "is", "A"), W(2.0, 2.6, "right.", "A"),
            ]
        on = diarize_transcript(make(), None)
        self.assertEqual(len(on), 1, "sentence smoothing should heal the straddle")
        off = diarize_transcript(make(), None, sentence_smoothing=False)
        self.assertEqual(len(off), 3, "without it, the flip splits the turn three ways")


class TestOverlapSummation(unittest.TestCase):
    def test_two_short_turns_beat_one_longer(self):
        # A holds 0.2 + 0.2 = 0.4s of the word; B holds 0.3s in one turn.
        words = [W(1.0, 2.0, "word")]
        turns = [Turn(1.0, 1.2, "A"), Turn(1.2, 1.5, "B"), Turn(1.5, 1.7, "A")]
        assign_speakers(words, turns)
        self.assertEqual(words[0].speaker, "A")


class TestSegments(unittest.TestCase):
    def test_split_on_speaker_change(self):
        words = [W(0.0, 0.5, "Hello", "A"), W(0.6, 1.0, "Hi", "B")]
        segs = build_segments(words)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].text, "Hello")
        self.assertEqual(segs[1].speaker, "B")

    def test_split_on_long_gap(self):
        words = [W(0.0, 0.5, "Hello", "A"), W(5.0, 5.5, "again", "A")]
        segs = build_segments(words, max_gap=1.0)
        self.assertEqual(len(segs), 2)

    def test_no_split_on_short_gap(self):
        words = [W(0.0, 0.5, "Hello", "A"), W(0.9, 1.5, "again", "A")]
        segs = build_segments(words, max_gap=1.0)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "Hello again")

    def test_long_run_splits_at_sentence_end(self):
        words = []
        t = 0.0
        for i in range(60):
            text = "word." if i % 10 == 9 else "word"
            words.append(W(t, t + 0.4, text, "A"))
            t += 0.5
        segs = build_segments(words, max_dur=5.0, max_gap=1.0)
        self.assertGreater(len(segs), 1)
        for s in segs[:-1]:
            self.assertTrue(s.text.rstrip().endswith("."), s.text)

    def test_segment_boundaries_and_words_preserved(self):
        words = [W(0.0, 0.5, "a", "A"), W(0.6, 1.2, "b", "A")]
        segs = build_segments(words)
        self.assertAlmostEqual(segs[0].start, 0.0)
        self.assertAlmostEqual(segs[0].end, 1.2)
        self.assertEqual(len(segs[0].words), 2)

    def test_empty_input(self):
        self.assertEqual(build_segments([]), [])


class TestJoin(unittest.TestCase):
    def test_punctuation_not_spaced(self):
        words = [W(0, 0.1, "Hello"), W(0.1, 0.2, ","), W(0.2, 0.3, "world"), W(0.3, 0.4, "!")]
        self.assertEqual(_join_words(words), "Hello, world!")

    def test_contraction_apostrophe(self):
        words = [W(0, 0.1, "it"), W(0.1, 0.2, "'s"), W(0.2, 0.3, "fine")]
        self.assertEqual(_join_words(words), "it's fine")

    def test_normal_words_spaced(self):
        words = [W(0, 0.1, "one"), W(0.1, 0.2, "two")]
        self.assertEqual(_join_words(words), "one two")


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        words = [
            W(0.0, 0.4, "Hello"), W(0.4, 0.9, "everyone."),
            W(1.0, 1.4, "Hi"), W(1.4, 1.9, "there."),
        ]
        turns = [Turn(0.0, 0.95, "A"), Turn(0.95, 3.0, "B")]
        segs = diarize_transcript(words, turns)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].text, "Hello everyone.")
        self.assertEqual(segs[1].text, "Hi there.")

    def test_merge_adjacent(self):
        words = [W(0.0, 0.4, "One.", "A"), W(2.0, 2.4, "Two.", "A")]
        segs = build_segments(words, max_gap=1.0)
        self.assertEqual(len(segs), 2)
        merged = merge_adjacent(segs, max_gap=2.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "One. Two.")
        self.assertEqual(len(merged[0].words), 2)

    def test_merge_does_not_cross_speakers(self):
        words = [W(0.0, 0.4, "One.", "A"), W(0.5, 0.9, "Two.", "B")]
        merged = merge_adjacent(build_segments(words), max_gap=5.0)
        self.assertEqual(len(merged), 2)

    def test_merge_is_non_destructive(self):
        words = [W(0.0, 0.4, "One.", "A"), W(2.0, 2.4, "Two.", "A")]
        segs = build_segments(words, max_gap=1.0)
        merge_adjacent(segs, max_gap=2.0)
        self.assertEqual(len(segs), 2, "merge_adjacent must not mutate its input")

    def test_speaker_stats(self):
        words = [W(0.0, 3.0, "aaa", "A"), W(4.0, 5.0, "b", "B")]
        stats = speaker_stats(build_segments(words))
        self.assertEqual(stats[0]["speaker"], "A")
        self.assertAlmostEqual(stats[0]["seconds"], 3.0)
        self.assertAlmostEqual(sum(s["share"] for s in stats), 1.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
