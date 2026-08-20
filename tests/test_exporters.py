import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe.align import Segment, Word
from transcribe.exporters import (
    srt_time, vtt_time, hms, to_srt, to_vtt, to_text, to_markdown, to_json, to_csv, export, _label,
)


def seg(start, end, speaker, text):
    return Segment(start=start, end=end, speaker=speaker, text=text,
                   words=[Word(start, end, text, speaker)])


SEGS = [
    seg(0.0, 2.5, "A", "Hello everyone."),
    seg(2.6, 5.0, "B", "Hi, good to be here."),
    seg(5.1, 8.0, "A", "Let's get started."),
]


class TestTimeFormats(unittest.TestCase):
    def test_srt_uses_comma(self):
        self.assertEqual(srt_time(3661.5), "01:01:01,500")

    def test_vtt_uses_period(self):
        self.assertEqual(vtt_time(3661.5), "01:01:01.500")

    def test_zero(self):
        self.assertEqual(srt_time(0), "00:00:00,000")

    def test_negative_clamped(self):
        self.assertEqual(srt_time(-1.0), "00:00:00,000")

    def test_past_24_hours(self):
        self.assertEqual(srt_time(90000.0), "25:00:00,000")

    def test_millisecond_rounding_does_not_overflow(self):
        # 1.9999 must not become "00:00:01,1000"
        self.assertEqual(srt_time(1.9999), "00:00:02,000")
        self.assertEqual(srt_time(59.9999), "00:01:00,000")
        self.assertEqual(srt_time(3599.9999), "01:00:00,000")

    def test_hms_human(self):
        self.assertEqual(hms(65), "1:05")
        self.assertEqual(hms(3665), "1:01:05")
        self.assertEqual(hms(0), "0:00")


class TestLabels(unittest.TestCase):
    def test_letter_speaker(self):
        self.assertEqual(_label("A", None), "Speaker A")

    def test_pyannote_style(self):
        self.assertEqual(_label("SPEAKER_00", None), "Speaker 0")
        self.assertEqual(_label("SPEAKER_12", None), "Speaker 12")

    def test_custom_name_wins(self):
        self.assertEqual(_label("A", {"A": "Justin"}), "Justin")

    def test_blank_custom_name_falls_back(self):
        self.assertEqual(_label("A", {"A": ""}), "Speaker A")

    def test_none_speaker(self):
        self.assertEqual(_label(None, None), "Speaker")


class TestSRT(unittest.TestCase):
    def test_structure(self):
        out = to_srt(SEGS)
        blocks = [b for b in out.strip().split("\n\n") if b.strip()]
        self.assertEqual(len(blocks), 3)
        first = blocks[0].split("\n")
        self.assertEqual(first[0], "1")
        self.assertEqual(first[1], "00:00:00,000 --> 00:00:02,500")
        self.assertEqual(first[2], "[Speaker A] Hello everyone.")

    def test_numbering_is_one_based_and_sequential(self):
        out = to_srt(SEGS)
        nums = [b.split("\n")[0] for b in out.strip().split("\n\n") if b.strip()]
        self.assertEqual(nums, ["1", "2", "3"])

    def test_names_applied(self):
        self.assertIn("[Justin]", to_srt(SEGS, {"A": "Justin"}))

    def test_without_speaker(self):
        out = to_srt(SEGS, with_speaker=False)
        self.assertNotIn("[Speaker", out)


class TestVTT(unittest.TestCase):
    def test_header_present(self):
        self.assertTrue(to_vtt(SEGS).startswith("WEBVTT\n"))

    def test_voice_span(self):
        self.assertIn("<v Speaker A>Hello everyone.", to_vtt(SEGS))

    def test_arrow_format(self):
        self.assertIn("00:00:00.000 --> 00:00:02.500", to_vtt(SEGS))

    def test_blank_line_after_header(self):
        lines = to_vtt(SEGS).split("\n")
        self.assertEqual(lines[1], "")


class TestText(unittest.TestCase):
    def test_speaker_blocks(self):
        out = to_text(SEGS)
        self.assertIn("Speaker A  [0:00]", out)
        self.assertIn("Speaker B  [0:02]", out)

    def test_same_speaker_not_repeated(self):
        segs = [seg(0, 1, "A", "One."), seg(1.1, 2, "A", "Two.")]
        self.assertEqual(to_text(segs).count("Speaker A"), 1)

    def test_speaker_repeats_after_switch_back(self):
        self.assertEqual(to_text(SEGS).count("Speaker A"), 2)

    def test_no_timestamps_option(self):
        self.assertNotIn("[0:00]", to_text(SEGS, timestamps=False))


class TestJSONAndCSV(unittest.TestCase):
    def test_json_roundtrip(self):
        data = json.loads(to_json(SEGS, {"A": "Justin"}, meta={"engine": "test"}))
        self.assertEqual(len(data["segments"]), 3)
        self.assertEqual(data["speakers"]["A"], "Justin")
        self.assertEqual(data["meta"]["engine"], "test")
        self.assertEqual(data["segments"][0]["start"], 0.0)
        self.assertIn("words", data["segments"][0])

    def test_json_can_omit_words(self):
        data = json.loads(to_json(SEGS, include_words=False))
        self.assertNotIn("words", data["segments"][0])

    def test_json_is_unicode_safe(self):
        out = to_json([seg(0, 1, "A", "café — naïve 日本語")])
        self.assertIn("café", out)
        self.assertIn("日本語", out)

    def test_csv_header_and_rows(self):
        rows = to_csv(SEGS).strip().split("\n")
        self.assertEqual(rows[0], "start,end,start_hms,speaker,text")
        self.assertEqual(len(rows), 4)

    def test_csv_quotes_commas(self):
        out = to_csv([seg(0, 1, "A", "Hi, there")])
        self.assertIn('"Hi, there"', out)


class TestExportDispatch(unittest.TestCase):
    def test_all_formats_produce_output(self):
        for fmt in ("txt", "md", "srt", "vtt", "json", "csv"):
            ctype, ext, body = export(fmt, SEGS, {"A": "Justin"}, {"name": "Demo", "duration": 8.0})
            self.assertTrue(body.strip(), fmt)
            self.assertEqual(ext, fmt)
            self.assertIn("charset=utf-8", ctype)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            export("docx", SEGS)

    def test_markdown_has_meta_header(self):
        out = to_markdown(SEGS, {"A": "Justin"}, meta={"name": "Demo", "duration": 8.0, "engine": "x"})
        self.assertIn("# Demo", out)
        self.assertIn("**Justin**", out)

    def test_empty_segments_do_not_crash(self):
        for fmt in ("txt", "md", "srt", "vtt", "json", "csv"):
            export(fmt, [], None, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
