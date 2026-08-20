"""The offline engine, driven with stub binaries.

whisper.cpp and sherpa-onnx cannot be built here, but almost everything that
breaks in this engine is the plumbing around them: discovering the model files,
parsing whisper.cpp's JSON, reading its progress output, and degrading sensibly
when the diarizer is missing or fails. A fake `whisper-cli` exercises all of it.
"""

import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-local-"))

from transcribe import config                                    # noqa: E402
from transcribe.engines import base, registry                    # noqa: E402
from transcribe.engines.base import Context, EngineError         # noqa: E402
from transcribe.engines.local import _parse_whisper_json, _dtw_preset   # noqa: E402

_SAVED, _SAVED_ENV = {}, {}
_PATHS = ("HOME", "CONFIG_PATH", "UPLOAD_DIR", "JOB_DIR", "MODEL_DIR", "BIN_DIR")


def setUpModule():
    for attr in _PATHS:
        _SAVED[attr] = getattr(config, attr)
    home = pathlib.Path(tempfile.mkdtemp(prefix="transcribe-localengine-"))
    config.HOME = home
    config.CONFIG_PATH = home / "config.json"
    config.UPLOAD_DIR = home / "uploads"
    config.JOB_DIR = home / "jobs"
    config.MODEL_DIR = home / "models"
    config.BIN_DIR = home / "bin"
    _SAVED_ENV["WHISPER_CLI"] = os.environ.pop("WHISPER_CLI", None)
    config.ensure_dirs()
    config.load(force=True)


def tearDownModule():
    for attr, value in _SAVED.items():
        setattr(config, attr, value)
    if _SAVED_ENV.get("WHISPER_CLI"):
        os.environ["WHISPER_CLI"] = _SAVED_ENV["WHISPER_CLI"]
    else:
        os.environ.pop("WHISPER_CLI", None)
    config.load(force=True)


WHISPER_JSON = {
    "result": {"language": "en"},
    "params": {"language": "auto"},
    "transcription": [
        {"offsets": {"from": 0, "to": 400}, "text": " Hello"},
        {"offsets": {"from": 400, "to": 900}, "text": " everyone."},
        {"offsets": {"from": 1200, "to": 1600}, "text": " Hi"},
        {"offsets": {"from": 1600, "to": 2100}, "text": " there."},
    ],
}


def fake_whisper(path, *, fail_on_dtw=False, exit_code=0, emit_json=True):
    """A stand-in for whisper-cli that honours -of and prints progress."""
    script = f'''#!/bin/sh
out=""
dtw=0
while [ $# -gt 0 ]; do
  case "$1" in
    -of) out="$2"; shift 2 ;;
    -dtw) dtw=1; shift 2 ;;
    --help) echo "usage: whisper-cli"; exit 0 ;;
    *) shift ;;
  esac
done
if [ "$dtw" = "1" ] && [ "{int(fail_on_dtw)}" = "1" ]; then
  echo "error: unknown DTW preset"
  exit 3
fi
echo "whisper_print_progress_callback: progress =  25%"
echo "whisper_print_progress_callback: progress =  75%"
echo "whisper_print_progress_callback: progress = 100%"
if [ "{int(emit_json)}" = "1" ]; then
  cat > "$out.json" <<'JSONEOF'
{json.dumps(WHISPER_JSON)}
JSONEOF
fi
exit {exit_code}
'''
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


class ParsingTest(unittest.TestCase):
    def test_offsets_are_milliseconds(self):
        words = _parse_whisper_json(WHISPER_JSON)
        self.assertEqual([w.text for w in words], ["Hello", "everyone.", "Hi", "there."])
        self.assertEqual(words[0].start, 0.0)
        self.assertAlmostEqual(words[0].end, 0.4)
        self.assertAlmostEqual(words[2].start, 1.2)

    def test_non_speech_annotations_are_dropped(self):
        data = {"transcription": [
            {"offsets": {"from": 0, "to": 100}, "text": " [BLANK_AUDIO]"},
            {"offsets": {"from": 100, "to": 200}, "text": " (music)"},
            {"offsets": {"from": 200, "to": 400}, "text": " Hello"},
        ]}
        self.assertEqual([w.text for w in _parse_whisper_json(data)], ["Hello"])

    def test_zero_length_words_are_widened(self):
        data = {"transcription": [{"offsets": {"from": 900, "to": 900}, "text": " x"}]}
        w = _parse_whisper_json(data)[0]
        self.assertGreater(w.end, w.start)

    def test_missing_offsets_are_skipped(self):
        data = {"transcription": [{"text": " orphan"},
                                  {"offsets": {"from": 0, "to": 10}, "text": " ok"}]}
        self.assertEqual([w.text for w in _parse_whisper_json(data)], ["ok"])

    def test_empty_transcription(self):
        self.assertEqual(_parse_whisper_json({"transcription": []}), [])


class DtwPresetTest(unittest.TestCase):
    def test_known_models_map_to_valid_presets(self):
        """An unrecognised preset makes whisper-cli exit outright."""
        cases = {
            "ggml-large-v3-turbo-q5_0.bin": "large.v3.turbo",
            "ggml-large-v3-q5_0.bin": "large.v3",
            "ggml-large-v2.bin": "large.v2",
            "ggml-medium.en-q5_0.bin": "medium.en",
            "ggml-small.en-q5_1.bin": "small.en",
            "ggml-base.en-q5_1.bin": "base.en",
            "ggml-tiny.en-q5_1.bin": "tiny.en",
        }
        valid = {"tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large.v1", "large.v2", "large.v3",
                 "large.v3.turbo"}
        for name, expected in cases.items():
            got = _dtw_preset(name)
            self.assertEqual(got, expected, name)
            self.assertIn(got, valid, name)

    def test_unmappable_models_get_no_flag(self):
        # Plain "large" is not a preset, and distil/tdrz are other architectures.
        for name in ("ggml-large.bin", "distil-large-v3.5-ggml.bin",
                     "ggml-small.en-tdrz.bin", "something-else.bin"):
            self.assertEqual(_dtw_preset(name), "", name)


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.eng = base.get("local")
        for f in list(config.MODEL_DIR.glob("*")) + list(config.BIN_DIR.glob("*")):
            f.unlink()

    def test_unavailable_messages_name_the_fix(self):
        os.environ.pop("WHISPER_CLI", None)
        ok, reason = self.eng.available()
        if not shutil.which("whisper-cli"):
            self.assertFalse(ok)
            self.assertIn("install.sh --local", reason)

    def test_falls_back_to_any_model_present(self):
        """A filename mismatch should not defeat a working install."""
        config.save({"local_model": "ggml-does-not-exist.bin"})
        real = config.MODEL_DIR / "ggml-small.en-q5_1.bin"
        real.write_bytes(b"0" * 4096)
        self.assertEqual(self.eng.model_path().name, real.name)
        config.save({"local_model": "ggml-small.en-q5_1.bin"})

    def test_prefers_the_largest_model_when_falling_back(self):
        config.save({"local_model": "ggml-missing.bin"})
        (config.MODEL_DIR / "ggml-tiny.bin").write_bytes(b"0" * 1024)
        big = config.MODEL_DIR / "ggml-large-v3-turbo-q5_0.bin"
        big.write_bytes(b"0" * 8192)
        self.assertEqual(self.eng.model_path().name, big.name)

    def test_diarizer_models_prefer_int8(self):
        (config.MODEL_DIR / "pyannote-segmentation.onnx").write_bytes(b"0")
        (config.MODEL_DIR / "pyannote-segmentation.int8.onnx").write_bytes(b"0")
        (config.MODEL_DIR / "speaker-embedding.onnx").write_bytes(b"0")
        seg, emb = self.eng.diarizer_models()
        self.assertIn("int8", seg.name, "int8 is much faster on a phone CPU")
        self.assertIsNotNone(emb)

    def test_no_diarizer_models_returns_none(self):
        seg, emb = self.eng.diarizer_models()
        self.assertIsNone(seg)
        self.assertIsNone(emb)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.eng = base.get("local")
        for f in list(config.MODEL_DIR.glob("*")) + list(config.BIN_DIR.glob("*")):
            f.unlink()
        (config.MODEL_DIR / "ggml-small.en-q5_1.bin").write_bytes(b"0" * 4096)
        config.save({"local_model": "ggml-small.en-q5_1.bin", "local_diarize": False})
        self.bin = fake_whisper(config.BIN_DIR / "whisper-cli")
        self.wav = config.HOME / "input.wav"
        self.wav.write_bytes(b"RIFF")
        self.progress = []
        self.logs = []

    def ctx(self, **kw):
        c = Context(job_id="loc1", source=self.wav, duration=10.0,
                    workdir=config.UPLOAD_DIR, **kw)
        c.progress = lambda stage, frac: self.progress.append((stage, frac))
        c.log = self.logs.append
        # The WAV is already 16 kHz mono as far as this engine is concerned.
        c._wav = self.wav
        return c

    def test_transcribes_and_reports_progress(self):
        result = self.eng.transcribe(self.ctx())
        self.assertEqual([w.text for w in result.words],
                         ["Hello", "everyone.", "Hi", "there."])
        self.assertEqual(result.language, "en", "detected language comes from result.language")
        self.assertEqual(result.model, "ggml-small.en-q5_1.bin")
        stages = {s for s, _ in self.progress}
        self.assertIn("transcribing", stages)
        fractions = [f for s, f in self.progress if s == "transcribing"]
        self.assertGreater(max(fractions), 0.5, "progress output should be parsed")

    def test_detected_language_does_not_leak_between_jobs(self):
        """Engines are process-wide singletons; per-job state must not stick."""
        self.eng.transcribe(self.ctx())
        data = dict(WHISPER_JSON)
        data["result"] = {"language": ""}
        fake_whisper(self.bin)
        original = json.dumps(WHISPER_JSON)
        try:
            WHISPER_JSON["result"] = {"language": ""}
            fake_whisper(self.bin)
            second = self.eng.transcribe(self.ctx(language="auto"))
            self.assertEqual(second.language, "",
                             "a previous job's language must not carry over")
        finally:
            WHISPER_JSON.update(json.loads(original))
            fake_whisper(self.bin)

    def test_dtw_failure_retries_without_the_flag(self):
        """A bad preset must cost the flag, not the transcript."""
        fake_whisper(self.bin, fail_on_dtw=True)
        result = self.eng.transcribe(self.ctx())
        self.assertTrue(result.words)
        self.assertTrue(any("DTW" in m or "standard timestamps" in m for m in self.logs),
                        self.logs)

    def test_whisper_failure_surfaces_its_output(self):
        fake_whisper(self.bin, exit_code=2, emit_json=False)
        with self.assertRaises(EngineError) as cm:
            self.eng.transcribe(self.ctx())
        self.assertIn("whisper.cpp failed", str(cm.exception))

    def test_missing_json_is_a_clear_error(self):
        fake_whisper(self.bin, emit_json=False)
        with self.assertRaises(EngineError) as cm:
            self.eng.transcribe(self.ctx())
        self.assertIn("no JSON output", str(cm.exception))

    def test_missing_diarizer_keeps_the_transcript(self):
        """Half an hour of phone CPU must not be lost to a missing diarizer."""
        config.save({"local_diarize": True})
        try:
            result = self.eng.transcribe(self.ctx())
            self.assertTrue(result.words, "the transcript must survive")
            self.assertIsNone(result.turns)
            self.assertTrue(any("speaker" in m.lower() for m in self.logs), self.logs)
        finally:
            config.save({"local_diarize": False})

    def test_intermediate_json_is_cleaned_up(self):
        self.eng.transcribe(self.ctx())
        leftovers = list(config.UPLOAD_DIR.glob("*.whisper.json"))
        self.assertEqual(leftovers, [], leftovers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
