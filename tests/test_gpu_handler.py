"""The GPU worker's logic, and its contract with the client that calls it.

The worker needs a CUDA GPU and several gigabytes of weights, so it cannot run
here. What *can* be tested — and what actually breaks in practice — is the
plumbing: decoding the audio the phone sent, the shape of what comes back, and
whether the client can parse it. The two halves are written independently and
talk over JSON, so this pins that seam from both sides.
"""

import base64
import importlib.util
import json
import os
import pathlib
import struct
import sys
import tempfile
import types
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-gpu-"))


# --------------------------------------------------------------------------
# Stand-ins for the heavy GPU libraries
# --------------------------------------------------------------------------

class FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class FakeSegment:
    def __init__(self, text, words):
        self.text, self.words = text, words


class FakeInfo:
    language = "en"
    duration = 3.0


class FakeWhisperModel:
    last_kwargs = None

    def __init__(self, name, device=None, compute_type=None):
        self.name = name
        FakeWhisperModel.instances.append(name)

    instances = []

    def transcribe(self, path, **kwargs):
        FakeWhisperModel.last_kwargs = kwargs
        segs = [
            FakeSegment(" Hello there.", [FakeWord("Hello", 0.0, 0.4),
                                          FakeWord("there.", 0.4, 0.9)]),
            FakeSegment(" Hi.", [FakeWord("Hi.", 1.2, 1.6)]),
        ]
        return iter(segs), FakeInfo()


class FakeBatched:
    def __init__(self, model=None):
        self.model = model

    def transcribe(self, path, batch_size=None, **kwargs):
        return self.model.transcribe(path, **kwargs)


class FakeTrack:
    def __init__(self, start, end):
        self.start, self.end = start, end


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        yield FakeTrack(0.0, 1.0), None, "SPEAKER_00"
        yield FakeTrack(1.1, 2.0), None, "SPEAKER_01"


class FakePipeline:
    last_kwargs = None
    should_fail = False

    @classmethod
    def from_pretrained(cls, name, token=None):
        return cls()

    def to(self, device):
        return self

    def __call__(self, path, **kwargs):
        FakePipeline.last_kwargs = kwargs
        if FakePipeline.should_fail:
            raise RuntimeError("diarizer exploded")
        return FakeAnnotation()


def install_fakes():
    runpod_mod = types.ModuleType("runpod")
    runpod_mod.serverless = types.SimpleNamespace(start=lambda cfg: None)
    sys.modules["runpod"] = runpod_mod

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = FakeWhisperModel
    fw.BatchedInferencePipeline = FakeBatched
    sys.modules["faster_whisper"] = fw

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    sys.modules["torch"] = torch

    pa = types.ModuleType("pyannote")
    pa_audio = types.ModuleType("pyannote.audio")
    pa_audio.Pipeline = FakePipeline
    pa.audio = pa_audio
    sys.modules["pyannote"] = pa
    sys.modules["pyannote.audio"] = pa_audio


def load_handler():
    install_fakes()
    spec = importlib.util.spec_from_file_location("gpu_handler", ROOT / "gpu" / "handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


handler = load_handler()


def wav_bytes(seconds=1.0, rate=16000):
    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(buf.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", (i * 37) % 3000 - 1500)
                               for i in range(int(rate * seconds))))
    data = pathlib.Path(buf.name).read_bytes()
    os.unlink(buf.name)
    return data


HAVE_FFMPEG = bool(__import__("shutil").which("ffmpeg"))


class InputHandlingTest(unittest.TestCase):
    def setUp(self):
        handler._whisper = None
        handler._batched = None
        handler._whisper_name = None
        handler._diarizer = None
        FakePipeline.should_fail = False
        FakeWhisperModel.instances = []

    def test_missing_audio_is_a_clear_error(self):
        out = handler.handler({"input": {}})
        self.assertIn("error", out)
        self.assertIn("audio_base64", out["error"])

    def test_empty_audio_is_rejected(self):
        out = handler.handler({"input": {"audio_base64": ""}})
        self.assertIn("error", out)

    def test_never_raises_out_of_the_handler(self):
        """RunPod needs a JSON result, not a traceback."""
        for bad in ({"audio_base64": "not-base64!!!"},
                    {"audio_url": "http://127.0.0.1:1/nope"},
                    {"audio_base64": base64.b64encode(b"not audio").decode()}):
            out = handler.handler({"input": bad})
            self.assertIsInstance(out, dict)
            self.assertIn("error", out, bad)
            json.dumps(out)          # must be serialisable back to the caller

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg to decode")
    def test_data_url_prefix_is_tolerated(self):
        payload = "data:audio/wav;base64," + base64.b64encode(wav_bytes()).decode()
        out = handler.handler({"input": {"audio_base64": payload, "diarize": False}})
        self.assertNotIn("error", out)
        self.assertTrue(out["words"])


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg to decode the audio")
class TranscriptionTest(unittest.TestCase):
    def setUp(self):
        handler._whisper = None
        handler._batched = None
        handler._whisper_name = None
        handler._diarizer = None
        FakePipeline.should_fail = False
        FakeWhisperModel.instances = []

    def run_job(self, **extra):
        payload = {"audio_base64": base64.b64encode(wav_bytes()).decode()}
        payload.update(extra)
        return handler.handler({"input": payload})

    def test_returns_words_and_turns(self):
        out = self.run_job()
        self.assertEqual([w["word"] for w in out["words"]], ["Hello", "there.", "Hi."])
        self.assertEqual(out["words"][0]["start"], 0.0)
        self.assertEqual([t["speaker"] for t in out["turns"]], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(out["language"], "en")
        self.assertIn("elapsed", out)

    def test_hallucination_guards_are_set(self):
        """Whisper loops on silence without these; they are the documented fix."""
        self.run_job()
        kw = FakeWhisperModel.last_kwargs
        self.assertFalse(kw["condition_on_previous_text"])
        self.assertTrue(kw["word_timestamps"], "word timings are the whole point")
        self.assertTrue(kw["vad_filter"])
        self.assertEqual(kw["compression_ratio_threshold"], 2.4)
        self.assertEqual(kw["no_speech_threshold"], 0.6)

    def test_speaker_hints_reach_the_diarizer(self):
        self.run_job(num_speakers=3)
        self.assertEqual(FakePipeline.last_kwargs, {"num_speakers": 3})
        self.run_job(min_speakers=2, max_speakers=5)
        self.assertEqual(FakePipeline.last_kwargs,
                         {"min_speakers": 2, "max_speakers": 5})

    def test_diarization_failure_keeps_the_transcript(self):
        """Losing speaker labels must not lose an expensive transcript."""
        FakePipeline.should_fail = True
        out = self.run_job()
        self.assertTrue(out["words"], "the transcript must survive")
        self.assertEqual(out["turns"], [])
        self.assertIn("diarizer exploded", out["diarization_error"])

    def test_diarize_can_be_switched_off(self):
        out = self.run_job(diarize=False)
        self.assertNotIn("turns", out)

    def test_requested_model_is_honoured(self):
        out = self.run_job(model="large-v3-turbo")
        self.assertIn("large-v3-turbo", FakeWhisperModel.instances)
        self.assertEqual(out["model"], "large-v3-turbo")

    def test_model_is_cached_across_jobs_but_reloaded_on_change(self):
        self.run_job(model="large-v3")
        self.run_job(model="large-v3")
        self.assertEqual(FakeWhisperModel.instances, ["large-v3"], "should load once")
        self.run_job(model="small")
        self.assertEqual(FakeWhisperModel.instances, ["large-v3", "small"])


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg to decode the audio")
class ContractTest(unittest.TestCase):
    """The worker's output must be exactly what the phone-side client expects."""

    def test_worker_output_parses_in_the_client(self):
        handler._whisper = handler._batched = handler._whisper_name = None
        handler._diarizer = None
        FakePipeline.should_fail = False

        out = handler.handler({"input": {
            "audio_base64": base64.b64encode(wav_bytes()).decode(),
            "num_speakers": 2,
        }})
        self.assertNotIn("error", out)

        # Round-trip through JSON exactly as RunPod would deliver it.
        delivered = json.loads(json.dumps(out))

        from transcribe.engines.runpod import _parse_output
        from transcribe.engines.base import Context
        ctx = Context(job_id="c1", source=pathlib.Path("/dev/null"), duration=3.0)
        result = _parse_output(delivered, ctx, "large-v3")

        self.assertEqual([w.text for w in result.words], ["Hello", "there.", "Hi."])
        self.assertEqual([t.speaker for t in result.turns], ["0", "1"],
                         "SPEAKER_00 must normalise to 0 on the client")
        self.assertEqual(result.language, "en")

        # And the full attribution pipeline runs on it end to end.
        from transcribe.align import diarize_transcript
        segments = diarize_transcript(result.words, result.turns)
        self.assertEqual([s.speaker for s in segments], ["0", "1"])
        self.assertEqual(segments[0].text, "Hello there.")

    def test_worker_error_is_surfaced_by_the_client(self):
        from transcribe.engines.runpod import _parse_output
        from transcribe.engines.base import Context, EngineError
        ctx = Context(job_id="c2", source=pathlib.Path("/dev/null"), duration=1.0)
        with self.assertRaises(EngineError) as cm:
            _parse_output({"error": "ValueError: No audio supplied."}, ctx, "large-v3")
        self.assertIn("No audio supplied", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
