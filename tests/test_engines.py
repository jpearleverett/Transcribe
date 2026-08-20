"""Engine adapters against the documented provider response shapes.

Each fixture mirrors the response shape published in the provider's own API
reference. These are the parts most likely to rot — a provider renames a field
and the app silently produces an empty or single-speaker transcript — so the
shapes are pinned here rather than discovered in production.
"""

import json
import os
import sys
import pathlib
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Must be set before the engine registry is imported: it decides at import time
# whether the mock engine registers, and the module is cached thereafter — so
# whichever test module imports it first settles this for the whole process.
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-eng-"))

from transcribe import config, httpclient                      # noqa: E402
from transcribe.engines import base, registry                  # noqa: E402
from transcribe.engines.base import Context, EngineError       # noqa: E402

# config's paths are module globals resolved at import, so every test module in
# one process shares them. These tests write API keys and endpoint ids, which
# would otherwise leak into the server tests' assertions. Swap the paths for the
# duration of this module and put them back afterwards.
_SAVED = {}
_PATH_ATTRS = ("HOME", "CONFIG_PATH", "UPLOAD_DIR", "JOB_DIR", "MODEL_DIR", "BIN_DIR")
_ENV_KEYS_TO_CLEAR = list(config.ENV_KEYS.values()) + ["RUNPOD_ENDPOINT_ID"]
_SAVED_ENV = {}


def setUpModule():
    for attr in _PATH_ATTRS:
        _SAVED[attr] = getattr(config, attr)
    home = pathlib.Path(tempfile.mkdtemp(prefix="transcribe-engines-"))
    config.HOME = home
    config.CONFIG_PATH = home / "config.json"
    config.UPLOAD_DIR = home / "uploads"
    config.JOB_DIR = home / "jobs"
    config.MODEL_DIR = home / "models"
    config.BIN_DIR = home / "bin"
    # A real key in the environment would otherwise shadow the test fixtures.
    for name in _ENV_KEYS_TO_CLEAR:
        _SAVED_ENV[name] = os.environ.pop(name, None)
    config.ensure_dirs()
    config.load(force=True)


def tearDownModule():
    for attr, value in _SAVED.items():
        setattr(config, attr, value)
    for name, value in _SAVED_ENV.items():
        if value is not None:
            os.environ[name] = value
    config.load(force=True)


def make_ctx(**kw):
    """A context over a real, decodable WAV.

    It has to be genuine audio: where ffmpeg exists — as it does on Termux —
    the engines compress before uploading, and a stub file would exercise the
    fallback path instead of the one that actually runs on a phone.
    """
    src = config.HOME / "sample.wav"
    if not src.exists():
        import math
        import struct
        import wave
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(
                struct.pack("<h", int(9000 * math.sin(i / 14.0))) for i in range(16000)))
    return Context(job_id="testjob", source=src, duration=120.0,
                   workdir=src.parent, **kw)


class Patched:
    """Swap httpclient functions for canned responses, recording the calls."""

    def __init__(self, **responses):
        self.responses = responses
        self.calls = []
        self._orig = {}

    def __enter__(self):
        for name in ("get", "post", "upload_raw", "upload_multipart"):
            self._orig[name] = getattr(httpclient, name)
            setattr(httpclient, name, self._make(name))
        return self

    def _make(self, name):
        def fn(url, *a, **kw):
            self.calls.append({"fn": name, "url": url, **kw})
            value = self.responses.get(name)
            if callable(value):
                return value(url, *a, **kw)
            if isinstance(value, list):
                return value.pop(0) if value else {}
            return value if value is not None else {}
        return fn

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(httpclient, name, orig)

    def call_urls(self):
        return [c["url"] for c in self.calls]


def set_key(engine, key="test-key"):
    config.save({"keys": {engine: key}})
    assert config.api_key(engine) == key, (
        f"a stored key must win over any {config.ENV_KEYS.get(engine)} in the environment")


class ElevenLabsTest(unittest.TestCase):
    # Shape per the ElevenLabs speech-to-text reference: speaker_id sits on each
    # word, and "spacing"/"audio_event" entries are interleaved with real words.
    FIXTURE = {
        "language_code": "en", "language_probability": 0.99,
        "text": "Hello there. Hi!",
        "words": [
            {"text": "Hello", "type": "word", "start": 0.0, "end": 0.4,
             "speaker_id": "speaker_1", "logprob": -0.05},
            {"text": " ", "type": "spacing", "start": 0.4, "end": 0.42, "speaker_id": "speaker_1"},
            {"text": "there.", "type": "word", "start": 0.42, "end": 0.9, "speaker_id": "speaker_1"},
            {"text": "[door]", "type": "audio_event", "start": 0.9, "end": 1.0},
            {"text": "Hi!", "type": "word", "start": 1.2, "end": 1.6, "speaker_id": "speaker_2"},
        ],
    }

    def test_parses_words_and_speakers(self):
        set_key("elevenlabs")
        eng = base.get("elevenlabs")
        with Patched(upload_multipart=self.FIXTURE) as p:
            result = eng.transcribe(make_ctx())
        self.assertEqual([w.text for w in result.words], ["Hello", "there.", "Hi!"],
                         "spacing and audio_event entries must be dropped")
        # speaker_1 normalises to "1" so labels are consistent across providers.
        self.assertEqual([w.speaker for w in result.words], ["1", "1", "2"])
        self.assertEqual(result.language, "en")
        self.assertEqual(result.model, "scribe_v2")
        self.assertAlmostEqual(result.words[0].end, 0.4)
        self.assertIsNotNone(result.words[0].confidence)

        sent = p.calls[0]
        self.assertIn("api.elevenlabs.io/v1/speech-to-text", sent["url"])
        self.assertEqual(sent["headers"]["xi-api-key"], "test-key")
        self.assertEqual(sent["fields"]["model_id"], "scribe_v2")
        self.assertEqual(sent["fields"]["diarize"], "true")
        self.assertEqual(sent["fields"]["timestamps_granularity"], "word")

    def test_num_speakers_forwarded(self):
        set_key("elevenlabs")
        with Patched(upload_multipart=self.FIXTURE) as p:
            base.get("elevenlabs").transcribe(make_ctx(num_speakers=3))
        self.assertEqual(p.calls[0]["fields"]["num_speakers"], "3")

    def test_max_speakers_drives_num_speakers(self):
        """ElevenLabs documents num_speakers as the maximum, not an exact count."""
        set_key("elevenlabs")
        with Patched(upload_multipart=self.FIXTURE) as p:
            base.get("elevenlabs").transcribe(make_ctx(max_speakers=5))
        self.assertEqual(p.calls[0]["fields"]["num_speakers"], "5")

    def test_empty_response_raises(self):
        set_key("elevenlabs")
        with Patched(upload_multipart={"words": [], "text": ""}):
            with self.assertRaises(EngineError):
                base.get("elevenlabs").transcribe(make_ctx())

    def test_garbage_response_raises_clearly(self):
        set_key("elevenlabs")
        with Patched(upload_multipart={"detail": "quota exceeded"}):
            with self.assertRaises(EngineError) as cm:
                base.get("elevenlabs").transcribe(make_ctx())
        self.assertIn("quota exceeded", str(cm.exception))

    def test_missing_key_raises_before_upload(self):
        config.save({"keys": {"elevenlabs": ""}})
        with Patched(upload_multipart=self.FIXTURE) as p:
            with self.assertRaises(EngineError):
                base.get("elevenlabs").transcribe(make_ctx())
        self.assertEqual(p.calls, [], "must not upload audio without a key")


class DeepgramTest(unittest.TestCase):
    FIXTURE = {
        "results": {
            "channels": [{
                "detected_language": "en",
                "alternatives": [{
                    "transcript": "Hello there. Hi.",
                    "words": [
                        {"word": "hello", "punctuated_word": "Hello", "start": 0.0,
                         "end": 0.4, "confidence": 0.99, "speaker": 0,
                         "speaker_confidence": 0.87},
                        {"word": "there", "punctuated_word": "there.", "start": 0.4,
                         "end": 0.9, "confidence": 0.98, "speaker": 0},
                        {"word": "hi", "punctuated_word": "Hi.", "start": 1.2,
                         "end": 1.6, "confidence": 0.97, "speaker": 1},
                    ],
                }],
            }],
        },
    }

    def test_diarization_parameter_is_diarize_model_only(self):
        set_key("deepgram")
        with Patched(upload_raw=self.FIXTURE) as p:
            result = base.get("deepgram").transcribe(make_ctx())
        url = p.calls[0]["url"]
        # diarize_model both enables diarization and picks v2. The deprecated
        # diarize=true would silently route to v1 — and, critically, Deepgram
        # REJECTS any request that sets both, so sending them together as
        # belt-and-braces 400s every single call.
        self.assertIn("diarize_model=latest", url)
        self.assertNotIn("diarize=true", url)
        self.assertNotIn("diarize=", url.replace("diarize_model=", ""))
        self.assertIn("model=nova-3", url)
        self.assertEqual(p.calls[0]["headers"]["Authorization"], "Token test-key")
        self.assertEqual([w.text for w in result.words], ["Hello", "there.", "Hi."],
                         "punctuated_word should win over the bare word")
        self.assertEqual([w.speaker for w in result.words], ["0", "0", "1"])
        self.assertEqual(result.language, "en")

    def test_language_pin_disables_detection(self):
        set_key("deepgram")
        with Patched(upload_raw=self.FIXTURE) as p:
            base.get("deepgram").transcribe(make_ctx(language="es"))
        url = p.calls[0]["url"]
        self.assertIn("language=es", url)
        self.assertNotIn("detect_language", url)

    def test_missing_speakers_is_reported_not_silent(self):
        """A 200 with no speaker labels must be surfaced, not shown as 1 speaker."""
        set_key("deepgram")
        no_speakers = {"results": {"channels": [{"alternatives": [{
            "transcript": "Hello there.",
            "words": [{"word": "hello", "punctuated_word": "Hello", "start": 0.0,
                       "end": 0.4, "confidence": 0.99}],
        }]}]}}
        logged = []
        ctx = make_ctx()
        ctx.log = logged.append
        with Patched(upload_raw=no_speakers):
            result = base.get("deepgram").transcribe(ctx)
        self.assertTrue(result.words)
        self.assertIsNone(result.words[0].speaker)
        self.assertTrue(any("no speaker labels" in m for m in logged),
                        f"expected a warning in the job log, got {logged}")

    def test_malformed_response_raises(self):
        set_key("deepgram")
        with Patched(upload_raw={"results": {}}):
            with self.assertRaises(EngineError):
                base.get("deepgram").transcribe(make_ctx())


class AssemblyAITest(unittest.TestCase):
    # AssemblyAI timestamps are MILLISECONDS, unlike every other provider here.
    COMPLETED = {
        "status": "completed", "language_code": "en",
        "speech_model_used": "universal-3-5-pro",
        "text": "Hello there. Hi.",
        "utterances": [
            {"speaker": "A", "words": [
                {"text": "Hello", "start": 0, "end": 400, "confidence": 0.99},
                {"text": "there.", "start": 400, "end": 900, "confidence": 0.98},
            ]},
            {"speaker": "B", "words": [
                {"text": "Hi.", "start": 1200, "end": 1600, "confidence": 0.97},
            ]},
        ],
    }

    def test_full_flow_and_millisecond_conversion(self):
        set_key("assemblyai")
        with Patched(upload_raw={"upload_url": "https://cdn.assemblyai.com/x"},
                     post={"id": "tid-1", "status": "queued"},
                     get=self.COMPLETED) as p:
            result = base.get("assemblyai").transcribe(make_ctx(num_speakers=2))

        submit = next(c for c in p.calls if c["fn"] == "post")
        body = submit["json_body"]
        self.assertEqual(body["speech_models"], ["universal-3-5-pro"],
                         "the field is plural; the singular form is deprecated")
        self.assertTrue(body["speaker_labels"])
        self.assertTrue(body["punctuate"], "speaker_labels requires punctuate")
        self.assertEqual(body["speakers_expected"], 2)
        self.assertEqual(body["audio_url"], "https://cdn.assemblyai.com/x")

        self.assertEqual([w.text for w in result.words], ["Hello", "there.", "Hi."])
        self.assertEqual([w.speaker for w in result.words], ["A", "A", "B"])
        self.assertAlmostEqual(result.words[0].end, 0.4, msg="ms must become seconds")
        self.assertAlmostEqual(result.words[2].start, 1.2)
        self.assertEqual(result.model, "universal-3-5-pro")

    def test_speaker_bounds_use_speaker_options(self):
        """speakers_expected and speaker_options are mutually exclusive."""
        set_key("assemblyai")
        with Patched(upload_raw={"upload_url": "u"}, post={"id": "t"},
                     get=self.COMPLETED) as p:
            base.get("assemblyai").transcribe(make_ctx(min_speakers=2, max_speakers=4))
        body = next(c for c in p.calls if c["fn"] == "post")["json_body"]
        self.assertEqual(body["speaker_options"],
                         {"min_speakers_expected": 2, "max_speakers_expected": 4})
        self.assertNotIn("speakers_expected", body)

        with Patched(upload_raw={"upload_url": "u"}, post={"id": "t"},
                     get=self.COMPLETED) as p:
            base.get("assemblyai").transcribe(make_ctx(num_speakers=3, max_speakers=4))
        body = next(c for c in p.calls if c["fn"] == "post")["json_body"]
        self.assertEqual(body["speakers_expected"], 3)
        self.assertNotIn("speaker_options", body)

    def test_polls_until_completed(self):
        set_key("assemblyai")
        pending = {"status": "processing"}
        with Patched(upload_raw={"upload_url": "u"},
                     post={"id": "tid-1"},
                     get=[pending, pending, self.COMPLETED]) as p:
            result = base.get("assemblyai").transcribe(make_ctx())
        self.assertEqual(len([c for c in p.calls if c["fn"] == "get"]), 3)
        self.assertTrue(result.words)

    def test_error_status_raises_with_reason(self):
        set_key("assemblyai")
        with Patched(upload_raw={"upload_url": "u"}, post={"id": "t"},
                     get={"status": "error", "error": "Download error"}):
            with self.assertRaises(EngineError) as cm:
                base.get("assemblyai").transcribe(make_ctx())
        self.assertIn("Download error", str(cm.exception))

    def test_null_timestamp_does_not_crash(self):
        """A present-but-null timestamp must be skipped, not raise TypeError."""
        set_key("assemblyai")
        nulls = {"status": "completed", "text": "Hey there.",
                 "utterances": [{"speaker": "A", "words": [
                     {"text": "Hey", "start": None, "end": None},
                     {"text": "there.", "start": 100, "end": None},
                 ]}]}
        with Patched(upload_raw={"upload_url": "u"}, post={"id": "t"}, get=nulls):
            result = base.get("assemblyai").transcribe(make_ctx())
        self.assertEqual([w.text for w in result.words], ["there."])
        self.assertGreater(result.words[0].end, result.words[0].start)

    def test_model_substitution_is_noticed(self):
        set_key("assemblyai")
        swapped = dict(self.COMPLETED)
        swapped.pop("speech_model_used", None)
        swapped["speech_model"] = "universal-2"
        logged = []
        ctx = make_ctx()
        ctx.log = logged.append
        with Patched(upload_raw={"upload_url": "u"}, post={"id": "t"}, get=swapped):
            result = base.get("assemblyai").transcribe(ctx)
        self.assertEqual(result.model, "universal-2")
        self.assertTrue(any("universal-2" in m for m in logged), logged)

    def test_falls_back_to_flat_word_list(self):
        set_key("assemblyai")
        flat = {"status": "completed", "text": "Hey.",
                "words": [{"text": "Hey.", "start": 100, "end": 500, "speaker": "A"}]}
        with Patched(upload_raw={"upload_url": "u"}, post={"id": "t"}, get=flat):
            result = base.get("assemblyai").transcribe(make_ctx())
        self.assertEqual(result.words[0].speaker, "A")
        self.assertAlmostEqual(result.words[0].start, 0.1)


class OpenAITest(unittest.TestCase):
    FIXTURE = {
        "text": "How can I help? I need a refund.",
        "segments": [
            {"speaker": "agent", "text": "How can I help?", "start": 0.5, "end": 2.1},
            {"speaker": "customer", "text": "I need a refund.", "start": 2.3, "end": 4.0},
        ],
    }

    def test_returns_segments_not_fabricated_words(self):
        set_key("openai")
        with Patched(upload_multipart=self.FIXTURE) as p:
            result = base.get("openai").transcribe(make_ctx())
        self.assertEqual(result.words, [], "OpenAI gives no word timings; do not invent them")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].speaker, "agent")
        self.assertAlmostEqual(result.segments[1].start, 2.3)
        fields = p.calls[0]["fields"]
        self.assertEqual(fields["response_format"], "diarized_json")
        self.assertEqual(fields["chunking_strategy"], "auto",
                         "required for audio longer than 30 seconds")

    def test_oversize_without_ffmpeg_explains_itself(self):
        set_key("openai")
        big = config.HOME / "big.wav"
        big.write_bytes(b"0" * (26 * 1024 * 1024))
        ctx = Context(job_id="j2", source=big, duration=0.0, workdir=big.parent)
        from transcribe import audio as audio_mod
        orig, audio_mod.FFMPEG = audio_mod.FFMPEG, None
        try:
            with Patched(upload_multipart=self.FIXTURE):
                with self.assertRaises(EngineError) as cm:
                    base.get("openai").transcribe(ctx)
        finally:
            audio_mod.FFMPEG = orig
        self.assertIn("25 MB", str(cm.exception))
        big.unlink()


class RunPodTest(unittest.TestCase):
    OUTPUT = {
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.99},
            {"word": "there.", "start": 0.4, "end": 0.9, "score": 0.98},
        ],
        "turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        "language": "en", "model": "large-v3", "text": "Hello there.",
    }

    def setUp(self):
        set_key("runpod")
        config.save({"runpod_endpoint": "ep123", "runpod_audio_url": ""})

    def test_unavailable_without_endpoint(self):
        config.save({"runpod_endpoint": ""})
        ok, reason = base.get("runpod").available()
        self.assertFalse(ok)
        self.assertIn("endpoint", reason.lower())

    def test_submit_poll_and_parse(self):
        with Patched(post={"id": "job-1", "status": "IN_QUEUE"},
                     get={"status": "COMPLETED", "output": self.OUTPUT}) as p:
            ctx = make_ctx()
            ctx.duration = 60.0
            result = base.get("runpod").transcribe(ctx)

        submit = p.calls[0]
        self.assertEqual(submit["url"], "https://api.runpod.ai/v2/ep123/run",
                         "api.runpod.ai is the invoke host; .io is the management API")
        self.assertEqual(submit["headers"]["Authorization"], "Bearer test-key")
        self.assertIn("input", submit["json_body"])
        self.assertIn("policy", submit["json_body"])
        self.assertGreaterEqual(submit["json_body"]["policy"]["executionTimeout"], 900_000,
                                "executionTimeout is milliseconds")
        self.assertIn("audio_base64", submit["json_body"]["input"])

        self.assertEqual([w.text for w in result.words], ["Hello", "there."])
        self.assertEqual(result.turns[0].speaker, "0", "SPEAKER_00 normalises to 0")
        self.assertEqual(result.model, "large-v3")

    def test_runsync_result_returned_inline(self):
        with Patched(post={"status": "COMPLETED", "output": self.OUTPUT}):
            result = base.get("runpod").transcribe(make_ctx())
        self.assertTrue(result.words)

    def test_failed_job_surfaces_error(self):
        with Patched(post={"id": "j"}, get={"status": "FAILED", "error": "OOM"}):
            with self.assertRaises(EngineError) as cm:
                base.get("runpod").transcribe(make_ctx())
        self.assertIn("OOM", str(cm.exception))

    def test_worker_error_surfaces(self):
        with Patched(post={"id": "j"},
                     get={"status": "COMPLETED", "output": {"error": "no audio supplied"}}):
            with self.assertRaises(EngineError) as cm:
                base.get("runpod").transcribe(make_ctx())
        self.assertIn("no audio supplied", str(cm.exception))

    def test_whisperx_shaped_output_also_parses(self):
        whisperx = {"segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello there.", "speaker": "SPEAKER_01",
             "words": [{"word": "Hello", "start": 0.0, "end": 0.4},
                       {"word": "there.", "start": 0.4, "end": 0.9}]},
        ], "language": "en"}
        with Patched(post={"id": "j"}, get={"status": "COMPLETED", "output": whisperx}):
            result = base.get("runpod").transcribe(make_ctx())
        self.assertEqual([w.text for w in result.words], ["Hello", "there."])
        self.assertEqual(result.words[0].speaker, "1")

    def test_audio_url_skips_base64(self):
        config.save({"runpod_audio_url": "https://example.com/a.mp3"})
        with Patched(post={"id": "j"}, get={"status": "COMPLETED", "output": self.OUTPUT}) as p:
            base.get("runpod").transcribe(make_ctx())
        body = p.calls[0]["json_body"]["input"]
        self.assertEqual(body["audio_url"], "https://example.com/a.mp3")
        self.assertNotIn("audio_base64", body)
        config.save({"runpod_audio_url": ""})

    def test_payload_budget_rejects_absurdly_long_audio(self):
        from transcribe.engines.runpod import _fit_payload
        ctx = make_ctx()
        ctx.duration = 20 * 3600.0        # 20 hours
        from transcribe import audio as audio_mod
        orig, audio_mod.FFMPEG = audio_mod.FFMPEG, "/usr/bin/ffmpeg"
        try:
            with self.assertRaises(EngineError) as cm:
                _fit_payload(ctx, 19)
        finally:
            audio_mod.FFMPEG = orig
        self.assertIn("hours", str(cm.exception))


    def test_cancel_reaches_runpod(self):
        """Cancelling must actually stop the GPU job, not just the local poll.

        The poll loop spends nearly all its time asleep or inside a 60s GET, so
        a cancel issued at the top of the loop was never reached and the GPU
        kept running — and billing.
        """
        state = {"polls": 0}

        def fake_get(url, *a, **kw):
            state["polls"] += 1
            return {"status": "IN_PROGRESS"}

        ctx = make_ctx()
        ctx.cancelled = lambda: state["polls"] >= 1     # cancel after first poll

        with Patched(post={"id": "job-9"}, get=fake_get) as p:
            with self.assertRaises(base.Cancelled):
                base.get("runpod").transcribe(ctx)

        cancels = [c["url"] for c in p.calls if "/cancel/" in c["url"]]
        self.assertEqual(cancels, ["https://api.runpod.ai/v2/ep123/cancel/job-9"],
                         "a cancelled job must be cancelled remotely, exactly once")

    def test_submit_is_not_retried(self):
        """/run enqueues a job, so a replay would start a second GPU run."""
        with Patched(post={"id": "j"}, get={"status": "COMPLETED", "output": self.OUTPUT}) as p:
            base.get("runpod").transcribe(make_ctx())
        submit = p.calls[0]
        self.assertEqual(submit.get("retries"), 0,
                         "submitting is not idempotent and must not be retried")

    def test_oversized_payload_rejected_when_duration_unknown(self):
        """With no duration, compression can't be sized — check the bytes."""
        from transcribe.engines import runpod as rp
        from transcribe import audio as audio_mod
        oversized = int(rp.RUNSYNC_LIMIT_MB * 1024 * 1024 * 0.85)   # ~23 MB once base64'd

        def fake_compress(src, dst, bitrate="48k"):
            # duration==0 means the bitrate could not be sized, so compression
            # runs at the default and can still land over the limit.
            dst.write_bytes(b"0" * oversized)
            return dst

        big = config.HOME / "unknown.wav"
        big.write_bytes(b"0" * 1024)
        ctx = Context(job_id="j9", source=big, duration=0.0, workdir=big.parent)
        orig_ff, audio_mod.FFMPEG = audio_mod.FFMPEG, "/usr/bin/ffmpeg"
        orig_c, audio_mod.to_compressed = audio_mod.to_compressed, fake_compress
        try:
            # A terminal get() so a regression in the guard fails this test
            # fast instead of polling forever and hanging the whole suite.
            with Patched(post={"id": "j"},
                         get={"status": "FAILED", "error": "should not have submitted"}) as p:
                with self.assertRaises(EngineError) as cm:
                    base.get("runpod").transcribe(ctx)
        finally:
            audio_mod.FFMPEG = orig_ff
            audio_mod.to_compressed = orig_c
            big.unlink()
        self.assertIn("request limit", str(cm.exception))
        self.assertEqual(p.calls, [], "must fail before submitting an oversized body")

    def test_temp_audio_is_deleted_after_encoding(self):
        from transcribe import audio as audio_mod
        made = {}

        def fake_compress(src, dst, bitrate="48k"):
            dst.write_bytes(b"opusdata" * 100)
            made["path"] = dst
            return dst

        orig_ff, audio_mod.FFMPEG = audio_mod.FFMPEG, "/usr/bin/ffmpeg"
        orig_c, audio_mod.to_compressed = audio_mod.to_compressed, fake_compress
        try:
            with Patched(post={"id": "j"}, get={"status": "COMPLETED", "output": self.OUTPUT}):
                base.get("runpod").transcribe(make_ctx())
        finally:
            audio_mod.FFMPEG = orig_ff
            audio_mod.to_compressed = orig_c
        self.assertIn("path", made)
        self.assertFalse(made["path"].exists(),
                         "the compressed copy must not be left on the phone's disk")


class RegistryTest(unittest.TestCase):
    def test_all_engines_describe_cleanly(self):
        for eng in base.all_engines():
            d = eng.describe()
            for field in ("name", "label", "description", "needs_key",
                          "has_key", "available", "config_fields"):
                self.assertIn(field, d, eng.name)
            json.dumps(d)      # must be serialisable for the API

    def test_unknown_engine_raises(self):
        with self.assertRaises(EngineError):
            base.get("does-not-exist")

    def test_speaker_normalisation(self):
        self.assertEqual(base.normalize_speaker("SPEAKER_00"), "0")
        self.assertEqual(base.normalize_speaker("SPEAKER_07"), "7")
        self.assertEqual(base.normalize_speaker(0), "0")
        self.assertEqual(base.normalize_speaker("A"), "A")
        self.assertEqual(base.normalize_speaker("A:"), "A", "some providers append a colon")
        self.assertEqual(base.normalize_speaker("agent:"), "agent")
        self.assertIsNone(base.normalize_speaker(None))
        self.assertIsNone(base.normalize_speaker("  "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
