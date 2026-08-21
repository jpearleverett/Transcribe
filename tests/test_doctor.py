"""The setup check has to be reliable — it is what a user runs when nothing works."""

import io
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-doc-"))

from transcribe import config, doctor                      # noqa: E402

_SAVED = {}
_PATH_ATTRS = ("HOME", "CONFIG_PATH", "UPLOAD_DIR", "JOB_DIR", "MODEL_DIR", "BIN_DIR")
_SAVED_ENV = {}
_ENV_NAMES = list(config.ENV_KEYS.values()) + ["RUNPOD_ENDPOINT_ID"]


def setUpModule():
    for attr in _PATH_ATTRS:
        _SAVED[attr] = getattr(config, attr)
    home = pathlib.Path(tempfile.mkdtemp(prefix="transcribe-doctor-"))
    config.HOME = home
    config.CONFIG_PATH = home / "config.json"
    config.UPLOAD_DIR = home / "uploads"
    config.JOB_DIR = home / "jobs"
    config.MODEL_DIR = home / "models"
    config.BIN_DIR = home / "bin"
    for name in _ENV_NAMES:
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


def run_check():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = doctor.run(network=False)
    return code, buf.getvalue()


class DoctorTest(unittest.TestCase):
    def test_runs_without_network_and_reports(self):
        code, out = run_check()
        for heading in ("Python", "Audio tools", "Video tools", "Storage",
                        "Network", "Engines", "Self-test"):
            self.assertIn(heading, out, f"missing section: {heading}")
        self.assertIn("attribution and all six export formats work", out,
                      "the pipeline self-test must actually run")
        self.assertIn(str(config.HOME), out)
        # No engine configured in the test environment, so it must say so.
        self.assertIn("No engine is ready", out)
        self.assertEqual(code, 1, "an unusable setup must exit non-zero")

    def test_reports_ready_once_an_engine_has_a_key(self):
        config.save({"keys": {"deepgram": "test-key-1234567890"}})
        try:
            _, out = run_check()
            self.assertNotIn("No engine is ready", out)
            self.assertIn("Deepgram", out)
        finally:
            config.save({"keys": {"deepgram": ""}})

    def test_never_prints_a_key(self):
        secret = "sk-super-secret-value-abcdef123456"
        config.save({"keys": {"openai": secret}})
        try:
            _, out = run_check()
            self.assertNotIn(secret, out, "the check must never echo a key")
        finally:
            config.save({"keys": {"openai": ""}})

    def test_flags_a_world_readable_config(self):
        config.save({"keys": {"deepgram": "k" * 20}})
        try:
            config.CONFIG_PATH.chmod(0o644)
            _, out = run_check()
            self.assertIn("expected 600", out)
            self.assertIn("chmod 600", out, "must give the exact fix")
        finally:
            config.CONFIG_PATH.chmod(0o600)
            config.save({"keys": {"deepgram": ""}})

    def test_missing_ffmpeg_is_a_problem_with_a_fix(self):
        from transcribe import audio
        saved = audio.FFMPEG
        audio.FFMPEG = None
        try:
            code, out = run_check()
            self.assertIn("ffmpeg is missing", out)
            self.assertIn("pkg install ffmpeg", out)
            self.assertEqual(code, 1)
        finally:
            audio.FFMPEG = saved

    def test_video_section_reports_what_the_build_can_do(self):
        """Whether a phone has a hardware encoder decides hours of waiting."""
        _code, out = run_check()
        video_part = out.split("Video tools", 1)[1].split("Storage", 1)[0]
        self.assertIn("libx264", video_part)
        self.assertTrue("hardware encoder available" in video_part
                        or "no hardware video encoder" in video_part, video_part)
        self.assertIn("zscale", video_part,
                      "HDR handling is worth a line either way")

    def test_missing_ffmpeg_skips_the_video_section_rather_than_crashing(self):
        from transcribe import audio
        saved = audio.FFMPEG
        audio.FFMPEG = None
        try:
            _code, out = run_check()
            self.assertNotIn("Video tools", out,
                             "with no ffmpeg there is nothing to report about it")
        finally:
            audio.FFMPEG = saved

    def test_survives_an_unreadable_home(self):
        saved = config.HOME
        config.HOME = pathlib.Path("/proc/nonexistent/nope")
        try:
            code, out = run_check()          # must not raise
            self.assertIn("Storage", out)
        finally:
            config.HOME = saved

    def test_report_counts_levels(self):
        rep = doctor.Report()
        rep.add(doctor.OK, "fine")
        rep.add(doctor.WARN, "hmm")
        rep.add(doctor.BAD, "broken", "do this")
        self.assertEqual((rep.problems, rep.warnings), (1, 1))
        self.assertIn("do this", rep.render())


if __name__ == "__main__":
    unittest.main(verbosity=2)
