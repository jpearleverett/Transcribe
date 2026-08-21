"""Browsing device media and extracting audio from local video.

The point of this path is that nothing is uploaded and nothing is copied: a
29 GB video read in place costs no extra disk, where an upload would make the
phone hold a second copy of it to accomplish nothing.

That makes one property safety-critical — the app must never delete a file the
user already had.
"""

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-local-media-"))

from http.server import ThreadingHTTPServer                          # noqa: E402
from transcribe import audio, config, files, jobs as jobs_mod, runner, server  # noqa: E402
from transcribe.engines import registry                              # noqa: E402

HAVE_FFMPEG = audio.have_ffmpeg()

_SAVED = {}
_PATHS = ("HOME", "CONFIG_PATH", "UPLOAD_DIR", "JOB_DIR", "MODEL_DIR", "BIN_DIR")


def setUpModule():
    for attr in _PATHS:
        _SAVED[attr] = getattr(config, attr)
    home = Path(tempfile.mkdtemp(prefix="transcribe-lm-"))
    config.HOME = home
    config.CONFIG_PATH = home / "config.json"
    config.UPLOAD_DIR = home / "uploads"
    config.JOB_DIR = home / "jobs"
    config.MODEL_DIR = home / "models"
    config.BIN_DIR = home / "bin"
    config.ensure_dirs()
    config.load(force=True)


def tearDownModule():
    for attr, value in _SAVED.items():
        setattr(config, attr, value)
    config.load(force=True)


def make_video(path, seconds=6.0):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-c:a", "aac", "-b:a", "160k", "-shortest",
         "-y", str(path)], check=True, capture_output=True)
    return path


def make_wav(path, seconds=2.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 12.0)))
                               for i in range(int(16000 * seconds))))
    return path


class BrowseSafetyTest(unittest.TestCase):
    """The browser is a filesystem exposed over HTTP; it must stay fenced."""

    @classmethod
    def setUpClass(cls):
        cls.media = Path(tempfile.mkdtemp(prefix="media-root-"))
        cls.outside = Path(tempfile.mkdtemp(prefix="outside-"))
        (cls.outside / "secrets.txt").write_text("private")
        (cls.media / "sub").mkdir()
        make_wav(cls.media / "clip.wav")
        (cls.media / "notes.txt").write_text("not media")
        (cls.media / ".hidden.wav").write_bytes(b"x")
        config.save({"media_roots": [str(cls.media)]})

    @classmethod
    def tearDownClass(cls):
        config.save({"media_roots": []})
        shutil.rmtree(cls.media, ignore_errors=True)
        shutil.rmtree(cls.outside, ignore_errors=True)

    def test_root_is_listed(self):
        paths = [r["path"] for r in files.roots()]
        self.assertIn(str(self.media.resolve()), paths)

    def test_lists_only_media_and_folders(self):
        names = [e["name"] for e in files.listing(str(self.media))["entries"]]
        self.assertIn("clip.wav", names)
        self.assertIn("sub", names)
        self.assertNotIn("notes.txt", names, "non-media must not be listed")
        self.assertNotIn(".hidden.wav", names, "dotfiles must not be listed")

    def test_traversal_is_refused(self):
        for attempt in (str(self.media / ".." / ".."), "/etc", str(self.outside), "/"):
            with self.assertRaises(PermissionError, msg=attempt):
                files.listing(attempt)

    def test_symlink_out_of_the_root_is_refused(self):
        link = self.media / "escape"
        try:
            link.symlink_to(self.outside)
        except OSError:
            self.skipTest("symlinks unavailable")
        try:
            # Resolved before checking, so the link cannot smuggle us out.
            with self.assertRaises(PermissionError):
                files.listing(str(link))
        finally:
            link.unlink()

    def test_resolve_media_rejects_non_media(self):
        with self.assertRaises(ValueError):
            files.resolve_media(str(self.media / "notes.txt"))

    def test_resolve_media_rejects_outside(self):
        with self.assertRaises(PermissionError):
            files.resolve_media(str(self.outside / "secrets.txt"))

    def test_resolve_media_accepts_a_real_file(self):
        self.assertEqual(files.resolve_media(str(self.media / "clip.wav")).name, "clip.wav")


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class ExtractPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="extract-plan-"))
        cls.video = make_video(cls.dir / "clip.mp4")

    def test_copy_picks_a_container_the_codec_fits(self):
        ext, args, info = audio.plan_extract(self.video, "copy")
        self.assertEqual(ext, ".m4a", "AAC belongs in an MP4 container")
        self.assertIn("copy", args)
        self.assertEqual(info["codec"], "aac")

    def test_copy_falls_back_when_the_codec_has_no_container(self):
        # A codec with no known container must re-encode rather than produce a
        # file that will not play.
        original = audio._COPY_CONTAINER.copy()
        audio._COPY_CONTAINER.clear()
        try:
            ext, args, _ = audio.plan_extract(self.video, "copy")
            self.assertEqual(ext, ".opus")
            self.assertNotIn("copy", args)
        finally:
            audio._COPY_CONTAINER.update(original)

    def test_every_mode_produces_playable_audio(self):
        for mode, ext in (("copy", ".m4a"), ("opus", ".opus"),
                          ("mp3", ".mp3"), ("wav", ".wav")):
            out = audio.extract_audio(self.video, self.dir / f"o_{mode}{ext}", mode,
                                      duration=6.0)
            info = audio.probe(out)
            self.assertAlmostEqual(info["duration"], 6.0, delta=1.0, msg=mode)
            self.assertGreater(out.stat().st_size, 1000, mode)

    def test_copy_is_smaller_than_the_video_and_keeps_the_codec(self):
        out = audio.extract_audio(self.video, self.dir / "copy.m4a", "copy")
        self.assertLess(out.stat().st_size, self.video.stat().st_size)
        self.assertEqual(audio.probe(out)["codec"], "aac", "copy must not re-encode")

    def test_progress_is_reported(self):
        seen = []
        audio.extract_audio(self.video, self.dir / "p.opus", "opus",
                            duration=6.0, on_progress=seen.append)
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 1.0)

    def test_video_with_no_audio_track_is_a_clear_error(self):
        silent = self.dir / "silent.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2",
                        "-c:v", "libx264", "-y", str(silent)],
                       check=True, capture_output=True)
        with self.assertRaises(audio.AudioError):
            audio.extract_audio(silent, self.dir / "none.opus", "opus")


class SizeEstimateTest(unittest.TestCase):
    """Sizing the output from the audio track, not the whole file.

    Regression: a 26 GB / 2:13:09 phone recording was refused with "about
    28111 MB needed, 1775 MB free". The estimate had used the *container*
    bitrate — video included — as though it were the audio rate, overshooting
    by more than a hundredfold and blocking a job that needed ~200 MB.
    """

    # The real recording that failed.
    DURATION = 2 * 3600 + 13 * 60 + 9
    SIZE = 26 * 1000 ** 3

    def real_info(self, audio_bitrate=192_000):
        return {
            "duration": float(self.DURATION),
            "bitrate": int(self.SIZE * 8 / self.DURATION),   # ~26 Mbps
            "audio_bitrate": audio_bitrate,
            "codec": "aac", "sample_rate": 48000, "channels": 2,
        }

    def test_the_recording_that_failed_now_fits(self):
        needed = audio.estimate_extract_bytes(self.real_info(), "copy", self.DURATION)
        self.assertLess(needed, 400e6, "a 192 kbps track for 2h13m is about 190 MB")
        self.assertGreater(needed, 100e6)
        self.assertLess(needed, 1775e6, "must fit the free space it was refused for")

    def test_container_bitrate_is_not_used(self):
        info = self.real_info()
        container_based = info["bitrate"] / 8 * self.DURATION
        needed = audio.estimate_extract_bytes(info, "copy", self.DURATION)
        self.assertLess(needed, container_based / 50,
                        "using the container rate is the bug this guards")

    def test_each_mode_is_sized_independently(self):
        info = self.real_info()
        sizes = {m: audio.estimate_extract_bytes(info, m, self.DURATION)
                 for m in ("copy", "opus", "mp3", "wav")}
        self.assertLess(sizes["opus"], sizes["mp3"])
        self.assertLess(sizes["mp3"], sizes["wav"])
        self.assertLess(sizes["opus"], 100e6, "Opus for 2h13m is well under 100 MB")

    def test_missing_stream_rate_falls_back_sanely(self):
        needed = audio.estimate_extract_bytes(self.real_info(0), "copy", self.DURATION)
        self.assertLess(needed, 400e6, "the fallback must not reach for the container")
        self.assertGreater(needed, 100e6, "and must not guess so low we run out")

    def test_a_bogus_stream_rate_cannot_exceed_the_container(self):
        info = self.real_info(audio_bitrate=99 * 10 ** 9)
        needed = audio.estimate_extract_bytes(info, "copy", self.DURATION)
        self.assertLessEqual(needed, self.SIZE * 1.01)

    def test_lossless_source_is_sized_from_its_own_rate(self):
        info = {"codec": "pcm_s16le", "audio_bitrate": 0,
                "sample_rate": 48000, "channels": 2, "bitrate": 1536000}
        needed = audio.estimate_extract_bytes(info, "copy", 60)
        self.assertAlmostEqual(needed, 48000 * 2 * 2 * 60, delta=60000)

    def test_zero_duration_does_not_divide_by_anything(self):
        self.assertGreater(audio.estimate_extract_bytes(self.real_info(), "copy", 0), 0)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class ProbeBitrateTest(unittest.TestCase):
    def test_probe_separates_audio_from_container(self):
        d = Path(tempfile.mkdtemp(prefix="bitrate-"))
        vid = d / "lopsided.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=6",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
             "-c:v", "libx264", "-b:v", "8M", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-y", str(vid)], check=True, capture_output=True)
        info = audio.probe(vid)
        self.assertGreater(info["audio_bitrate"], 0, "the audio rate must be reported")
        self.assertLess(info["audio_bitrate"], info["bitrate"],
                        "audio alone must be smaller than the whole container")

        est = audio.estimate_extract_bytes(info, "copy", info["duration"])
        out = audio.extract_audio(vid, d / "out.m4a", "copy")
        actual = out.stat().st_size
        self.assertLess(abs(est - actual), max(actual, 50_000),
                        f"estimate {est} should be close to actual {actual}")


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class LocalJobTest(unittest.TestCase):
    """End to end over HTTP, including the file the user must not lose."""

    @classmethod
    def setUpClass(cls):
        jobs_mod.store().set_runner(runner.run_job)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

        cls.media = Path(tempfile.mkdtemp(prefix="user-media-"))
        cls.out = Path(tempfile.mkdtemp(prefix="user-out-"))
        cls.video = make_video(cls.media / "long meeting.mp4")
        config.save({"media_roots": [str(cls.media)], "extract_dir": str(cls.out)})

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        config.save({"media_roots": [], "extract_dir": ""})

    def req(self, path, method="GET", body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                   data=data, method=method, headers=headers)
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}

    def wait(self, job_id, status="done", timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            job = self.req(f"/api/jobs/{job_id}")["job"]
            if job["status"] == status:
                return job
            if job["status"] in ("failed", "cancelled") and status == "done":
                self.fail(f"job failed: {job['error']}")
            time.sleep(0.15)
        self.fail(f"never reached {status}")

    def test_browse_over_http(self):
        data = self.req("/api/browse?path=" + urllib.parse.quote(str(self.media)))
        names = [e["name"] for e in data["entries"]]
        self.assertIn("long meeting.mp4", names)
        entry = next(e for e in data["entries"] if e["name"] == "long meeting.mp4")
        self.assertTrue(entry["video"])
        self.assertGreater(entry["size"], 0)

    def test_browse_refuses_outside_paths(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req("/api/browse?path=/etc")
        self.assertEqual(cm.exception.code, 403)

    def test_extract_then_the_source_video_survives(self):
        """The whole point: extraction must not touch the user's video."""
        before = self.video.stat()
        created = self.req("/api/local", "POST",
                           {"path": str(self.video), "kind": "extract",
                            "extract_mode": "copy"})
        job = self.wait(created["job"]["id"])

        out = Path(job["output_file"])
        self.assertTrue(out.exists())
        self.assertEqual(out.parent, self.out, "must land in the configured folder")
        self.assertEqual(out.suffix, ".m4a")
        self.assertLess(out.stat().st_size, before.st_size)

        self.assertTrue(self.video.exists(), "the source video must still be there")
        self.assertEqual(self.video.stat().st_size, before.st_size)

    def test_deleting_the_job_never_deletes_the_users_video(self):
        created = self.req("/api/local", "POST",
                           {"path": str(self.video), "kind": "extract"})
        job_id = created["job"]["id"]
        self.wait(job_id)
        self.req(f"/api/jobs/{job_id}", "DELETE")
        self.assertTrue(self.video.exists(),
                        "deleting a job must never remove a file the user owns")

    def test_extraction_does_not_copy_the_video(self):
        """No second copy of the source may appear anywhere we manage."""
        created = self.req("/api/local", "POST",
                           {"path": str(self.video), "kind": "extract"})
        self.wait(created["job"]["id"])
        for d in (config.UPLOAD_DIR, config.HOME):
            for f in d.rglob("*"):
                if f.is_file():
                    self.assertLess(f.stat().st_size, self.video.stat().st_size,
                                    f"{f} looks like a copy of the source video")

    def test_transcribe_a_local_file_without_uploading(self):
        created = self.req("/api/local", "POST",
                           {"path": str(self.video), "kind": "transcribe",
                            "engine": "mock"})
        job = self.wait(created["job"]["id"])
        self.assertEqual(job["kind"], "transcribe")
        result = self.req(f"/api/jobs/{job['id']}/result")
        self.assertTrue(result["segments"])
        self.assertTrue(self.video.exists(), "transcribing must not consume the source")

    def test_output_downloads(self):
        created = self.req("/api/local", "POST",
                           {"path": str(self.video), "kind": "extract"})
        job = self.wait(created["job"]["id"])
        r = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/jobs/{job['id']}/output")
        with urllib.request.urlopen(r, timeout=60) as resp:
            body = resp.read()
            self.assertIn("attachment", resp.headers["Content-Disposition"])
        self.assertEqual(len(body), Path(job["output_file"]).stat().st_size)

    def test_repeat_extraction_does_not_overwrite(self):
        a = self.wait(self.req("/api/local", "POST",
                               {"path": str(self.video), "kind": "extract"})["job"]["id"])
        b = self.wait(self.req("/api/local", "POST",
                               {"path": str(self.video), "kind": "extract"})["job"]["id"])
        self.assertNotEqual(a["output_file"], b["output_file"])
        self.assertTrue(Path(a["output_file"]).exists())

    def test_a_failed_extraction_leaves_no_partial_file(self):
        """Half a file is worse than none when storage is nearly full."""
        silent = self.media / "no-audio.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2",
             "-c:v", "libx264", "-y", str(silent)], check=True, capture_output=True)
        before = set(self.out.iterdir())
        created = self.req("/api/local", "POST",
                           {"path": str(silent), "kind": "extract"})
        self.wait(created["job"]["id"], "failed", timeout=60)
        self.assertEqual(set(self.out.iterdir()), before,
                         "a failed extraction must not leave a file behind")
        silent.unlink()

    # ---------------- compressing video ----------------

    def test_compress_shrinks_the_video_and_leaves_the_original_alone(self):
        """A compress job must be additive: a new file, the old one untouched."""
        source = make_video(self.media / "to shrink.mp4", seconds=6.0)
        # Give it something worth compressing, at a size worth reducing.
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=6",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
             "-c:v", "libx264", "-b:v", "10M", "-preset", "ultrafast",
             "-c:a", "aac", "-shortest", "-y", str(source)],
            check=True, capture_output=True)
        before = source.stat().st_size

        created = self.req("/api/local", "POST",
                           {"path": str(source), "kind": "compress",
                            "compress_quality": "small"})
        self.assertEqual(created["job"]["kind"], "compress")
        job = self.wait(created["job"]["id"], timeout=300)

        out = Path(job["output_file"])
        self.assertTrue(out.exists())
        self.assertEqual(out.parent, self.out)
        self.assertEqual(out.suffix, ".mp4")
        self.assertLess(out.stat().st_size, before)
        self.assertEqual(source.stat().st_size, before,
                         "the original video must be untouched")
        self.assertEqual(job["source_size"], before,
                         "the original size has to survive for the UI to show a saving")

    def test_compressing_an_audio_file_is_refused_before_anything_runs(self):
        audio_file = make_wav(self.media / "just audio.wav", seconds=1.0)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req("/api/local", "POST",
                     {"path": str(audio_file), "kind": "compress"})
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("no video", cm.exception.read().decode().lower())

    def test_a_made_up_quality_never_reaches_ffmpeg(self):
        created = self.req("/api/local", "POST",
                           {"path": str(self.video), "kind": "compress",
                            "compress_quality": "; rm -rf /",
                            "compress_codec": "made-up"})
        options = created["job"]["options"]
        self.assertNotIn("compress_quality", options)
        self.assertNotIn("compress_codec", options)
        self.req(f"/api/jobs/{created['job']['id']}/cancel", "POST")

    def test_config_reports_what_this_device_can_do(self):
        media = self.req("/api/config")["media"]
        self.assertTrue(media["ffmpeg"])
        self.assertTrue(media["output_dir"])
        self.assertIsInstance(media["hardware_encoders"], list)

    def test_rejects_a_path_outside_the_media_roots(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req("/api/local", "POST", {"path": "/etc/hosts", "kind": "extract"})
        self.assertIn(cm.exception.code, (403, 404))


if __name__ == "__main__":
    unittest.main(verbosity=2)
