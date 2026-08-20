"""End-to-end test: real HTTP server, real job worker, mock engine."""

import json
import math
import os
import struct
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
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-test-"))

from http.server import ThreadingHTTPServer                       # noqa: E402
from transcribe import config, jobs as jobs_mod, runner, server    # noqa: E402
from transcribe.engines import registry                           # noqa: E402


def make_wav(path: Path, seconds: float = 2.0, rate: int = 16000):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        n = int(rate * seconds)
        f.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(i / 12.0))) for i in range(n)
        ))


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.ensure_dirs()
        jobs_mod.store().set_runner(runner.run_job)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.wav = cls.tmp / "meeting notes.wav"
        make_wav(cls.wav)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    # ---------- helpers ----------

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def req(self, path, method="GET", data=None, headers=None, raw=False):
        r = urllib.request.Request(self.url(path), data=data, method=method,
                                   headers=headers or {})
        with urllib.request.urlopen(r, timeout=30) as resp:
            body = resp.read()
            if raw:
                return resp.status, resp.headers, body
            return json.loads(body.decode()) if body else {}

    def upload(self, path=None, **params):
        path = path or self.wav
        qs = urllib.parse.urlencode({"name": "meeting notes.wav", "engine": "mock", **params})
        return self.req(f"/api/upload?{qs}", "POST", data=path.read_bytes(),
                        headers={"Content-Type": "audio/wav"})

    def wait_for(self, job_id, status="done", timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            data = self.req(f"/api/jobs/{job_id}")
            if data["job"]["status"] == status:
                return data["job"]
            if data["job"]["status"] in ("failed", "cancelled") and status == "done":
                self.fail(f"job failed: {data['job']['error']}")
            time.sleep(0.1)
        self.fail(f"job did not reach {status} in {timeout}s")

    # ---------- tests ----------

    def test_01_index_and_static(self):
        status, headers, body = self.req("/", raw=True)
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Transcribe</title>", body)
        status, headers, body = self.req("/static/app.js", raw=True)
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])

    def test_02_path_traversal_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req("/static/../transcribe/config.py")
        self.assertEqual(cm.exception.code, 404)

    def test_03_config_roundtrip(self):
        data = self.req("/api/config")
        self.assertIn("engines", data)
        self.assertIn("mock", [e["name"] for e in data["engines"]])
        self.assertNotIn("keys", data["config"], "raw API keys must never reach the browser")
        payload = json.dumps({"language": "es", "keys": {"deepgram": "secret-key-value"}}).encode()
        data = self.req("/api/config", "POST", payload, {"Content-Type": "application/json"})
        self.assertEqual(data["config"]["language"], "es")
        self.assertTrue(data["config"]["has_key"]["deepgram"])
        self.assertNotIn("secret-key-value", json.dumps(data))
        self.req("/api/config", "POST", json.dumps({"language": "auto"}).encode(),
                 {"Content-Type": "application/json"})

    def test_04_full_transcription_flow(self):
        created = self.upload()
        job_id = created["job"]["id"]
        self.assertEqual(created["job"]["status"], "pending")

        job = self.wait_for(job_id)
        self.assertEqual(job["engine"], "mock")
        self.assertEqual(job["model"], "mock-1")
        self.assertGreater(job["duration"], 1.5)
        self.assertTrue(job["has_audio"])

        result = self.req(f"/api/jobs/{job_id}/result")
        self.assertGreater(len(result["segments"]), 1)
        speakers = {s["speaker"] for s in result["segments"]}
        self.assertEqual(speakers, {"A", "B"}, "both mock speakers should survive the pipeline")
        self.assertEqual(len(result["stats"]), 2)
        self.assertAlmostEqual(sum(s["share"] for s in result["stats"]), 1.0, places=3)

        first = result["segments"][0]
        self.assertIn("Hello everyone", first["text"])
        self.assertTrue(first["words"], "word timings must survive to the client")
        self.assertLessEqual(first["words"][0]["start"], first["words"][-1]["start"])
        ServerTest.job_id = job_id

    def test_05_exports(self):
        job_id = ServerTest.job_id
        expected = {
            "txt": "Speaker A", "md": "**Speaker A**", "srt": "-->",
            "vtt": "WEBVTT", "csv": "start,end", "json": '"segments"',
        }
        for fmt, needle in expected.items():
            status, headers, body = self.req(f"/api/jobs/{job_id}/export.{fmt}", raw=True)
            self.assertEqual(status, 200, fmt)
            text = body.decode()
            self.assertIn(needle, text, fmt)
            self.assertIn("attachment", headers["Content-Disposition"], fmt)
            self.assertIn("meeting notes", headers["Content-Disposition"], fmt)

        # srt timing format, precisely
        _, _, srt = self.req(f"/api/jobs/{job_id}/export.srt", raw=True)
        lines = srt.decode().strip().split("\n")
        self.assertEqual(lines[0], "1")
        self.assertRegex(lines[1], r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")

        _, headers, _ = self.req(f"/api/jobs/{job_id}/export.txt?inline=1", raw=True)
        self.assertNotIn("Content-Disposition", headers)

        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req(f"/api/jobs/{job_id}/export.docx")
        self.assertEqual(cm.exception.code, 404)

    def test_06_audio_range_requests(self):
        job_id = ServerTest.job_id
        status, headers, body = self.req(f"/api/jobs/{job_id}/audio", raw=True)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        total = len(body)

        status, headers, part = self.req(f"/api/jobs/{job_id}/audio", raw=True,
                                         headers={"Range": "bytes=0-99"})
        self.assertEqual(status, 206)
        self.assertEqual(len(part), 100)
        self.assertEqual(headers["Content-Range"], f"bytes 0-99/{total}")
        self.assertEqual(part, body[:100])

        status, headers, part = self.req(f"/api/jobs/{job_id}/audio", raw=True,
                                         headers={"Range": "bytes=100-"})
        self.assertEqual(status, 206)
        self.assertEqual(part, body[100:])

        # suffix range (last 50 bytes)
        status, headers, part = self.req(f"/api/jobs/{job_id}/audio", raw=True,
                                         headers={"Range": "bytes=-50"})
        self.assertEqual(status, 206)
        self.assertEqual(part, body[-50:])

        try:
            self.req(f"/api/jobs/{job_id}/audio", raw=True,
                     headers={"Range": f"bytes={total + 10}-"})
            self.fail("out-of-range should be rejected")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 416)

    def test_07_speaker_rename_persists(self):
        job_id = ServerTest.job_id
        payload = json.dumps({"speakers": {"A": "Justin", "B": ""}}).encode()
        self.req(f"/api/jobs/{job_id}/speakers", "POST", payload,
                 {"Content-Type": "application/json"})
        result = self.req(f"/api/jobs/{job_id}/result")
        self.assertEqual(result["speakers"], {"A": "Justin"})
        _, _, txt = self.req(f"/api/jobs/{job_id}/export.txt", raw=True)
        self.assertIn("Justin", txt.decode())
        self.assertIn("Speaker B", txt.decode(), "unnamed speakers keep their default label")

    def test_08_merged_vs_split(self):
        job_id = ServerTest.job_id
        merged = self.req(f"/api/jobs/{job_id}/result?merged=1")
        split = self.req(f"/api/jobs/{job_id}/result?merged=0")
        self.assertLessEqual(len(merged["segments"]), len(split["segments"]))

    def test_09_failure_is_reported(self):
        created = self.upload(name="broken.wav", fail="1")
        job_id = created["job"]["id"]
        job = self.wait_for(job_id, "failed")
        self.assertIn("deliberate test failure", job["error"])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req(f"/api/jobs/{job_id}/result")
        self.assertEqual(cm.exception.code, 409)

    def test_10_retry_after_failure(self):
        created = self.upload(name="retry.wav", fail="1")
        job_id = created["job"]["id"]
        self.wait_for(job_id, "failed")
        # Clear the failure flag by retrying a job whose options no longer fail.
        jobs_mod.store().update(job_id, options={})
        self.req(f"/api/jobs/{job_id}/retry", "POST", b"")
        job = self.wait_for(job_id, "done")
        self.assertEqual(job["error"], "")

    def test_11_rename_and_delete(self):
        created = self.upload(name="temp.wav")
        job_id = created["job"]["id"]
        self.wait_for(job_id)
        data = self.req(f"/api/jobs/{job_id}", "PATCH",
                        json.dumps({"name": "Renamed"}).encode(),
                        {"Content-Type": "application/json"})
        self.assertEqual(data["job"]["name"], "Renamed")

        audio_path = Path(jobs_mod.store().get(job_id).audio_file)
        self.assertTrue(audio_path.exists())
        self.req(f"/api/jobs/{job_id}", "DELETE")
        self.assertFalse(audio_path.exists(), "deleting a job must delete its audio")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req(f"/api/jobs/{job_id}")
        self.assertEqual(cm.exception.code, 404)

    def test_12_unknown_engine_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.upload(engine="nope")
        self.assertEqual(cm.exception.code, 400)

    def test_13_empty_upload_rejected(self):
        empty = self.tmp / "empty.wav"
        empty.write_bytes(b"")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.upload(path=empty)
        self.assertEqual(cm.exception.code, 400)

    def test_14_sse_stream_delivers_updates(self):
        req = urllib.request.Request(self.url("/api/events"))
        resp = urllib.request.urlopen(req, timeout=20)
        self.assertIn("text/event-stream", resp.headers["Content-Type"])

        got = {"id": None}

        def read_stream():
            for _ in range(400):
                line = resp.readline()
                if not line:
                    return
                if line.startswith(b"data: "):
                    ev = json.loads(line[6:].decode())
                    if ev.get("type") == "job" and ev["job"]["name"] == "sse-test.wav":
                        got["id"] = ev["job"]["id"]
                        return

        t = threading.Thread(target=read_stream, daemon=True)
        t.start()
        time.sleep(0.3)
        created = self.upload(name="sse-test.wav")
        t.join(timeout=15)
        resp.close()
        self.assertEqual(got["id"], created["job"]["id"], "SSE should push the new job")

    def test_15_cancel_midflight(self):
        created = self.upload(name="slow.wav", steps="200", delay="0.05")
        job_id = created["job"]["id"]
        end = time.time() + 10
        while time.time() < end:
            if self.req(f"/api/jobs/{job_id}")["job"]["status"] == "running":
                break
            time.sleep(0.05)
        self.req(f"/api/jobs/{job_id}/cancel", "POST", b"")
        job = self.wait_for(job_id, "cancelled", timeout=20)
        self.assertEqual(job["status"], "cancelled")

    def test_16_interrupted_jobs_recover_on_restart(self):
        store = jobs_mod.store()
        job = store.create(name="ghost.wav", engine="mock")
        store.update(job.id, status="running", stage="transcribing")
        fresh = jobs_mod.JobStore()
        recovered = fresh.get(job.id)
        self.assertEqual(recovered.status, "failed")
        self.assertIn("Interrupted", recovered.error)
        store.delete(job.id)

    def test_17_engine_config_fields(self):
        data = self.req("/api/config")
        engines = {e["name"]: e for e in data["engines"]}

        rp = engines["runpod"]
        keys = {f["key"] for f in rp["config_fields"]}
        self.assertEqual(keys, {"runpod_endpoint", "runpod_model", "runpod_audio_url"})
        self.assertFalse(rp["available"], "runpod is unavailable until an endpoint is set")

        payload = json.dumps({"runpod_endpoint": "abc123xyz",
                              "runpod_model": "large-v3"}).encode()
        data = self.req("/api/config", "POST", payload, {"Content-Type": "application/json"})
        engines = {e["name"]: e for e in data["engines"]}
        self.assertTrue(engines["runpod"]["available"],
                        "setting an endpoint id should make runpod available")
        field = next(f for f in engines["runpod"]["config_fields"]
                     if f["key"] == "runpod_endpoint")
        self.assertEqual(field["value"], "abc123xyz")

        # And it survives a reload from disk.
        self.assertEqual(config.load(force=True)["runpod_endpoint"], "abc123xyz")
        self.req("/api/config", "POST", json.dumps({"runpod_endpoint": ""}).encode(),
                 {"Content-Type": "application/json"})

    def test_18_numeric_config_coercion(self):
        payload = json.dumps({"local_threads": "", "max_dur": "45"}).encode()
        data = self.req("/api/config", "POST", payload, {"Content-Type": "application/json"})
        self.assertEqual(data["config"]["local_threads"], 0, "blank number falls back to default")
        self.assertEqual(data["config"]["max_dur"], 45.0)
        self.req("/api/config", "POST", json.dumps({"max_dur": 30}).encode(),
                 {"Content-Type": "application/json"})

    def test_19_health(self):
        data = self.req("/api/health")
        self.assertTrue(data["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
