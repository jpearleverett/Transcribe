"""HTTP server: zero third-party dependencies, on purpose.

Every pip package is a chance for a phone to fail to compile something. The
whole server is stdlib, so `pkg install python` is the entire runtime
requirement for the cloud engines.
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import queue
import shutil
import re
import secrets
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, files as files_mod, jobs as jobs_mod, runner
from .align import Segment, Word, merge_adjacent, speaker_stats
from . import audio as audio_mod, video as video_mod
from .engines import base as engines
from .exporters import export

VERSION = "1.0.0"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CHUNK = 1 << 20  # 1 MiB: big enough to be fast, small enough for a phone's RAM

# Set when the server binds to a non-loopback address, so a recording is not
# readable by anything else on a coffee-shop wifi.
AUTH_TOKEN = ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"Transcribe/{VERSION}"

    # -------------------- plumbing --------------------

    def log_message(self, fmt, *args):
        if os.environ.get("TRANSCRIBE_DEBUG"):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes = b"", ctype: str = "application/json",
              extra: dict = None, head_only: bool = False):
        # A HEAD response must carry the headers but no body. Every JSON route
        # is reachable by HEAD, and sending a body there leaves bytes in a
        # keep-alive socket that the next request parses as its request line.
        head_only = head_only or self.command == "HEAD"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page only ever talks to its own origin; deny the rest.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.close_connection:
            # Setting close_connection alone drops the socket without telling
            # the client why; say so explicitly so the browser reconnects
            # cleanly instead of reporting a network error.
            self.send_header("Connection", "close")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body and not head_only:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def json(self, obj, status: int = 200, extra: dict = None):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", extra)

    def fail(self, message: str, status: int = 400):
        self.json({"error": message}, status)

    def fail_unread(self, message: str, status: int = 400):
        """Reject a request whose body we have not consumed.

        With HTTP/1.1 keep-alive, leaving unread bytes in the socket makes the
        next request parse the tail of this one's body as a request line — the
        connection silently corrupts rather than failing cleanly. Closing it is
        the only correct move.
        """
        self.close_connection = True
        self.json({"error": message}, status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 8 * 1024 * 1024:
            self.close_connection = True
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("invalid JSON body")

    def query(self) -> dict:
        q = urllib.parse.urlparse(self.path).query
        return {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}

    def same_origin(self) -> bool:
        """Reject cross-site state-changing requests.

        The server listens on loopback, but any web page the user opens in the
        same browser can still reach it: a cross-origin `fetch` to
        http://127.0.0.1:8756 has its *response* hidden by CORS, yet the request
        is still delivered and the side effect still happens. So a page could
        silently DELETE transcripts or rewrite settings.

        Chrome sends `Origin` on cross-origin requests and on same-origin
        non-GET fetches, so requiring it to match the Host we were addressed by
        blocks that while leaving curl (which sends no Origin) working.
        """
        origin = self.headers.get("Origin")
        if not origin:
            # No Origin: not a browser fetch. Sec-Fetch-Site still catches the
            # form-submission case, where Chrome omits Origin for some methods.
            return self.headers.get("Sec-Fetch-Site", "same-origin") in (
                "same-origin", "same-site", "none")
        host = self.headers.get("Host", "")
        try:
            parsed = urllib.parse.urlparse(origin)
        except ValueError:
            return False
        return bool(host) and parsed.netloc == host

    def authorized(self) -> bool:
        if not AUTH_TOKEN:
            return True
        supplied = (
            self.headers.get("X-Auth-Token")
            or self.query().get("token")
            or _cookie(self.headers.get("Cookie", ""), "transcribe_token")
        )
        if not supplied:
            return False
        # compare_digest raises TypeError on non-ASCII str arguments, and this
        # runs before the request try/except — so a token with an accent in it
        # would drop the connection with no response at all. Compare bytes.
        return secrets.compare_digest(supplied.encode("utf-8", "replace"),
                                      AUTH_TOKEN.encode("utf-8"))

    # -------------------- routing --------------------

    def do_GET(self):
        self._route("GET")

    def do_HEAD(self):
        self._route("HEAD")

    def do_POST(self):
        self._route("POST")

    def do_PATCH(self):
        self._route("PATCH")

    def do_DELETE(self):
        self._route("DELETE")

    def _route(self, method: str):
        path = urllib.parse.urlparse(self.path).path

        if not self.authorized():
            return self.fail_unread(
                "Not authorised. Open the link printed by the server, including its ?token=.", 401)

        if method in ("POST", "PATCH", "DELETE") and not self.same_origin():
            return self.fail_unread(
                "Blocked a cross-site request. Open the app directly rather than "
                "letting another page talk to it.", 403)

        try:
            if path == "/" or path == "/index.html":
                return self.serve_static("index.html", head_only=(method == "HEAD"))
            if path.startswith("/static/"):
                return self.serve_static(path[len("/static/"):], head_only=(method == "HEAD"))
            if path == "/api/events" and method == "GET":
                return self.sse()
            if path == "/api/config":
                if method == "GET":
                    return self.json(self._config_payload())
                if method == "POST":
                    body = self.read_json()
                    config.save(body)
                    return self.json(self._config_payload())
            if path == "/api/jobs" and method == "GET":
                return self.json({"jobs": [j.public() for j in jobs_mod.store().list()]})
            if path == "/api/upload" and method == "POST":
                return self.upload()
            if path == "/api/browse" and method == "GET":
                return self.browse()
            if path == "/api/local" and method == "POST":
                return self.local_job()
            if path == "/api/health":
                return self.json({"ok": True, "version": VERSION,
                                  "queue": jobs_mod.store().queue_depth()})

            m = re.match(r"^/api/jobs/([A-Za-z0-9]+)(/[a-z.]*)?$", path)
            if m:
                return self.job_route(method, m.group(1), (m.group(2) or "").strip("/"))

            return self.fail("Not found", 404)

        except ValueError as e:
            return self.fail(str(e), 400)
        except engines.EngineError as e:
            return self.fail(str(e), 400)
        except (BrokenPipeError, ConnectionResetError):
            return                     # the phone navigated away mid-response
        except Exception as e:         # noqa: BLE001 - never take the server down
            import traceback
            traceback.print_exc()
            return self.fail(f"Server error: {e}", 500)

    def _config_payload(self) -> dict:
        cfg = config.redacted()
        cfg["server_info"] = f"Transcribe {VERSION} · Python {'.'.join(map(str, __import__('sys').version_info[:3]))}"
        return {"config": cfg, "engines": engines.describe_all(),
                "media": self._media_payload()}

    @staticmethod
    def _media_payload() -> dict:
        """What the device can do with local media, so the UI can say so.

        Whether a hardware video encoder exists is worth surfacing before
        somebody starts a job: it is the difference between half an hour and
        half a day. It is still only a *candidate* here — nothing is promised
        until a trial encode proves it works on this phone.
        """
        payload = {"output_dir": "", "free_bytes": 0, "ffmpeg": audio_mod.have_ffmpeg(),
                   "hardware_encoders": []}
        try:
            out = files_mod.output_dir()
            payload["output_dir"] = str(out)
            payload["free_bytes"] = files_mod.free_bytes(out)
        except OSError:
            pass
        if audio_mod.have_ffmpeg():
            payload["hardware_encoders"] = sorted(
                n for n in video_mod.encoders() if n.endswith("_mediacodec"))
        return payload

    # -------------------- static --------------------

    def serve_static(self, rel: str, head_only: bool = False):
        rel = posixpath.normpath("/" + rel).lstrip("/")
        target = (WEB_DIR / rel).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            return self.fail("Not found", 404)     # traversal attempt
        if not target.is_file():
            return self.fail("Not found", 404)

        ctype, _ = mimetypes.guess_type(str(target))
        if target.suffix == ".webmanifest":
            ctype = "application/manifest+json"
        body = target.read_bytes()
        # No caching: the app updates when the user pulls a new version, and a
        # stale cached app.js on a phone is miserable to debug.
        self._send(200, body, ctype or "application/octet-stream",
                   {"Cache-Control": "no-cache"}, head_only=head_only)

    # -------------------- upload --------------------

    def upload(self):
        q = self.query()
        name = (q.get("name") or "recording").strip()[:200]
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self.fail("No audio was sent.")
        max_bytes = config.load()["max_upload_mb"] * 1024 * 1024
        if length > max_bytes:
            return self.fail_unread(
                f"That file is larger than the {config.load()['max_upload_mb']} MB limit.")

        engine_name = q.get("engine") or config.load()["engine"]
        try:
            engine = engines.get(engine_name)
        except engines.EngineError as e:
            return self.fail_unread(str(e))
        ok, reason = engine.available()
        if not ok:
            return self.fail_unread(reason or f"{engine.label} is not available.")
        if engine.needs_key and not engine.has_key():
            return self.fail_unread(f"{engine.label} needs an API key. Add one in Settings.")

        config.ensure_dirs()
        safe = _safe_filename(name)
        dest = config.UPLOAD_DIR / f"{int(time.time())}-{secrets.token_hex(4)}-{safe}"

        # Stream straight to disk. Reading Content-Length bytes into memory
        # would mean a 500 MB recording tries to allocate 500 MB on a phone.
        written = 0
        try:
            with open(dest, "wb") as fh:
                while written < length:
                    chunk = self.rfile.read(min(CHUNK, length - written))
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            dest.unlink(missing_ok=True)
            if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
                return self.fail_unread("The phone is out of storage space.", 507)
            return self.fail_unread("The upload was interrupted.", 400)

        if written < length:
            dest.unlink(missing_ok=True)
            return self.fail_unread("The upload ended early — try again.", 400)

        store = jobs_mod.store()
        job = store.create(
            name=name,
            engine=engine_name,
            language=q.get("language") or config.load()["language"],
            size=written,
            audio_file=str(dest),
            media_type=self.headers.get("Content-Type", ""),
            options=_job_options(q),
        )
        store.enqueue(job.id)
        return self.json({"job": job.public()}, 201)

    # -------------------- local media --------------------

    def browse(self):
        try:
            return self.json(files_mod.listing(self.query().get("path", "")))
        except PermissionError as e:
            return self.fail(str(e), 403)
        except (NotADirectoryError, FileNotFoundError) as e:
            return self.fail(str(e), 404)
        except OSError as e:
            return self.fail(f"Could not read that folder: {e}", 400)

    def local_job(self):
        """Start work on a file already on the device — no upload, no copy.

        This is the only workable path for large video: uploading a 29 GB file
        through the browser would make the phone hold a second copy of it to
        achieve nothing, since ffmpeg can read the original in place.
        """
        body = self.read_json()
        try:
            source = files_mod.resolve_media(str(body.get("path") or ""))
        except PermissionError as e:
            return self.fail(str(e), 403)
        except (FileNotFoundError, ValueError) as e:
            return self.fail(str(e), 404)

        kind = str(body.get("kind") or "transcribe")
        if kind not in ("extract", "compress", "transcribe"):
            kind = "transcribe"
        store = jobs_mod.store()
        options = {
            "num_speakers": _int(body.get("num_speakers"), 0),
            "extract_mode": str(body.get("extract_mode") or "") or None,
            "compress_quality": _choice(body.get("compress_quality"),
                                        video_mod.QUALITY_PRESETS),
            "compress_speed": _choice(body.get("compress_speed"),
                                      video_mod.SPEED_PRESETS),
            "compress_codec": _choice(body.get("compress_codec"), video_mod.CODECS),
        }
        options = {k: v for k, v in options.items() if v not in (None, "")}

        if kind == "transcribe":
            engine_name = str(body.get("engine") or "") or config.load()["engine"]
            try:
                engine = engines.get(engine_name)
            except engines.EngineError as e:
                return self.fail(str(e))
            ok, reason = engine.available()
            if not ok:
                return self.fail(reason or f"{engine.label} is not available.")
            if engine.needs_key and not engine.has_key():
                return self.fail(f"{engine.label} needs an API key. Add one in Settings.")
        else:
            engine_name = ""
            if not audio_mod.have_ffmpeg():
                return self.fail("ffmpeg is needed for this. Run:  pkg install ffmpeg")
            if kind == "compress" and source.suffix.lower() not in files_mod.VIDEO_EXTS:
                return self.fail("That is an audio file — there is no video in it "
                                 "to compress.")

        job = store.create(
            name=source.name,
            kind=kind,
            engine=engine_name,
            language=str(body.get("language") or "") or config.load()["language"],
            size=source.stat().st_size,
            source_size=source.stat().st_size,
            audio_file=str(source),
            owns_audio=False,          # the user's file; never delete it
            media_type=mimetypes.guess_type(str(source))[0] or "",
            options=options,
        )
        store.enqueue(job.id)
        return self.json({"job": job.public()}, 201)

    # -------------------- per-job routes --------------------

    def job_route(self, method: str, job_id: str, action: str):
        store = jobs_mod.store()
        job = store.get(job_id)
        if not job:
            return self.fail("That transcript no longer exists.", 404)

        if action == "" and method == "GET":
            return self.json({"job": job.public()})

        if action == "" and method == "PATCH":
            body = self.read_json()
            name = str(body.get("name", "")).strip()[:200]
            if not name:
                return self.fail("A name is required.")
            return self.json({"job": store.update(job_id, name=name).public()})

        if action == "" and method == "DELETE":
            store.delete(job_id)
            return self.json({"ok": True})

        if action == "cancel" and method == "POST":
            return self.json({"ok": store.cancel(job_id)})

        if action == "rerun" and method == "POST":
            # Re-run the same audio through a different engine, as a NEW job so
            # the two can be compared side by side. Vendor diarization
            # benchmarks contradict each other, so the only reliable comparison
            # is on your own recording.
            if not job.audio_file or not Path(job.audio_file).exists():
                return self.fail("The original audio is gone, so this can't be re-run.")
            body = self.read_json()
            engine_name = str(body.get("engine") or "").strip()
            try:
                engine = engines.get(engine_name)
            except engines.EngineError as e:
                return self.fail(str(e))
            ok, reason = engine.available()
            if not ok:
                return self.fail(reason or f"{engine.label} is not available.")
            if engine.needs_key and not engine.has_key():
                return self.fail(f"{engine.label} needs an API key. Add one in Settings.")

            # Hard-link rather than copy. Both jobs need the file to survive the
            # other being deleted, and a link gives exactly that — the data goes
            # only when the last reference does — without a second copy of a
            # hundred-megabyte recording on a phone. Falls back to copying where
            # links are unavailable.
            src = Path(job.audio_file)
            dest = config.UPLOAD_DIR / f"{int(time.time())}-{secrets.token_hex(4)}-{src.name.split('-', 2)[-1]}"
            try:
                os.link(src, dest)
            except OSError:
                try:
                    shutil.copy2(src, dest)
                except OSError as e:
                    return self.fail(f"Could not duplicate the audio: {e}", 507)

            opts = dict(job.options or {})
            if "num_speakers" in body:
                opts["num_speakers"] = _int(body.get("num_speakers"), 0)
            new_job = store.create(
                name=f"{job.name} · {engine.label}",
                engine=engine_name,
                language=job.language,
                size=job.size,
                duration=job.duration,
                audio_file=str(dest),
                media_type=job.media_type,
                options=opts,
                speakers=dict(job.speakers or {}),
            )
            store.enqueue(new_job.id)
            return self.json({"job": new_job.public()}, 201)

        if action == "retry" and method == "POST":
            if not job.audio_file or not Path(job.audio_file).exists():
                return self.fail("The original audio is gone, so this can't be retried.")
            store.update(job_id, status=jobs_mod.PENDING, stage="queued",
                         progress=0.0, error="", log=[])
            store.enqueue(job_id)
            return self.json({"job": store.get(job_id).public()})

        if action == "speakers" and method == "POST":
            body = self.read_json()
            speakers = body.get("speakers") or {}
            if not isinstance(speakers, dict):
                return self.fail("Invalid speaker names.")
            clean = {str(k): str(v)[:80] for k, v in speakers.items() if str(v).strip()}
            return self.json({"job": store.update(job_id, speakers=clean).public()})

        if action == "result" and method == "GET":
            return self.result(job)

        if action == "audio":
            return self.audio(job, head_only=(method == "HEAD"))

        if action == "output" and method in ("GET", "HEAD"):
            if not job.output_file or not Path(job.output_file).exists():
                return self.fail("That file is no longer there.", 404)
            return self.send_file(Path(job.output_file), head_only=(method == "HEAD"))

        if action.startswith("export."):
            return self.export(job, action.split(".", 1)[1])

        return self.fail("Not found", 404)

    def result(self, job):
        data = jobs_mod.store().load_result(job.id)
        if data is None:
            if job.status == jobs_mod.DONE:
                return self.fail("The transcript file is missing. Try running it again.", 404)
            return self.fail("This transcript isn't ready yet.", 409)

        segments = _load_segments(data)
        if self.query().get("merged", "1") not in ("0", "false"):
            cfg = config.load()
            segments = merge_adjacent(segments, max_gap=cfg["max_gap"] * 2, max_dur=90.0)
        return self.json({
            "meta": data.get("meta", {}),
            "speakers": job.speakers or {},
            "stats": speaker_stats(segments),
            "segments": [s.to_dict() for s in segments],
        })

    def export(self, job, fmt: str):
        data = jobs_mod.store().load_result(job.id)
        if data is None:
            return self.fail("This transcript isn't ready yet.", 409)
        segments = _load_segments(data)
        cfg = config.load()
        if fmt in ("txt", "md", "csv", "json"):
            segments = merge_adjacent(segments, max_gap=cfg["max_gap"] * 2, max_dur=90.0)
        meta = dict(data.get("meta", {}))
        meta["name"] = job.name
        try:
            ctype, ext, body = export(fmt, segments, job.speakers or {}, meta)
        except ValueError as e:
            return self.fail(str(e), 404)

        inline = self.query().get("inline") in ("1", "true")
        headers = {"Cache-Control": "no-store"}
        if not inline:
            fname = _safe_filename(job.name).rsplit(".", 1)[0] or "transcript"
            headers["Content-Disposition"] = f'attachment; filename="{fname}.{ext}"'
        self._send(200, body.encode("utf-8"), ctype, headers)

    def send_file(self, path: Path, head_only: bool = False):
        """Hand a produced file to the browser as a download."""
        try:
            fh = open(path, "rb")
        except OSError:
            return self.fail("That file could not be read.", 404)
        try:
            size = os.fstat(fh.fileno()).st_size
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{_safe_filename(path.name)}"')
            self.end_headers()
            if head_only:
                return
            while True:
                chunk = fh.read(CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError:
            self.close_connection = True
        finally:
            fh.close()

    def audio(self, job, head_only: bool = False):
        """Serve the original upload with Range support.

        Chrome on Android will not let you seek in a long recording unless the
        server answers Range requests with 206 — without this, tapping a
        timestamp two hours in does nothing.
        """
        if not job.audio_file:
            return self.fail("No audio stored for this transcript.", 404)
        path = Path(job.audio_file)
        if not path.exists():
            return self.fail("The audio file has been deleted.", 404)

        try:
            fh = open(path, "rb")
        except OSError:
            # Opening before we send any headers matters: once the status line
            # is out, an error here would fall through to the generic 500
            # handler and write a *second* response into the same connection,
            # which hangs the browser rather than failing cleanly.
            return self.fail("The audio file could not be read.", 404)

        size = os.fstat(fh.fileno()).st_size   # same fd, so no TOCTOU with stat()
        ctype = job.media_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ";" in ctype:
            ctype = ctype.split(";")[0].strip()
        rng = self.headers.get("Range", "")
        start, end = 0, size - 1
        status = 200
        _close = fh.close

        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m and size:
            g1, g2 = m.group(1), m.group(2)
            if g1:
                start = int(g1)
                end = int(g2) if g2 else size - 1
            elif g2:
                start = max(0, size - int(g2))     # suffix range: last N bytes
            end = min(end, size - 1)
            if start >= size or start > end:
                # "bytes=100-50" would otherwise yield a negative Content-Length.
                _close()
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            _close()
            return

        try:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass                        # seeking mid-download does this; harmless
        except OSError:
            # Headers are already out, so there is no way to report this in
            # band. Drop the connection rather than desync it.
            self.close_connection = True
        finally:
            _close()

    # -------------------- SSE --------------------

    def sse(self):
        store = jobs_mod.store()
        q = store.subscribe()
        self.close_connection = True    # streamed body: no keep-alive reuse
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            # Tell the browser to back off if the connection drops; the default
            # 3s reconnect storm is wasteful on a phone radio.
            self.wfile.write(b"retry: 5000\n\n")
            for job in store.list():
                self._sse_frame({"type": "job", "job": job.public()})
            self.wfile.flush()

            while True:
                try:
                    event = q.get(timeout=15)
                    self._sse_frame(event)
                except queue.Empty:
                    # Comment frame: keeps the socket from being reaped by
                    # Android's radio power management while a job runs.
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass
        finally:
            store.unsubscribe(q)

    def _sse_frame(self, event: dict):
        payload = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))


# ---------------------------------------------------------------- helpers

def _load_segments(data: dict) -> list:
    out = []
    for s in data.get("segments", []):
        words = [Word(w["start"], w["end"], w["text"], w.get("speaker"), w.get("confidence"))
                 for w in s.get("words", [])]
        out.append(Segment(s["start"], s["end"], s.get("speaker"), s.get("text", ""), words))
    return out


def _safe_filename(name: str) -> str:
    name = os.path.basename(name).replace("\x00", "")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return (name or "audio")[:120]


# Query params the upload endpoint consumes itself; everything else is handed
# to the engine, which is how per-job engine tuning gets through.
_UPLOAD_PARAMS = {"name", "engine", "language", "token"}


def _job_options(q: dict) -> dict:
    opts = {
        "num_speakers": _int(q.get("num_speakers"), 0),
        "min_speakers": _int(q.get("min_speakers"), 0),
        "max_speakers": _int(q.get("max_speakers"), 0),
    }
    for k, v in q.items():
        if k not in _UPLOAD_PARAMS and k not in opts:
            opts[k] = v
    return opts


def _choice(value, allowed):
    """Only ever pass through a name the module itself defines.

    These land in ffmpeg arguments, so an unrecognised one is dropped rather
    than forwarded and the server-side default applies.
    """
    name = str(value or "").strip()
    return name if name in allowed else None


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cookie(header: str, key: str) -> str:
    for part in header.split(";"):
        k, _, v = part.strip().partition("=")
        if k == key:
            return v
    return ""


def local_ips() -> list:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return ips


def serve(host: str = None, port: int = None) -> None:
    global AUTH_TOKEN

    cfg = config.load()
    host = host or cfg["host"]
    port = port or cfg["port"]
    config.ensure_dirs()

    # Import for the registration side effects, then wire the worker.
    from .engines import registry  # noqa: F401
    jobs_mod.store().set_runner(runner.run_job)

    # An upload is written to disk before the job that references it exists, so
    # a process killed in between leaves a file nothing points at. Starting up
    # is precisely when to notice, since it is the same event.
    from . import storage
    count, freed = storage.reap()
    if count:
        print(f"  Reclaimed {storage.human(freed)} from {count} interrupted "
              f"upload(s).")

    loopback = host in ("127.0.0.1", "localhost", "::1")
    if not loopback:
        AUTH_TOKEN = os.environ.get("TRANSCRIBE_TOKEN") or secrets.token_urlsafe(16)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True

    url = f"http://{'127.0.0.1' if loopback else host}:{port}/"
    print()
    print("  Transcribe is running.")
    print()
    print(f"  Open this in Chrome:  {url}{'?token=' + AUTH_TOKEN if AUTH_TOKEN else ''}")
    if not loopback:
        for ip in local_ips():
            print(f"  From another device:  http://{ip}:{port}/?token={AUTH_TOKEN}")
        print()
        print("  Note: this port is open to your local network. The token above is")
        print("  what keeps other devices on that network out.")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping…")
    finally:
        httpd.shutdown()
        httpd.server_close()
