"""A small HTTP client built on urllib.

Deliberately dependency-free: `pip install requests` is one more thing that can
fail on a phone, and every provider we talk to is a plain JSON/multipart REST
API. Adds the things urllib lacks out of the box: streaming file upload without
reading the file into RAM, retry with backoff on transient failures, and error
bodies that actually say what went wrong.
"""

from __future__ import annotations

import gzip
import io
import json
import mimetypes
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Optional

USER_AGENT = "Transcribe/1.0 (+termux)"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
DEFAULT_TIMEOUT = 300


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {body[:600]}")


def _ssl_context() -> ssl.SSLContext:
    # Termux ships its own CA bundle; certifi is not installed and we don't
    # want to require it. create_default_context() finds the system store.
    return ssl.create_default_context()


class _ProgressFile(io.RawIOBase):
    """File wrapper that reports upload progress and supports abort.

    urllib will happily accept any object with .read(); giving it a real
    file-like object (rather than bytes) is what keeps a 500 MB upload from
    being loaded into a phone's RAM.
    """

    def __init__(self, path: Path, on_progress: Optional[Callable] = None,
                 should_abort: Optional[Callable] = None,
                 prefix: bytes = b"", suffix: bytes = b""):
        self._fh = open(path, "rb")
        self._total = os.path.getsize(path) + len(prefix) + len(suffix)
        self._sent = 0
        self._on_progress = on_progress
        self._should_abort = should_abort
        self._prefix = io.BytesIO(prefix)
        self._suffix = io.BytesIO(suffix)
        self._last_report = 0.0

    @property
    def total(self) -> int:
        return self._total

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._should_abort and self._should_abort():
            raise Aborted("upload cancelled")
        if size is None or size < 0:
            size = 1 << 20
        chunk = self._prefix.read(size)
        if not chunk:
            chunk = self._fh.read(size)
        if not chunk:
            chunk = self._suffix.read(size)
        self._sent += len(chunk)
        if self._on_progress and chunk:
            now = time.monotonic()
            if now - self._last_report > 0.25 or self._sent >= self._total:
                self._last_report = now
                self._on_progress(self._sent, self._total)
        return chunk

    def close(self):
        try:
            self._fh.close()
        finally:
            super().close()


class Aborted(RuntimeError):
    pass


def request(
    method: str,
    url: str,
    *,
    headers: Optional[dict] = None,
    json_body: Optional[dict] = None,
    data: Optional[bytes] = None,
    body_file: Optional[_ProgressFile] = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 3,
    parse_json: bool = True,
    should_abort: Optional[Callable] = None,
):
    headers = dict(headers or {})
    headers.setdefault("User-Agent", USER_AGENT)
    headers.setdefault("Accept-Encoding", "gzip")

    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    last_err = None
    for attempt in range(retries + 1):
        if should_abort and should_abort():
            raise Aborted("cancelled")
        try:
            if body_file is not None:
                if attempt > 0:
                    # A streamed body can't be replayed; the caller rebuilds it.
                    raise last_err or RuntimeError("cannot retry a streamed body")
                headers["Content-Length"] = str(body_file.total)
                payload = body_file
            else:
                payload = data

            req = urllib.request.Request(url, data=payload, headers=headers, method=method.upper())
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                text = raw.decode("utf-8", "replace")
                if parse_json:
                    try:
                        return json.loads(text) if text.strip() else {}
                    except json.JSONDecodeError:
                        return {"_raw": text}
                return text

        except urllib.error.HTTPError as e:
            raw = e.read()
            if e.headers.get("Content-Encoding") == "gzip":
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    pass
            body = raw.decode("utf-8", "replace")
            last_err = HttpError(e.code, body, url)
            if e.code in RETRY_STATUS and attempt < retries and body_file is None:
                time.sleep(_backoff(attempt, e.headers.get("Retry-After")))
                continue
            raise last_err
        except Aborted:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, OSError) as e:
            last_err = RuntimeError(f"network error calling {url}: {e}")
            if attempt < retries and body_file is None:
                time.sleep(_backoff(attempt, None))
                continue
            raise last_err
    raise last_err or RuntimeError("request failed")


def _backoff(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(2.0 ** attempt, 16.0)


def get(url: str, **kw):
    return request("GET", url, **kw)


def post(url: str, **kw):
    return request("POST", url, **kw)


def upload_raw(url: str, path: Path, *, headers: Optional[dict] = None,
               content_type: Optional[str] = None, on_progress=None,
               should_abort=None, timeout: int = 3600):
    """POST a file as the raw request body (AssemblyAI/Deepgram style)."""
    headers = dict(headers or {})
    headers["Content-Type"] = content_type or _guess_type(path)
    bf = _ProgressFile(path, on_progress, should_abort)
    try:
        return request("POST", url, headers=headers, body_file=bf, timeout=timeout,
                       retries=0, should_abort=should_abort)
    finally:
        bf.close()


def upload_multipart(url: str, path: Path, *, field: str = "file",
                     fields: Optional[dict] = None, headers: Optional[dict] = None,
                     content_type: Optional[str] = None, filename: Optional[str] = None,
                     on_progress=None, should_abort=None, timeout: int = 3600):
    """POST a file as multipart/form-data, streaming it from disk.

    Built by hand because urllib has no multipart encoder and the obvious
    workaround (read the file, concatenate bytes) would put a 500 MB recording
    into a phone's memory.
    """
    boundary = f"----transcribe{uuid.uuid4().hex}"
    filename = filename or path.name
    ctype = content_type or _guess_type(path)

    pre = io.BytesIO()
    for k, v in (fields or {}).items():
        if v is None:
            continue
        pre.write(f"--{boundary}\r\n".encode())
        pre.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        pre.write(f"{v}\r\n".encode())
    pre.write(f"--{boundary}\r\n".encode())
    pre.write(
        f'Content-Disposition: form-data; name="{field}"; filename="{_escape(filename)}"\r\n'.encode()
    )
    pre.write(f"Content-Type: {ctype}\r\n\r\n".encode())

    post_bytes = f"\r\n--{boundary}--\r\n".encode()

    hdrs = dict(headers or {})
    hdrs["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    bf = _ProgressFile(path, on_progress, should_abort, prefix=pre.getvalue(), suffix=post_bytes)
    try:
        return request("POST", url, headers=hdrs, body_file=bf, timeout=timeout,
                       retries=0, should_abort=should_abort)
    finally:
        bf.close()


def _escape(name: str) -> str:
    return name.replace('"', "'").replace("\r", " ").replace("\n", " ")


def _guess_type(path: Path) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    return {
        ".m4a": "audio/mp4", ".opus": "audio/ogg", ".oga": "audio/ogg",
        ".amr": "audio/amr", ".caf": "audio/x-caf", ".aiff": "audio/aiff",
    }.get(path.suffix.lower(), "application/octet-stream")
