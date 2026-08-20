"""Job store and background worker.

The central design decision: transcription runs entirely in the Termux-side
worker thread and its state is persisted to disk after every change. The browser
is a *viewer*, not the driver. So when Android locks the screen, throttles the
background tab, or kills Chrome outright, the job keeps running and the page
picks up exactly where it left off on reconnect. Nothing depends on the
connection staying open.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from . import config

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL = {DONE, FAILED, CANCELLED}


@dataclass
class Job:
    id: str
    name: str
    status: str = PENDING
    stage: str = "queued"
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    engine: str = ""
    model: str = ""
    language: str = ""
    duration: float = 0.0
    size: int = 0
    error: str = ""
    audio_file: str = ""       # original upload, served to the <audio> player
    media_type: str = ""
    speakers: dict = field(default_factory=dict)   # label -> user-given name
    options: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("audio_file", None)
        d["has_audio"] = bool(self.audio_file and Path(self.audio_file).exists())
        return d


class JobStore:
    def __init__(self):
        self.dir = config.JOB_DIR
        self._jobs: dict = {}
        self._lock = threading.RLock()
        self._listeners: list = []
        self._cancelled: set = set()
        self._queue: queue.Queue = queue.Queue()
        self._runner: Optional[Callable] = None
        self._worker: Optional[threading.Thread] = None
        self._load_all()

    # ---------------- persistence ----------------

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _result_path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.result.json"

    def _load_all(self) -> None:
        config.ensure_dirs()
        for p in sorted(self.dir.glob("*.json")):
            if p.name.endswith(".result.json"):
                continue
            try:
                data = json.loads(p.read_text())
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
                # A job that was mid-flight when Termux died is not coming back
                # on its own; mark it so the UI can offer a retry.
                if job.status in (RUNNING, PENDING):
                    job.status = FAILED
                    job.error = job.error or "Interrupted — the server stopped while this job was running. Tap Retry."
                    job.stage = "interrupted"
                    self._jobs[job.id] = job
                    self._persist(job)
                else:
                    self._jobs[job.id] = job
            except (json.JSONDecodeError, OSError, TypeError) as e:
                print(f"[jobs] skipping unreadable {p.name}: {e}")

    def _persist(self, job: Job) -> None:
        try:
            tmp = self._path(job.id).with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(job), ensure_ascii=False))
            tmp.replace(self._path(job.id))
        except OSError as e:
            print(f"[jobs] failed to persist {job.id}: {e}")

    def save_result(self, job_id: str, payload: dict) -> None:
        tmp = self._result_path(job_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(self._result_path(job_id))

    def load_result(self, job_id: str) -> Optional[dict]:
        p = self._result_path(job_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    # ---------------- CRUD ----------------

    def create(self, name: str, **kw) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, **kw)
        with self._lock:
            self._jobs[job.id] = job
            self._persist(job)
        self._emit(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.created)

    def update(self, job_id: str, **kw) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for k, v in kw.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated = time.time()
            self._persist(job)
        self._emit(job)
        return job

    def add_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.log.append({"t": round(time.time() - job.created, 1), "m": message})
            job.log = job.log[-200:]
            job.updated = time.time()
            self._persist(job)
        self._emit(job)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            self._cancelled.discard(job_id)
        if not job:
            return False
        for p in (self._path(job_id), self._result_path(job_id)):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        if job.audio_file:
            try:
                Path(job.audio_file).unlink(missing_ok=True)
            except OSError:
                pass
        self._broadcast({"type": "deleted", "id": job_id})
        return True

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in TERMINAL:
                return False
            self._cancelled.add(job_id)
        self.update(job_id, status=CANCELLED, stage="cancelled", error="Cancelled")
        return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    # ---------------- progress ----------------

    def progress(self, job_id: str, stage: str, fraction: float) -> None:
        """Update coarse progress. Rate-limited so a fast upload can't spam SSE."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            fraction = max(0.0, min(1.0, fraction))
            same_stage = job.stage == stage
            tiny_move = abs(fraction - job.progress) < 0.01
            recent = time.time() - job.updated < 0.4
            job.stage = stage
            job.progress = fraction
            job.updated = time.time()
            if same_stage and tiny_move and recent:
                return          # skip both the disk write and the SSE frame
            self._persist(job)
        self._emit(job)

    # ---------------- pub/sub for SSE ----------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def _emit(self, job: Job) -> None:
        self._broadcast({"type": "job", "job": job.public()})

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for q in listeners:
            try:
                q.put_nowait(event)
            except queue.Full:
                # A browser that stopped reading (backgrounded tab, dead socket)
                # must never block the worker. It will resync on reconnect.
                pass

    # ---------------- worker ----------------

    def set_runner(self, runner: Callable) -> None:
        self._runner = runner

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._work_loop, name="job-worker", daemon=True)
            self._worker.start()

    def _work_loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=30)
            except queue.Empty:
                return  # idle: let the thread go, enqueue() will start a new one
            job = self.get(job_id)
            if not job or self.is_cancelled(job_id):
                continue
            try:
                self.update(job_id, status=RUNNING, stage="starting", progress=0.0, error="")
                self._runner(job)
                if self.is_cancelled(job_id):
                    self.update(job_id, status=CANCELLED, stage="cancelled")
                else:
                    self.update(job_id, status=DONE, stage="done", progress=1.0)
            except Exception as e:                      # noqa: BLE001 - worker must never die
                if self.is_cancelled(job_id):
                    self.update(job_id, status=CANCELLED, stage="cancelled")
                else:
                    detail = str(e) or e.__class__.__name__
                    traceback.print_exc()
                    self.update(job_id, status=FAILED, stage="failed", error=detail[:2000])
            finally:
                with self._lock:
                    self._cancelled.discard(job_id)

    def queue_depth(self) -> int:
        return self._queue.qsize()


STORE: Optional[JobStore] = None


def store() -> JobStore:
    global STORE
    if STORE is None:
        STORE = JobStore()
    return STORE
