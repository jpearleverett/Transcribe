"""Engine interface and registry.

An engine takes audio and returns words with timestamps, plus either per-word
speaker labels or a diarization timeline. Everything downstream (speaker
attribution, segmentation, export) is shared, so adding a provider means
implementing one method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..align import Word, Turn


class EngineError(RuntimeError):
    """A failure worth showing the user verbatim."""


@dataclass
class Context:
    """Everything an engine needs for one job."""
    job_id: str
    source: Path                   # the file exactly as uploaded
    duration: float                # seconds, 0 if unknown
    language: str = "auto"         # ISO code or "auto"
    num_speakers: int = 0          # 0 = unknown
    min_speakers: int = 0
    max_speakers: int = 0
    options: dict = field(default_factory=dict)

    # Injected by the runner
    progress: Callable = lambda stage, frac: None
    log: Callable = lambda msg: None
    cancelled: Callable = lambda: False
    workdir: Path = Path(".")

    _wav: Optional[Path] = None

    def check_cancel(self) -> None:
        if self.cancelled():
            raise Cancelled()

    def wav16k(self) -> Path:
        """16 kHz mono WAV, converted once and cached for the job.

        Both the local engine (which requires it) and the cloud engines (which
        upload far less data because of it) go through here.
        """
        from .. import audio
        if self._wav and self._wav.exists():
            return self._wav
        dst = self.workdir / f"{self.job_id}.16k.wav"
        self.progress("converting", 0.0)
        self.log("Converting audio to 16 kHz mono")
        audio.to_wav16k(self.source, dst, duration=self.duration,
                        on_progress=lambda f: self.progress("converting", f))
        self._wav = dst
        return dst

    def opus(self, bitrate: str = "48k") -> Path:
        """Small mono Opus copy, for uploading over a metered connection."""
        from .. import audio
        dst = self.workdir / f"{self.job_id}.48k.opus"
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        self.log("Compressing audio for upload")
        return audio.to_compressed(self.source, dst, bitrate=bitrate)


class Cancelled(RuntimeError):
    pass


@dataclass
class Result:
    words: list = field(default_factory=list)     # list[Word]
    turns: Optional[list] = None                  # list[Turn] or None
    language: str = ""
    model: str = ""
    text: str = ""                                # engine's own plain text, if any
    raw: Optional[dict] = None
    # Set only by engines that return speaker-labelled *segments* with no word
    # timings (OpenAI's diarize model, for one). Kept separate from `words`
    # rather than faking per-word timestamps we do not actually have.
    segments: Optional[list] = None               # list[Segment]

    def is_empty(self) -> bool:
        return not self.words and not self.segments and not self.text.strip()


class Engine:
    name = ""
    label = ""
    description = ""
    needs_key = True
    signup_url = ""
    key_help = ""
    # Roughly how many seconds of audio per second of wall clock, for ETAs.
    speed_factor = 30.0
    # Extra config this engine needs beyond an API key. Each entry names a key
    # in config.DEFAULTS and is rendered in the Settings sheet.
    config_fields: list = []

    def available(self) -> tuple:
        """Return (ok, reason). Reason is shown when ok is False."""
        return True, ""

    def has_key(self) -> bool:
        from .. import config
        return bool(config.api_key(self.name)) if self.needs_key else True

    def transcribe(self, ctx: Context) -> Result:
        raise NotImplementedError

    # -- helpers shared by the HTTP-based engines --------------------

    def key(self) -> str:
        from .. import config
        k = config.api_key(self.name)
        if not k and self.needs_key:
            raise EngineError(f"No API key set for {self.label}. Add one in Settings.")
        return k

    def describe(self) -> dict:
        from .. import config
        ok, reason = self.available()
        cfg = config.load()
        return {
            "config_fields": [
                {**f, "value": cfg.get(f["key"], "")} for f in self.config_fields
            ],
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "needs_key": self.needs_key,
            "has_key": self.has_key(),
            "available": ok,
            "unavailable_reason": reason,
            "signup_url": self.signup_url,
            "key_help": self.key_help,
        }


_REGISTRY: dict = {}


def register(engine_cls) -> type:
    engine = engine_cls()
    _REGISTRY[engine.name] = engine
    return engine_cls


def get(name: str) -> Engine:
    if name not in _REGISTRY:
        raise EngineError(f"Unknown engine '{name}'.")
    return _REGISTRY[name]


def all_engines() -> list:
    return list(_REGISTRY.values())


def describe_all() -> list:
    return [e.describe() for e in _REGISTRY.values()]


def normalize_speaker(value) -> Optional[str]:
    """Turn whatever an engine calls a speaker into a short stable label."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.upper().startswith("SPEAKER_"):
        tail = s.split("_", 1)[1].lstrip("0") or "0"
        return tail
    return s
