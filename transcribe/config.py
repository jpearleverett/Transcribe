"""Configuration: defaults, a JSON file on disk, and environment overrides.

API keys live in ~/.transcribe/config.json with 0600 permissions rather than in
the repo, so a stray `git add -A` can never publish them.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_LOCK = threading.RLock()

HOME = Path(os.environ.get("TRANSCRIBE_HOME", Path.home() / ".transcribe"))
CONFIG_PATH = HOME / "config.json"
UPLOAD_DIR = HOME / "uploads"
JOB_DIR = HOME / "jobs"
MODEL_DIR = HOME / "models"
BIN_DIR = HOME / "bin"

DEFAULTS = {
    "engine": "assemblyai",
    "keys": {},                  # engine name -> API key
    "language": "auto",
    "num_speakers": 0,           # 0 = let the engine decide
    "min_speakers": 0,
    "max_speakers": 0,
    "port": 8756,
    "host": "127.0.0.1",
    "keep_audio": True,          # keep uploads so the player can seek
    # Extracting audio from local video
    "media_roots": [],           # extra folders to browse, beyond the defaults
    "extract_dir": "",           # where extracted audio lands ("" = Downloads)
    "extract_mode": "copy",      # copy | opus | mp3 | wav
    "max_upload_mb": 2048,
    # Local engine settings
    "local_model": "ggml-small.en-q5_1.bin",
    "local_threads": 0,          # 0 = auto-detect
    "local_diarize": True,
    # RunPod serverless
    "runpod_endpoint": "",       # endpoint id, from the RunPod console
    "runpod_model": "large-v3",
    "runpod_audio_url": "",      # optional: a public base URL, if you have one
    "runpod_max_payload_mb": 19, # inline base64 budget (RunPod caps /runsync at 20 MB)
    # Segmentation tuning, exposed in the UI's advanced panel
    "max_gap": 1.0,
    "max_dur": 30.0,
    "max_chars": 320,
    "min_run_words": 2,
    "min_run_dur": 0.40,
}

# Environment overrides let you run without ever touching the settings UI.
ENV_KEYS = {
    "runpod": "RUNPOD_API_KEY",
    "assemblyai": "ASSEMBLYAI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "speechmatics": "SPEECHMATICS_API_KEY",
}

_cache: dict | None = None


def ensure_dirs() -> None:
    for d in (HOME, UPLOAD_DIR, JOB_DIR, MODEL_DIR, BIN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    try:
        HOME.chmod(0o700)
    except OSError:
        pass


def load(force: bool = False) -> dict:
    global _cache
    with _LOCK:
        if _cache is not None and not force:
            return _cache
        ensure_dirs()
        cfg = dict(DEFAULTS)
        cfg["keys"] = dict(DEFAULTS["keys"])
        if CONFIG_PATH.exists():
            try:
                stored = json.loads(CONFIG_PATH.read_text())
                keys = stored.pop("keys", None)
                cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
                if isinstance(keys, dict):
                    cfg["keys"].update({k: v for k, v in keys.items() if isinstance(v, str)})
            except (json.JSONDecodeError, OSError) as e:
                # A corrupt config must not brick the server; fall back to
                # defaults and keep the bad file around for inspection.
                print(f"[config] ignoring unreadable {CONFIG_PATH}: {e}")
        # Environment keys are a *default*, not an override: a key typed into
        # Settings must win, or saving one appears to do nothing and there is no
        # way to tell why.
        cfg["env_keys"] = []
        for engine, env in ENV_KEYS.items():
            val = (os.environ.get(env) or "").strip()
            if val and not cfg["keys"].get(engine):
                cfg["keys"][engine] = val
                cfg["env_keys"].append(engine)
        if os.environ.get("RUNPOD_ENDPOINT_ID") and not cfg.get("runpod_endpoint"):
            cfg["runpod_endpoint"] = os.environ["RUNPOD_ENDPOINT_ID"].strip()
        if os.environ.get("TRANSCRIBE_PORT"):
            try:
                cfg["port"] = int(os.environ["TRANSCRIBE_PORT"])
            except ValueError:
                pass
        if os.environ.get("TRANSCRIBE_HOST"):
            cfg["host"] = os.environ["TRANSCRIBE_HOST"]
        _cache = cfg
        return cfg


def save(update: dict) -> dict:
    """Merge `update` into the stored config and write it back atomically."""
    with _LOCK:
        cfg = load()
        stored = {}
        if CONFIG_PATH.exists():
            try:
                stored = json.loads(CONFIG_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                stored = {}
        stored.setdefault("keys", {})

        for k, v in update.items():
            if k == "keys" and isinstance(v, dict):
                for engine, key in v.items():
                    key = (key or "").strip()
                    if key:
                        stored["keys"][engine] = key
                    else:
                        stored["keys"].pop(engine, None)
            elif k in DEFAULTS:
                stored[k] = _coerce(k, v)

        ensure_dirs()
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(stored, indent=2))
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(CONFIG_PATH)
        try:
            CONFIG_PATH.chmod(0o600)
        except OSError:
            pass
        return load(force=True)


def _coerce(key: str, value):
    default = DEFAULTS[key]
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
    except (TypeError, ValueError):
        return default
    return value


def api_key(engine: str) -> str:
    return (load().get("keys") or {}).get(engine, "")


def redacted() -> dict:
    """Config safe to hand to the browser: keys become presence booleans."""
    cfg = dict(load())
    keys = cfg.pop("keys", {})
    cfg["has_key"] = {k: bool(v) for k, v in keys.items() if v}
    cfg["from_env"] = list(cfg.pop("env_keys", []))
    cfg["key_hint"] = {
        k: (v[:3] + "…" + v[-4:]) if len(v) > 10 else "…"
        for k, v in keys.items() if v
    }
    return cfg
