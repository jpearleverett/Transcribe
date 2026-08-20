"""Environment check: `python -m transcribe --check`.

This app runs on a device its author cannot test on, against APIs that change
under it. So rather than letting a first run fail somewhere deep in a worker
thread, this walks the whole setup and says exactly what is wrong and the exact
command that fixes it.

`--check --network` additionally proves each configured API key actually works,
which turns "transcription failed" into "your key is fine but the endpoint id is
wrong".
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from . import audio, config
from .engines import base as engines, registry  # noqa: F401  (registers engines)

OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"

MARKS = {OK: "  \033[32m✓\033[0m", WARN: "  \033[33m!\033[0m",
         BAD: "  \033[31m✗\033[0m", INFO: "   "}


class Report:
    def __init__(self):
        self.rows = []
        self.problems = 0
        self.warnings = 0

    def add(self, level, text, fix=""):
        self.rows.append((level, text, fix))
        if level == BAD:
            self.problems += 1
        elif level == WARN:
            self.warnings += 1

    def section(self, title):
        self.rows.append((None, title, ""))

    def render(self):
        out = []
        for level, text, fix in self.rows:
            if level is None:
                out.append(f"\n\033[1m{text}\033[0m")
            else:
                out.append(f"{MARKS[level]} {text}")
                if fix:
                    out.append(f"      → {fix}")
        return "\n".join(out)


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def check_python(rep):
    rep.section("Python")
    v = sys.version_info
    if v >= (3, 8):
        rep.add(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        rep.add(BAD, f"Python {v.major}.{v.minor} is too old (need 3.8+)",
                "pkg install python")


def check_ffmpeg(rep):
    rep.section("Audio tools")
    if not audio.have_ffmpeg():
        rep.add(BAD, "ffmpeg is missing — no format conversion or compression",
                "pkg install ffmpeg")
        return
    code, out = _run([audio.FFMPEG, "-version"])
    ver = out.split("\n")[0].split(" version ")[-1].split()[0] if code == 0 else "?"
    rep.add(OK, f"ffmpeg {ver}")

    if audio.FFPROBE:
        rep.add(OK, "ffprobe present (duration and format detection)")
    else:
        rep.add(WARN, "ffprobe missing — durations fall back to a WAV header read",
                "pkg install ffmpeg")

    # libopus decides whether uploads get compressed ~20x before leaving the
    # phone, which on mobile data is the difference between 13 MB and 300 MB.
    code, out = _run([audio.FFMPEG, "-hide_banner", "-encoders"])
    if code == 0 and "libopus" in out:
        rep.add(OK, "libopus encoder (uploads are compressed before sending)")
    else:
        rep.add(WARN, "no libopus encoder — audio uploads at full size",
                "pkg install ffmpeg  (or reinstall it; this build lacks libopus)")


def check_termux(rep):
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" not in prefix:
        return
    rep.section("Termux")
    if shutil.which("termux-wake-lock"):
        rep.add(OK, "termux-wake-lock available")
    else:
        rep.add(WARN, "termux-wake-lock missing — Android may freeze long jobs",
                "pkg install termux-tools")
    rep.add(INFO, "If long jobs still die with the screen off: Settings → Apps → "
                  "Termux → Battery → Unrestricted, and on Android 14+ enable "
                  "Developer options → Disable child process restrictions.")


def check_storage(rep):
    rep.section("Storage")
    try:
        config.ensure_dirs()
        probe = config.HOME / ".write-test"
        probe.write_bytes(b"x")
        probe.unlink()
        rep.add(OK, f"{config.HOME} is writable")
    except OSError as e:
        rep.add(BAD, f"cannot write to {config.HOME}: {e}")
        return

    if config.CONFIG_PATH.exists():
        mode = config.CONFIG_PATH.stat().st_mode & 0o777
        if mode == 0o600:
            rep.add(OK, "config.json is private (0600)")
        else:
            rep.add(WARN, f"config.json is mode {mode:o}, expected 600 — it holds API keys",
                    f"chmod 600 {config.CONFIG_PATH}")

    try:
        st = os.statvfs(config.HOME)
        free_gb = st.f_bavail * st.f_frsize / 1e9
        used = sum(f.stat().st_size for f in config.UPLOAD_DIR.glob("*") if f.is_file()) / 1e6
        level = OK if free_gb > 2 else (WARN if free_gb > 0.5 else BAD)
        rep.add(level, f"{free_gb:.1f} GB free · {used:.0f} MB of stored audio",
                "Delete old transcripts in the app to reclaim space." if level != OK else "")
    except OSError:
        pass


def check_port(rep):
    cfg = config.load()
    rep.section("Network")
    host, port = cfg["host"], cfg["port"]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    in_use = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    if in_use:
        rep.add(INFO, f"Port {port} is already serving — the app is probably running: "
                      f"http://127.0.0.1:{port}/")
    else:
        rep.add(OK, f"Port {port} is free")
    if host not in ("127.0.0.1", "localhost", "::1"):
        rep.add(WARN, f"Bound to {host}, which exposes this to your local network",
                "Access is token-protected, but prefer the default 127.0.0.1 unless "
                "you need another device.")


def check_engines(rep, network=False):
    rep.section("Engines")
    cfg = config.load()
    default = cfg["engine"]
    usable = 0

    for eng in engines.all_engines():
        if eng.name == "mock":
            continue
        available, reason = eng.available()
        has_key = eng.has_key()
        marker = " (default)" if eng.name == default else ""

        if available and has_key:
            usable += 1
            src = " [key from environment]" if eng.name in cfg.get("env_keys", []) else ""
            rep.add(OK, f"{eng.label}{marker}{src}")
        elif not has_key:
            rep.add(INFO, f"{eng.label}{marker} — no API key set",
                    f"Add one in Settings, or export {config.ENV_KEYS.get(eng.name, 'THE_KEY')}=…")
        else:
            level = BAD if eng.name == default else INFO
            rep.add(level, f"{eng.label}{marker} — {reason}")

    if usable == 0:
        rep.add(BAD, "No engine is ready, so nothing can be transcribed yet",
                "Add an API key in Settings. Deepgram gives $200 of free credit "
                "with no card: https://console.deepgram.com/signup")

    if network:
        _check_credentials(rep)


def _check_credentials(rep):
    """Prove each configured key actually works, without transcribing anything."""
    from . import httpclient as http

    rep.section("Credential checks (live)")
    cfg = config.load()
    probes = {
        "deepgram": ("https://api.deepgram.com/v1/projects",
                     lambda k: {"Authorization": f"Token {k}"}),
        "assemblyai": ("https://api.assemblyai.com/v2/transcript?limit=1",
                       lambda k: {"Authorization": k}),
        "elevenlabs": ("https://api.elevenlabs.io/v1/user",
                       lambda k: {"xi-api-key": k}),
        "openai": ("https://api.openai.com/v1/models",
                   lambda k: {"Authorization": f"Bearer {k}"}),
    }

    for name, (url, headers) in probes.items():
        key = config.api_key(name)
        if not key:
            continue
        try:
            http.get(url, headers=headers(key), timeout=20, retries=0)
            rep.add(OK, f"{name}: key accepted")
        except http.HttpError as e:
            hint = "the key is wrong or revoked" if e.status in (401, 403) else f"HTTP {e.status}"
            rep.add(BAD, f"{name}: {hint}", "Paste a fresh key in Settings.")
        except Exception as e:                       # noqa: BLE001
            rep.add(WARN, f"{name}: could not reach the API ({type(e).__name__})")

    _check_runpod(rep)


def _check_runpod(rep):
    from . import httpclient as http

    key = config.api_key("runpod")
    if not key:
        return
    endpoint = (config.load().get("runpod_endpoint") or "").strip()
    try:
        data = http.get("https://rest.runpod.io/v1/endpoints",
                        headers={"Authorization": f"Bearer {key}"}, timeout=25, retries=0)
    except http.HttpError as e:
        rep.add(BAD, f"runpod: key rejected (HTTP {e.status})", "Paste a fresh key in Settings.")
        return
    except Exception as e:                           # noqa: BLE001
        rep.add(WARN, f"runpod: could not reach the API ({type(e).__name__})")
        return

    rep.add(OK, "runpod: key accepted")
    found = [e for e in (data or []) if isinstance(e, dict)]
    if not endpoint:
        if found:
            names = ", ".join(f"{e.get('id')} ({e.get('name')})" for e in found[:5])
            rep.add(WARN, "runpod: no endpoint id configured", f"Available: {names}")
        else:
            rep.add(WARN, "runpod: this account has no serverless endpoints yet",
                    "Deploy the worker in gpu/ — see gpu/README.md")
        return

    if any(e.get("id") == endpoint for e in found):
        rep.add(OK, f"runpod: endpoint {endpoint} exists")
    else:
        ids = ", ".join(str(e.get("id")) for e in found) or "none"
        rep.add(BAD, f"runpod: endpoint '{endpoint}' not found on this account",
                f"Endpoints on the account: {ids}")


def check_local_engine(rep):
    eng = engines.get("local")
    available, reason = eng.available()
    if not available and not eng.whisper_bin():
        return          # not installed and not asked for; nothing to report
    rep.section("Offline engine")

    binary = eng.whisper_bin()
    if binary:
        code, out = _run([binary, "--help"], timeout=25)
        if code == 0 or "usage" in out.lower():
            rep.add(OK, f"whisper.cpp at {binary}")
        else:
            rep.add(BAD, f"whisper.cpp at {binary} will not run",
                    "Rebuild with ./install.sh --local")
    else:
        rep.add(BAD, "whisper.cpp not built", "./install.sh --local")

    model = eng.model_path()
    if model.exists():
        rep.add(OK, f"model {model.name} ({model.stat().st_size / 1e6:.0f} MB)")
    else:
        rep.add(BAD, f"no speech model in {model.parent}", "./install.sh --local")

    seg, emb = eng.diarizer_models()
    if seg and emb:
        rep.add(OK, f"speaker models ({seg.name}, {emb.name})")
    else:
        rep.add(WARN, "speaker models missing — offline transcripts get no speaker labels",
                "./install.sh --local")

    try:
        import sherpa_onnx                                   # noqa: F401
        rep.add(OK, "sherpa-onnx importable")
    except ImportError:
        rep.add(WARN, "sherpa-onnx not installed — no offline diarization",
                "./install.sh --local  (it compiles against Termux's onnxruntime)")


def check_pipeline(rep):
    """Exercise the attribution and export code on known input."""
    rep.section("Self-test")
    try:
        from .align import Word, Turn, diarize_transcript
        from .exporters import export
        words = [Word(0.0, 0.4, "Hello"), Word(0.4, 0.9, "there."),
                 Word(1.2, 1.6, "Hi."), Word(1.7, 2.2, "Welcome.")]
        turns = [Turn(0.0, 1.0, "A"), Turn(1.0, 3.0, "B")]
        segs = diarize_transcript(words, turns)
        speakers = {s.speaker for s in segs}
        if speakers != {"A", "B"}:
            rep.add(BAD, f"speaker attribution produced {speakers}, expected A and B")
            return
        for fmt in ("txt", "srt", "vtt", "json", "csv", "md"):
            _, _, body = export(fmt, segs, None, {"name": "check"})
            if not body.strip():
                rep.add(BAD, f"{fmt} export produced nothing")
                return
        rep.add(OK, "attribution and all six export formats work")
    except Exception as e:                                   # noqa: BLE001
        rep.add(BAD, f"self-test failed: {type(e).__name__}: {e}")


def run(network: bool = False) -> int:
    rep = Report()
    print("\n\033[1mTranscribe — checking your setup\033[0m")
    check_python(rep)
    check_ffmpeg(rep)
    check_termux(rep)
    check_storage(rep)
    check_port(rep)
    check_engines(rep, network=network)
    check_local_engine(rep)
    check_pipeline(rep)
    print(rep.render())

    print()
    if rep.problems:
        print(f"\033[31m{rep.problems} problem(s)\033[0m"
              + (f", {rep.warnings} warning(s)" if rep.warnings else "")
              + " — see the arrows above.")
    elif rep.warnings:
        print(f"\033[33mReady, with {rep.warnings} warning(s).\033[0m")
    else:
        print("\033[32mEverything looks good.\033[0m")
    if not network:
        print("Add --network to also verify your API keys actually work.")
    print()
    return 1 if rep.problems else 0
