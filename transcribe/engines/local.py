"""Fully offline transcription on the phone itself.

whisper.cpp for ASR, sherpa-onnx for diarization. Both are native ARM64 binaries
built by `install.sh`; nothing here needs PyTorch, which does not exist for
Termux.

A note on whisper.cpp's own diarization flags, because they look like the
obvious answer and are not:

  * `-di/--diarize` is *stereo* diarization — it compares left/right channel
    energy. On a single-mic phone recording it does nothing at all.
  * `-tdrz/--tinydiarize` needs a model that has not been rebuilt since 2023,
    is English-only, and emits turn *boundaries* rather than speaker
    identities.

So speaker identity comes from sherpa-onnx, and the two timelines are merged by
the shared word-attribution code in align.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..align import Word, Turn
from .base import Context, Engine, EngineError, Result, register

TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)%")


@register
class Local(Engine):
    name = "local"
    label = "On this phone (offline)"
    description = "Runs entirely on the device. No API key, no data leaves the phone — but slow."
    needs_key = False
    speed_factor = 2.0     # roughly: an hour of audio takes half an hour
    config_fields = [
        {"key": "local_model", "label": "Model file",
         "placeholder": "ggml-small.en-q5_1.bin",
         "help": "A file in ~/.transcribe/models. Smaller models are much faster and less accurate."},
        {"key": "local_threads", "label": "Threads (0 = auto)", "type": "number",
         "help": "Fewer threads run cooler and slower; a hot phone throttles anyway."},
    ]

    # -------------------- discovery --------------------

    def whisper_bin(self) -> str:
        from .. import config
        explicit = os.environ.get("WHISPER_CLI")
        if explicit and Path(explicit).exists():
            return explicit
        local = config.BIN_DIR / "whisper-cli"
        if local.exists():
            return str(local)
        for name in ("whisper-cli", "whisper-cpp", "main"):
            found = shutil.which(name)
            if found:
                return found
        return ""

    def model_path(self) -> Path:
        from .. import config
        cfg = config.load()
        name = cfg.get("local_model") or "ggml-small.en-q5_1.bin"
        p = config.MODEL_DIR / name
        if p.exists():
            return p
        # Fall back to whatever ggml model is actually present, so a partial
        # install still works instead of erroring on a filename mismatch.
        candidates = sorted(config.MODEL_DIR.glob("ggml-*.bin"), key=lambda f: -f.stat().st_size)
        return candidates[0] if candidates else p

    def diarizer_models(self) -> tuple:
        from .. import config
        seg = emb = None
        for p in config.MODEL_DIR.glob("*.onnx"):
            n = p.name.lower()
            if "segmentation" in n or "diarization" in n:
                # Prefer the int8 build: a quarter of the size, negligible
                # accuracy cost, and much faster on a phone.
                if seg is None or ("int8" in n and "int8" not in seg.name.lower()):
                    seg = p
            elif any(k in n for k in ("eres2net", "titanet", "speaker", "embedding", "wespeaker", "3dspeaker")):
                if emb is None or ("int8" in n and "int8" not in emb.name.lower()):
                    emb = p
        return seg, emb

    def available(self) -> tuple:
        if not self.whisper_bin():
            return False, ("whisper.cpp isn't installed yet. Run  ./install.sh --local  in Termux "
                           "(it takes a while — it compiles from source).")
        model = self.model_path()
        if not model.exists():
            return False, (f"No speech model found in {model.parent}. Run  ./install.sh --local  "
                           "to download one.")
        return True, ""

    # -------------------- run --------------------

    def transcribe(self, ctx: Context) -> Result:
        from .. import config
        cfg = config.load()

        ok, reason = self.available()
        if not ok:
            raise EngineError(reason)

        wav = ctx.wav16k()
        ctx.check_cancel()

        words, detected = self._run_whisper(ctx, wav, cfg)
        ctx.check_cancel()

        turns = None
        if cfg.get("local_diarize", True):
            turns = self._run_diarizer(ctx, wav, cfg)

        return Result(
            words=words,
            turns=turns,
            language=detected or (ctx.language if ctx.language != "auto" else ""),
            model=self.model_path().name,
            text=" ".join(w.text for w in words),
        )

    def _run_whisper(self, ctx: Context, wav: Path, cfg: dict) -> tuple:
        model = self.model_path()
        threads = int(cfg.get("local_threads") or 0) or max(2, (os.cpu_count() or 4) - 1)
        out_prefix = ctx.workdir / f"{ctx.job_id}.whisper"

        cmd = [
            self.whisper_bin(),
            "-m", str(model),
            "-f", str(wav),
            "-t", str(threads),
            "-oj",                    # JSON output next to -of
            "-of", str(out_prefix),
            "-ml", "1",               # one word per segment => word timestamps
            "-sow",                   # split on word boundaries, not tokens
            "-pp",                    # print progress, which we parse below
        ]
        if ctx.language and ctx.language != "auto":
            cmd += ["-l", ctx.language]
        dtw = _dtw_preset(model.name)
        if dtw:
            # True DTW token timestamps rather than the coarse heuristic ones.
            cmd += ["-dtw", dtw]

        ctx.log(f"whisper.cpp: {model.name}, {threads} threads")
        ctx.progress("transcribing", 0.0)

        code, tail = self._spawn(ctx, cmd)
        if code != 0 and dtw and "DTW" in "".join(tail).upper():
            # Wrong alignment-heads preset for this model: drop the flag and
            # take the heuristic timestamps rather than losing the transcript.
            ctx.log("This model has no DTW alignment preset; using standard timestamps.")
            cmd = [c for c in cmd if c not in ("-dtw", dtw)]
            code, tail = self._spawn(ctx, cmd)

        ctx.check_cancel()
        if code != 0:
            raise EngineError("whisper.cpp failed:\n" + "".join(tail[-15:]).strip()[:800])

        out_json = Path(str(out_prefix) + ".json")
        if not out_json.exists():
            raise EngineError("whisper.cpp produced no JSON output. "
                              "Is this build too old to support -oj?")
        try:
            data = json.loads(out_json.read_text())
        except json.JSONDecodeError as e:
            raise EngineError(f"Could not read whisper.cpp output: {e}")
        finally:
            out_json.unlink(missing_ok=True)

        words = _parse_whisper_json(data)
        if not words:
            raise EngineError("whisper.cpp found no speech in this recording.")
        detected = ((data.get("result") or {}).get("language") or "").strip()
        if detected == "auto":
            detected = ""
        ctx.log(f"whisper.cpp: {len(words)} words"
                + (f", language {detected}" if detected else ""))
        # Returned rather than stashed on self: base.register instantiates each
        # engine exactly once, so anything written to the instance leaks into
        # every later job — a German recording would relabel the next one.
        return words, detected

    def _spawn(self, ctx: Context, cmd: list) -> tuple:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        tail = []
        try:
            for line in proc.stdout:
                tail.append(line)
                del tail[:-40]
                m = PROGRESS_RE.search(line)
                if m:
                    # whisper.cpp is the bulk of the work; diarization is the rest.
                    ctx.progress("transcribing", 0.05 + 0.7 * (int(m.group(1)) / 100.0))
                if ctx.cancelled():
                    proc.terminate()
                    raise EngineError("cancelled")
            proc.wait()
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
        return proc.returncode, tail

    def _run_diarizer(self, ctx: Context, wav: Path, cfg: dict) -> list:
        seg, emb = self.diarizer_models()
        if not seg or not emb:
            ctx.log("Speaker models aren't installed, so everything is attributed to one "
                    "speaker. Run ./install.sh --local to add them.")
            return None

        script = TOOLS / "diarize_sherpa.py"
        if not script.exists():
            ctx.log("Diarization helper is missing; skipping speaker labels.")
            return None

        out_file = ctx.workdir / f"{ctx.job_id}.turns.json"
        cmd = [
            sys.executable, str(script),
            "--wav", str(wav),
            "--segmentation", str(seg),
            "--embedding", str(emb),
            "--out", str(out_file),
            "--threads", str(int(cfg.get("local_threads") or 0) or max(2, (os.cpu_count() or 4) - 1)),
        ]
        if ctx.num_speakers:
            cmd += ["--num-speakers", str(ctx.num_speakers)]

        ctx.log(f"Diarizing with {seg.name}")
        ctx.progress("diarizing", 0.78)

        # The helper writes its JSON to a file and keeps stderr for progress, so
        # we only ever drain one pipe — draining two without threads deadlocks
        # as soon as either fills.
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
        tail = []
        try:
            for line in proc.stderr:
                tail.append(line)
                del tail[:-25]
                m = PROGRESS_RE.search(line)
                if m:
                    ctx.progress("diarizing", 0.78 + 0.14 * (int(m.group(1)) / 100.0))
                if ctx.cancelled():
                    proc.terminate()
                    raise EngineError("cancelled")
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            if proc.stderr:
                proc.stderr.close()
            if proc.poll() is None:
                proc.kill()

        ctx.check_cancel()
        if proc.returncode != 0:
            # A failed diarizer should cost you speaker labels, not the whole
            # transcript you just waited half an hour for.
            ctx.log("Diarization failed, so the transcript has no speaker labels: "
                    + "".join(tail).strip()[-400:])
            out_file.unlink(missing_ok=True)
            return None

        try:
            payload = json.loads(out_file.read_text() or "{}")
        except (json.JSONDecodeError, OSError):
            ctx.log("Diarization returned unreadable output; skipping speaker labels.")
            return None
        finally:
            out_file.unlink(missing_ok=True)

        turns = []
        for t in payload.get("turns") or []:
            try:
                turns.append(Turn(float(t["start"]), float(t["end"]), str(t["speaker"])))
            except (KeyError, TypeError, ValueError):
                continue
        if not turns:
            ctx.log("Diarization found no speaker turns.")
            return None

        ctx.log(f"Found {len({t.speaker for t in turns})} speaker(s) across {len(turns)} turns")
        ctx.progress("aligning", 0.9)
        return turns


def _parse_whisper_json(data: dict) -> list:
    """Read whisper.cpp's -oj output into Words.

    Its offsets are integer milliseconds; the `timestamps` strings are for
    humans. With `-ml 1` each entry holds a single word.
    """
    words = []
    for seg in data.get("transcription") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        offsets = seg.get("offsets") or {}
        start = offsets.get("from")
        end = offsets.get("to")
        if start is None:
            continue
        start_s = float(start) / 1000.0
        end_s = float(end) / 1000.0 if end is not None else start_s + 0.05
        if end_s <= start_s:
            end_s = start_s + 0.05
        # Skip whisper's non-speech annotations rather than reading them aloud.
        if text.startswith("[") and text.endswith("]"):
            continue
        if text.startswith("(") and text.endswith(")"):
            continue
        words.append(Word(start=start_s, end=end_s, text=text))
    return words


def _dtw_preset(model_filename: str) -> str:
    """Map a ggml filename to a valid -dtw preset, or "" if there isn't one.

    Passing an unrecognised preset makes whisper-cli exit outright, so anything
    we are not sure about returns empty.
    """
    n = model_filename.lower()
    if "distil" in n or "tdrz" in n:
        return ""       # different architecture; the presets do not apply
    for needle, preset in (
        ("large-v3-turbo", "large.v3.turbo"),
        ("large-v3", "large.v3"),
        ("large-v2", "large.v2"),
        ("large-v1", "large.v1"),
        ("medium.en", "medium.en"),
        ("medium", "medium"),
        ("small.en", "small.en"),
        ("small", "small"),
        ("base.en", "base.en"),
        ("base", "base"),
        ("tiny.en", "tiny.en"),
        ("tiny", "tiny"),
    ):
        if needle in n:
            return preset
    return ""
