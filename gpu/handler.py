"""RunPod serverless handler: audio in, word-level diarized transcript out.

Stack, and why:

  faster-whisper (CTranslate2) large-v3 with word_timestamps=True
      → the transcript and per-word timings
  pyannote speaker-diarization-community-1
      → the speaker timeline

We deliberately do *not* depend on WhisperX. It is a thin wrapper over exactly
these two libraries, it pins an exact torch version that fights everything else
in the image, and 3.8.3-3.8.6 carry an open alignment regression. Calling the
two libraries directly is less code and fewer ways to break.

Word→speaker assignment happens back on the phone, in align.py, so the same
attribution logic applies to every engine.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import runpod

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
DIARIZER_NAME = os.environ.get("DIARIZER_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")

# Loaded once per container and reused across jobs: on a warm worker this is the
# difference between a 5-second job and a 90-second one.
_whisper = None
_batched = None
_whisper_name = None
_diarizer = None


def log(msg):
    print(f"[handler] {msg}", flush=True)


def get_whisper(name=None):
    """Load (and cache) a model. Keyed by name so a per-request model works."""
    global _whisper, _batched, _whisper_name
    name = name or MODEL_NAME
    if _whisper is None or name != _whisper_name:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        t = time.time()
        log(f"loading {name} ({COMPUTE_TYPE}) on {DEVICE}")
        _whisper = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
        _whisper_name = name
        try:
            _batched = BatchedInferencePipeline(model=_whisper)
        except Exception as e:                      # noqa: BLE001
            log(f"batched pipeline unavailable ({e}); using sequential decoding")
            _batched = None
        log(f"whisper loaded in {time.time() - t:.1f}s")
    return _whisper, _batched


def check_audio_backend():
    """Fail loudly if torchcodec cannot load its FFmpeg backend.

    pyannote decodes through torchcodec, which only warns when its backend is
    missing — so the first sign is speaker labels quietly disappearing. Better
    to say so on the job than to return a one-speaker transcript.
    """
    try:
        from torchcodec.decoders import AudioDecoder      # noqa: F401
        return True
    except Exception as e:                                # noqa: BLE001
        log(f"WARNING: torchcodec unavailable ({e}); diarization will likely fail. "
            "The image is missing libpython3.10.")
        return False


def get_diarizer():
    global _diarizer
    if _diarizer is None:
        import torch
        from pyannote.audio import Pipeline
        check_audio_backend()
        t = time.time()
        log(f"loading diarizer {DIARIZER_NAME}")
        _diarizer = Pipeline.from_pretrained(DIARIZER_NAME, token=HF_TOKEN or None)
        if _diarizer is None:
            raise RuntimeError(
                f"Could not load {DIARIZER_NAME}. This model is gated on Hugging Face: "
                "accept its conditions on the model page and set HF_TOKEN on the endpoint."
            )
        if DEVICE == "cuda" and torch.cuda.is_available():
            _diarizer.to(torch.device("cuda"))
        log(f"diarizer loaded in {time.time() - t:.1f}s")
    return _diarizer


def fetch_audio(job_input, workdir: Path) -> Path:
    """Materialise the input audio as a 16 kHz mono WAV."""
    raw = workdir / "input.audio"

    if job_input.get("audio_base64"):
        data = job_input["audio_base64"]
        # Tolerate a data: URL, which is easy to send by accident.
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        raw.write_bytes(base64.b64decode(data))
    elif job_input.get("audio_url") or job_input.get("audio"):
        import urllib.request
        url = job_input.get("audio_url") or job_input.get("audio")
        log(f"downloading {url[:120]}")
        with urllib.request.urlopen(url, timeout=600) as resp, open(raw, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    else:
        raise ValueError("No audio supplied. Send 'audio_base64' or 'audio_url'.")

    if raw.stat().st_size == 0:
        raise ValueError("The supplied audio was empty.")

    wav = workdir / "audio.wav"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(raw), "-vn", "-sn", "-dn",
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav.exists():
        raise ValueError(f"ffmpeg could not decode the audio: {proc.stderr.strip()[:400]}")
    raw.unlink(missing_ok=True)
    return wav


def transcribe(wav: Path, job_input: dict) -> dict:
    requested = (job_input.get("model") or "").strip() or None
    try:
        model, batched = get_whisper(requested)
    except Exception as e:                          # noqa: BLE001
        if not requested or requested == MODEL_NAME:
            raise
        # A model this image cannot load should not fail the whole job when the
        # baked-in default would do.
        log(f"could not load requested model {requested!r} ({e}); using {MODEL_NAME}")
        model, batched = get_whisper(MODEL_NAME)
    language = job_input.get("language") or None
    beam_size = int(job_input.get("beam_size") or 5)

    kwargs = dict(
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        # Whisper's classic failure mode is looping on silence or music. These
        # three thresholds are the documented mitigations.
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    t = time.time()
    if batched is not None:
        segments, info = batched.transcribe(str(wav), batch_size=int(job_input.get("batch_size") or 16), **kwargs)
    else:
        segments, info = model.transcribe(str(wav), **kwargs)

    words, texts = [], []
    for seg in segments:
        texts.append(seg.text)
        for w in (seg.words or []):
            token = (w.word or "").strip()
            if not token:
                continue
            words.append({
                "word": token,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "score": round(float(w.probability), 4) if w.probability is not None else None,
            })
    log(f"transcribed {len(words)} words in {time.time() - t:.1f}s")
    return {
        "words": words,
        "text": "".join(texts).strip(),
        "language": getattr(info, "language", "") or (language or ""),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "model": _whisper_name or MODEL_NAME,
    }


def diarize(wav: Path, job_input: dict) -> list:
    pipeline = get_diarizer()
    kwargs = {}
    if job_input.get("num_speakers"):
        kwargs["num_speakers"] = int(job_input["num_speakers"])
    else:
        if job_input.get("min_speakers"):
            kwargs["min_speakers"] = int(job_input["min_speakers"])
        if job_input.get("max_speakers"):
            kwargs["max_speakers"] = int(job_input["max_speakers"])

    t = time.time()
    annotation = pipeline(str(wav), **kwargs)
    turns = [
        {"start": round(float(segment.start), 3),
         "end": round(float(segment.end), 3),
         "speaker": str(label)}
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t_: (t_["start"], t_["end"]))
    log(f"diarized into {len({t_['speaker'] for t_ in turns})} speakers "
        f"across {len(turns)} turns in {time.time() - t:.1f}s")
    return turns


def handler(job):
    started = time.time()
    job_input = job.get("input") or {}
    workdir = Path(tempfile.mkdtemp(prefix="job-"))
    try:
        wav = fetch_audio(job_input, workdir)
        result = transcribe(wav, job_input)

        if job_input.get("diarize", True):
            try:
                result["turns"] = diarize(wav, job_input)
            except Exception as e:                  # noqa: BLE001
                # Losing speaker labels beats losing the transcript.
                traceback.print_exc()
                result["turns"] = []
                result["diarization_error"] = str(e)[:500]

        result.setdefault("model", MODEL_NAME)
        result["elapsed"] = round(time.time() - started, 1)
        return result

    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        for f in workdir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            workdir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        # Sanity check the wiring without RunPod, e.g. inside the built image.
        path = sys.argv[sys.argv.index("--selftest") + 1]
        out = handler({"input": {"audio_base64": base64.b64encode(Path(path).read_bytes()).decode()}})
        print(json.dumps(out, indent=2)[:4000])
    else:
        runpod.serverless.start({"handler": handler})
