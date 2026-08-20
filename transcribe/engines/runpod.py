"""RunPod serverless GPU worker.

This is the accuracy ceiling for this app: a real GPU running large-v3 with
forced alignment and a proper diarizer does an hour of audio in a couple of
minutes, and the audio only ever touches infrastructure you control.

The one architectural constraint worth understanding: a phone has no public
URL, so the audio has to travel *inside* the request as base64, and serverless
request bodies are capped. We therefore pick an Opus bitrate from the recording's
duration so it fits, and fall back to telling you plainly when it cannot. If you
do have somewhere public to host the file, set `runpod_audio_url` and we send a
URL instead, which removes the cap entirely.

The matching worker lives in `gpu/` in this repo.
"""

from __future__ import annotations

import base64
import json
import time

from ..align import Word, Turn, Segment
from .. import httpclient as http
from .base import Context, Engine, EngineError, Result, register, normalize_speaker

# api.runpod.AI is the serverless invoke host. api.runpod.IO is the separate
# management REST API — a very easy and very confusing typo to make.
BASE = "https://api.runpod.ai/v2"
TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMED_OUT"}

# Hard, documented, non-raisable request-body caps on queue-based endpoints.
RUN_LIMIT_MB = 10
RUNSYNC_LIMIT_MB = 20


@register
class RunPod(Engine):
    name = "runpod"
    label = "RunPod GPU (your pod)"
    description = "Your own GPU worker: large-v3 + forced alignment + diarization. Fastest and most accurate."
    signup_url = "https://www.runpod.io/console/serverless"
    key_help = "Needs your RunPod API key and the endpoint ID of the worker deployed from this repo's gpu/ folder."
    speed_factor = 30.0       # ~30x realtime on a mid-range GPU
    config_fields = [
        {"key": "runpod_endpoint", "label": "Endpoint ID",
         "placeholder": "e.g. 4kx9v2abcd1234",
         "help": "From the RunPod Serverless console, after deploying the worker in gpu/."},
        {"key": "runpod_model", "label": "Whisper model",
         "placeholder": "large-v3",
         "help": "Must match a model your worker image can load."},
        {"key": "runpod_audio_url", "label": "Audio URL (optional)",
         "placeholder": "https://…",
         "help": "If you host audio somewhere the worker can reach, it downloads from "
                 "there instead and the 20 MB request cap stops applying."},
    ]

    def available(self) -> tuple:
        from .. import config
        if not config.load().get("runpod_endpoint"):
            return False, ("No RunPod endpoint ID set. Deploy the worker in gpu/ to RunPod "
                           "Serverless, then paste its endpoint ID in Settings.")
        return True, ""

    def transcribe(self, ctx: Context) -> Result:
        from .. import config
        cfg = config.load()
        endpoint = (cfg.get("runpod_endpoint") or "").strip()
        if not endpoint:
            raise EngineError("No RunPod endpoint ID set.")

        headers = {"Authorization": f"Bearer {self.key()}"}

        payload = {
            "language": None if ctx.language in ("", "auto") else ctx.language,
            "model": cfg.get("runpod_model") or "large-v3",
            "diarize": True,
            "align": True,
        }
        if ctx.num_speakers:
            payload["num_speakers"] = ctx.num_speakers
        if ctx.min_speakers:
            payload["min_speakers"] = ctx.min_speakers
        if ctx.max_speakers:
            payload["max_speakers"] = ctx.max_speakers

        # Give the worker room to finish: the default execution timeout is ten
        # minutes, which a multi-hour recording will blow straight through.
        payload_policy = {"executionTimeout": int(max(900, (ctx.duration or 600) * 1.2) * 1000)}

        audio_url = (cfg.get("runpod_audio_url") or "").strip()
        body_size = 0
        temp_audio = None
        if audio_url:
            payload["audio_url"] = audio_url
            ctx.log(f"Pointing the worker at {audio_url}")
        else:
            budget = int(cfg.get("runpod_max_payload_mb") or RUNSYNC_LIMIT_MB)
            path, note = _fit_payload(ctx, min(budget, RUNSYNC_LIMIT_MB))
            if path != ctx.source:
                temp_audio = path
            if note:
                ctx.log(note)
            ctx.progress("uploading", 0.1)
            payload["audio_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            payload["audio_format"] = path.suffix.lstrip(".") or "opus"
            body_size = len(payload["audio_base64"])
            # _fit_payload can only size the bitrate when the duration is known.
            # If the probe failed it compresses at the default and we could still
            # be over, so check the bytes we actually produced.
            if body_size > RUNSYNC_LIMIT_MB * 1024 * 1024 * 0.97:
                raise EngineError(
                    f"This audio is {body_size / 1e6:.0f} MB once encoded, over RunPod's "
                    f"{RUNSYNC_LIMIT_MB} MB request limit. Host it somewhere the worker can "
                    "reach and set 'runpod_audio_url' in Settings, or use a cloud engine for "
                    "this one — Deepgram and ElevenLabs take multi-gigabyte uploads.")

        if temp_audio is not None:
            try:
                temp_audio.unlink(missing_ok=True)   # already base64'd into the payload
            except OSError:
                pass

        ctx.check_cancel()
        ctx.progress("uploading", 0.35)

        envelope = {"input": payload, "policy": payload_policy}

        # /run caps the request body at 10 MB but /runsync allows 20 MB, so a
        # bigger payload goes through /runsync with a short wait — which still
        # returns a job id we can poll, it just accepts twice the audio.
        if body_size > RUN_LIMIT_MB * 1024 * 1024 * 0.95:
            ctx.log(f"Submitting to RunPod endpoint {endpoint} via /runsync ({body_size / 1e6:.1f} MB body)")
            url = f"{BASE}/{endpoint}/runsync?wait=10000"
        else:
            ctx.log(f"Submitting to RunPod endpoint {endpoint}")
            url = f"{BASE}/{endpoint}/run"

        # retries=0: /run enqueues a job, so replaying it after a lost response
        # starts (and bills for) a second GPU run. A failed submit is something
        # the user retries explicitly.
        submit = http.post(url, headers=headers, json_body=envelope, timeout=900, retries=0)
        if not isinstance(submit, dict):
            raise EngineError(f"RunPod did not accept the job: {str(submit)[:400]}")

        # /runsync may have finished inside the wait window.
        if submit.get("status") in TERMINAL_OK and submit.get("output") is not None:
            ctx.progress("transcribing", 0.92)
            return _parse_output(submit["output"], ctx, payload["model"])
        if submit.get("status") in TERMINAL_BAD:
            raise EngineError(f"RunPod job {submit['status'].lower()}: "
                              f"{json.dumps(submit.get('error') or submit.get('output'))[:500]}")

        job_id = submit.get("id")
        if not job_id:
            raise EngineError(f"RunPod did not return a job id: {json.dumps(submit)[:400]}")
        ctx.log(f"RunPod job {job_id}")

        output = self._poll(ctx, endpoint, job_id, headers)
        return _parse_output(output, ctx, payload["model"])

    def _poll(self, ctx: Context, endpoint: str, job_id: str, headers: dict) -> dict:
        url = f"{BASE}/{endpoint}/status/{job_id}"
        started = time.time()
        expected = max((ctx.duration or 600.0) / self.speed_factor, 20.0)
        # Generous: a cold start pulling a multi-GB image can take minutes
        # before any audio is even processed.
        budget = max(1800.0, (ctx.duration or 600.0) * 1.5)
        interval = 2.0
        finished = False

        try:
            while True:
                ctx.check_cancel()

                elapsed = time.time() - started
                if elapsed > budget:
                    raise EngineError(
                        f"RunPod job {job_id} did not finish within {budget / 60:.0f} minutes. "
                        "Check the endpoint's logs in the RunPod console."
                    )

                data = http.get(url, headers=headers, timeout=60)
                status = (data or {}).get("status", "")

                if status in TERMINAL_OK:
                    finished = True
                    out = data.get("output")
                    if out is None:
                        raise EngineError("RunPod finished but returned no output.")
                    ctx.progress("transcribing", 0.92)
                    return out
                if status in TERMINAL_BAD:
                    finished = True
                    detail = data.get("error") or data.get("output") or status
                    raise EngineError(f"RunPod job {status.lower()}: {json.dumps(detail)[:500]}")

                stage = "transcribing" if status == "IN_PROGRESS" else "polling"
                ctx.progress(stage, 0.4 + 0.5 * min(1.0, elapsed / expected))
                _sleep_cancellable(ctx, interval)
                interval = min(interval * 1.25, 10.0)
        finally:
            # Cancelling has to happen here. The poll loop spends nearly all its
            # time inside _sleep_cancellable and a 60 s http.get, both of which
            # raise or return straight out of the loop — so a cancel issued at
            # the top of the loop was effectively never reached, and the GPU
            # kept running (and billing) after the user hit Cancel.
            if not finished:
                try:
                    http.post(f"{BASE}/{endpoint}/cancel/{job_id}", headers=headers,
                              json_body={}, timeout=30, retries=0)
                    ctx.log("Asked RunPod to cancel the job")
                except Exception:            # noqa: BLE001 - best effort
                    pass


def _parse_output(output, ctx: Context, model: str) -> Result:
    """Accept our own worker's shape, and the common WhisperX shape too."""
    if isinstance(output, list) and output:
        output = output[0]
    if not isinstance(output, dict):
        raise EngineError(f"Unexpected RunPod output: {str(output)[:400]}")
    if output.get("error"):
        raise EngineError(f"The RunPod worker reported: {output['error']}")

    language = output.get("language") or ctx.language
    model_used = output.get("model") or model

    words = []
    # Preferred: a flat word list, each word already carrying its speaker.
    for w in output.get("words") or []:
        text = (w.get("word") or w.get("text") or "").strip()
        start, end = _f(w.get("start")), _f(w.get("end"))
        if not text or start is None:
            continue
        words.append(Word(start, end if end is not None and end > start else start + 0.05,
                          text, normalize_speaker(w.get("speaker")), _f(w.get("score") or w.get("confidence"))))

    segments_out = None
    if not words:
        # WhisperX shape: segments[], each with its own words[].
        for s in output.get("segments") or []:
            seg_spk = normalize_speaker(s.get("speaker"))
            for w in s.get("words") or []:
                text = (w.get("word") or w.get("text") or "").strip()
                start, end = _f(w.get("start")), _f(w.get("end"))
                if not text or start is None:
                    continue
                words.append(Word(start, end if end is not None and end > start else start + 0.05,
                                  text, normalize_speaker(w.get("speaker")) or seg_spk,
                                  _f(w.get("score") or w.get("confidence"))))
        if not words:
            # Segment-level only.
            segs = []
            for s in output.get("segments") or []:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                segs.append(Segment(_f(s.get("start")) or 0.0, _f(s.get("end")) or 0.0,
                                    normalize_speaker(s.get("speaker")), text, []))
            segments_out = segs or None

    turns = None
    for t in output.get("turns") or output.get("diarization") or []:
        start, end = _f(t.get("start")), _f(t.get("end"))
        spk = normalize_speaker(t.get("speaker"))
        if start is None or end is None or spk is None:
            continue
        turns = turns or []
        turns.append(Turn(start, end, spk))

    text = output.get("text") or " ".join(w.text for w in words)
    if not words and not segments_out and not text.strip():
        raise EngineError("The RunPod worker returned an empty transcript.")

    return Result(words=words, turns=turns, segments=segments_out,
                  language=language, model=model_used, text=text)


def _fit_payload(ctx: Context, budget_mb: int):
    """Compress so base64 of the result stays under the request-size budget.

    base64 inflates by 4/3; the 0.72 factor leaves the remainder for the JSON
    envelope. Roughly: an hour fits comfortably, three hours only just.
    """
    from .. import audio

    raw_budget = int(budget_mb * 1024 * 1024 * 0.72)

    if not audio.have_ffmpeg():
        size = ctx.source.stat().st_size
        if size > raw_budget:
            raise EngineError(
                f"This file is {size / 1e6:.0f} MB and needs compressing to fit in a RunPod "
                "request, but ffmpeg isn't installed. Run:  pkg install ffmpeg"
            )
        return ctx.source, ""

    duration = ctx.duration or 0.0
    bitrate, note = "48k", ""
    if duration > 0:
        kbps = int((raw_budget * 8) / duration / 1000)
        if kbps < 10:
            raise EngineError(
                f"This recording is {duration / 3600:.1f} hours long — too much audio to fit "
                f"inside a {budget_mb} MB RunPod request. Either raise "
                "the file somewhere your worker can reach and set 'runpod_audio_url' in "
                "Settings, or use a cloud engine for this one — Deepgram and ElevenLabs take "
                "multi-gigabyte uploads."
            )
        capped = max(10, min(48, kbps))
        bitrate = f"{capped}k"
        if capped < 32:
            note = (f"Compressing to {bitrate} Opus so it fits in the RunPod request "
                    "(fine for speech, but lower fidelity than the original).")
    ctx.progress("converting", 0.0)
    try:
        return audio.to_compressed(ctx.source, ctx.workdir / f"{ctx.job_id}.rp.opus",
                                   bitrate=bitrate), note
    except audio.AudioError as e:
        raise EngineError(str(e))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sleep_cancellable(ctx: Context, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        ctx.check_cancel()
        time.sleep(min(0.5, max(0.0, end - time.time())))
