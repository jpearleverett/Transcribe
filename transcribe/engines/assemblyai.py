"""AssemblyAI Universal.

Best published diarization accuracy of the managed APIs, at the cost of an
async flow: upload, submit, then poll. Two details that silently bite:

  * the request field is `speech_models` (plural, an array) — the singular
    form is deprecated and, worse, does not error, it just routes you to the
    default model. We read back `speech_model_used` and log it.
  * timestamps are in MILLISECONDS here and floating-point seconds everywhere
    else.
"""

from __future__ import annotations

import json
import time

from ..align import Word
from .. import httpclient as http
from .base import Context, Engine, EngineError, Result, register, normalize_speaker

BASE = "https://api.assemblyai.com/v2"
MODEL = "universal-3-5-pro"


@register
class AssemblyAI(Engine):
    name = "assemblyai"
    label = "AssemblyAI Universal"
    description = "Highest published diarization accuracy. ~$0.23/hour. $50 of free credit."
    signup_url = "https://www.assemblyai.com/dashboard/signup"
    key_help = "$50 of free credit on signup, no card required."
    speed_factor = 40.0

    def transcribe(self, ctx: Context) -> Result:
        key = self.key()
        headers = {"Authorization": key}
        path = _upload_copy(ctx)
        ctx.check_cancel()

        ctx.log("Uploading to AssemblyAI")
        ctx.progress("uploading", 0.05)
        up = http.upload_raw(
            f"{BASE}/upload", path,
            headers=headers, content_type="application/octet-stream",
            on_progress=lambda sent, total: ctx.progress("uploading", 0.05 + 0.45 * (sent / max(total, 1))),
            should_abort=ctx.cancelled,
            timeout=7200,
        )
        audio_url = (up or {}).get("upload_url")
        if not audio_url:
            raise EngineError(f"AssemblyAI did not accept the upload: {json.dumps(up)[:300]}")

        body = {
            "audio_url": audio_url,
            "speech_models": [MODEL],   # plural array; singular is deprecated
            "speaker_labels": True,
            "punctuate": True,          # required for speaker_labels
            "format_text": True,
        }
        if ctx.language and ctx.language != "auto":
            body["language_code"] = ctx.language
        else:
            body["language_detection"] = True
        if ctx.num_speakers:
            body["speakers_expected"] = ctx.num_speakers

        ctx.check_cancel()
        ctx.progress("transcribing", 0.55)
        job = http.post(f"{BASE}/transcript", headers=headers, json_body=body)
        tid = (job or {}).get("id")
        if not tid:
            raise EngineError(f"AssemblyAI rejected the job: {json.dumps(job)[:400]}")
        ctx.log(f"AssemblyAI job {tid}")

        data = self._poll(ctx, tid, headers)
        used = data.get("speech_model_used") or MODEL
        if used != MODEL:
            ctx.log(f"Note: AssemblyAI ran '{used}', not '{MODEL}'.")

        words = []
        # Prefer utterances: their nested words carry the speaker letter, while
        # the top-level `words` array can come back without one.
        for utt in data.get("utterances") or []:
            spk = normalize_speaker(utt.get("speaker"))
            for w in utt.get("words") or []:
                words.append(_word(w, spk))
        if not words:
            for w in data.get("words") or []:
                words.append(_word(w, normalize_speaker(w.get("speaker"))))

        if not words and not (data.get("text") or "").strip():
            raise EngineError("AssemblyAI returned an empty transcript.")

        return Result(
            words=words,
            language=data.get("language_code") or ctx.language,
            model=used,
            text=data.get("text") or "",
        )

    def _poll(self, ctx: Context, tid: str, headers: dict) -> dict:
        url = f"{BASE}/transcript/{tid}"
        started = time.time()
        budget = max(3600.0, (ctx.duration or 600.0) * 2)
        # AssemblyAI runs at roughly 30-60x realtime. No percentage is exposed,
        # so we estimate from elapsed wall clock against that throughput, and
        # back off the poll interval so a phone radio isn't woken every second
        # for an hour-long file.
        expected = max((ctx.duration or 600.0) / self.speed_factor, 5.0)
        interval = 3.0
        while True:
            ctx.check_cancel()
            elapsed = time.time() - started
            if elapsed > budget:
                raise EngineError("AssemblyAI timed out. The job may still finish — try again later.")

            data = http.get(url, headers=headers, timeout=60)
            status = (data or {}).get("status")
            if status == "completed":
                ctx.progress("transcribing", 0.9)
                return data
            if status == "error":
                raise EngineError(f"AssemblyAI failed: {data.get('error') or 'unknown error'}")

            ctx.progress("transcribing", 0.55 + 0.33 * min(1.0, elapsed / expected))
            _sleep_cancellable(ctx, interval)
            interval = min(interval * 1.3, 15.0)


def _word(w: dict, speaker) -> Word:
    # milliseconds -> seconds
    return Word(
        start=float(w.get("start", 0)) / 1000.0,
        end=float(w.get("end", 0)) / 1000.0,
        text=w.get("text") or "",
        speaker=speaker if speaker is not None else normalize_speaker(w.get("speaker")),
        confidence=w.get("confidence"),
    )


def _sleep_cancellable(ctx: Context, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        ctx.check_cancel()
        time.sleep(min(0.5, max(0.0, end - time.time())))


def _upload_copy(ctx: Context):
    from .. import audio
    if audio.have_ffmpeg():
        try:
            ctx.progress("converting", 0.0)
            return ctx.opus()
        except audio.AudioError as e:
            ctx.log(f"Could not compress audio ({e}); uploading the original.")
    return ctx.source
