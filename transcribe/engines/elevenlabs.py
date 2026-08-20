"""ElevenLabs Scribe v2.

The cleanest of the cloud APIs for this app's needs: one multipart POST, no job
polling, and `speaker_id` sits directly on every word object, so we never have
to align a separate diarization timeline against the transcript.
"""

from __future__ import annotations

import json

from ..align import Word
from .. import httpclient as http
from .base import Context, Engine, EngineError, Result, register, normalize_speaker

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL = "scribe_v2"


@register
class ElevenLabs(Engine):
    name = "elevenlabs"
    label = "ElevenLabs Scribe v2"
    description = "Word-level speakers in a single request. ~$0.22/hour. Handles files up to 5 GB."
    signup_url = "https://elevenlabs.io/app/settings/api-keys"
    key_help = "Free tier includes a few hours of transcription per month."
    speed_factor = 60.0

    def transcribe(self, ctx: Context) -> Result:
        path = _upload_copy(ctx)
        ctx.check_cancel()

        fields = {
            "model_id": MODEL,
            "diarize": "true",
            "timestamps_granularity": "word",
            "tag_audio_events": "false",
        }
        if ctx.language and ctx.language != "auto":
            fields["language_code"] = ctx.language
        if ctx.num_speakers:
            fields["num_speakers"] = str(ctx.num_speakers)

        ctx.log(f"Sending to ElevenLabs ({MODEL})")
        ctx.progress("uploading", 0.05)

        data = http.upload_multipart(
            ENDPOINT, path,
            field="file", fields=fields,
            headers={"xi-api-key": self.key()},
            on_progress=lambda sent, total: ctx.progress("uploading", 0.05 + 0.55 * (sent / max(total, 1))),
            should_abort=ctx.cancelled,
            timeout=7200,
        )
        ctx.check_cancel()
        ctx.progress("transcribing", 0.85)

        if not isinstance(data, dict) or "words" not in data:
            raise EngineError(f"Unexpected response from ElevenLabs: {json.dumps(data)[:400]}")

        words = []
        for w in data.get("words") or []:
            # Scribe interleaves "spacing" and "audio_event" entries with the
            # real words; including them would double every space in the output.
            if w.get("type") not in (None, "word"):
                continue
            text = (w.get("text") or "").strip()
            if not text:
                continue
            start = _f(w.get("start"))
            end = _f(w.get("end"))
            if start is None:
                continue
            words.append(Word(
                start=start,
                end=end if end is not None and end > start else start + 0.05,
                text=text,
                speaker=normalize_speaker(w.get("speaker_id")),
                confidence=_logprob_to_conf(w.get("logprob")),
            ))

        if not words and not (data.get("text") or "").strip():
            raise EngineError("ElevenLabs returned an empty transcript.")

        return Result(
            words=words,
            language=data.get("language_code") or ctx.language,
            model=MODEL,
            text=data.get("text") or "",
        )


def _upload_copy(ctx: Context):
    """Prefer a small Opus copy; fall back to the original if ffmpeg is absent."""
    from .. import audio
    if audio.have_ffmpeg():
        try:
            ctx.progress("converting", 0.0)
            return ctx.opus()
        except audio.AudioError as e:
            ctx.log(f"Could not compress audio ({e}); uploading the original.")
    return ctx.source


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _logprob_to_conf(lp):
    if lp is None:
        return None
    try:
        import math
        return round(math.exp(float(lp)), 4)
    except (TypeError, ValueError, OverflowError):
        return None
