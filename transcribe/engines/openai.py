"""OpenAI transcription.

Two honest limitations, both of which shape this adapter:

  * No OpenAI model returns speaker labels *and* word-level timestamps.
    `gpt-4o-transcribe-diarize` gives speaker-labelled segments;
    `timestamp_granularities=["word"]` is whisper-1 only. We use the diarizing
    model and return segments, rather than inventing word timings we do not have.
  * The upload cap is 25 MB — an order of magnitude smaller than the other
    providers. We pick an Opus bitrate from the duration so an hour-plus of
    audio still fits, and say so plainly when it cannot.
"""

from __future__ import annotations

import json

from ..align import Segment
from .. import httpclient as http
from .base import Context, Engine, EngineError, Result, register, normalize_speaker

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
MODEL = "gpt-4o-transcribe-diarize"
MAX_BYTES = 25 * 1024 * 1024
SAFE_BYTES = 23 * 1024 * 1024      # leave room for multipart overhead


@register
class OpenAI(Engine):
    name = "openai"
    label = "OpenAI (speaker segments)"
    description = "Speaker-labelled segments, but no word-level timings, and a 25 MB upload cap."
    signup_url = "https://platform.openai.com/api-keys"
    key_help = "Uses gpt-4o-transcribe-diarize. Timestamps are per segment, not per word."
    speed_factor = 40.0
    supports_speaker_count = False

    def transcribe(self, ctx: Context) -> Result:
        path = _fit_under_cap(ctx)
        size = path.stat().st_size
        if size > MAX_BYTES:
            raise EngineError(
                f"This recording is {size / 1e6:.0f} MB after compression and OpenAI's limit is 25 MB. "
                "Use Deepgram, AssemblyAI or ElevenLabs for long files — they accept multi-gigabyte uploads."
            )

        fields = {
            "model": MODEL,
            "response_format": "diarized_json",
            # Required for anything longer than 30 seconds; omitting it is a
            # known silent failure mode on long audio.
            "chunking_strategy": "auto",
        }
        if ctx.language and ctx.language != "auto":
            fields["language"] = ctx.language

        ctx.log(f"Sending to OpenAI ({MODEL})")
        ctx.progress("uploading", 0.05)

        data = http.upload_multipart(
            ENDPOINT, path,
            field="file", fields=fields,
            headers={"Authorization": f"Bearer {self.key()}"},
            on_progress=lambda sent, total: ctx.progress("uploading", 0.05 + 0.6 * (sent / max(total, 1))),
            should_abort=ctx.cancelled,
            timeout=7200,
        )
        ctx.check_cancel()
        ctx.progress("transcribing", 0.9)

        if not isinstance(data, dict):
            raise EngineError(f"Unexpected response from OpenAI: {str(data)[:400]}")

        raw_segments = data.get("segments") or []
        segments = []
        for s in raw_segments:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            segments.append(Segment(
                start=float(s.get("start") or 0.0),
                end=float(s.get("end") or 0.0),
                speaker=normalize_speaker(s.get("speaker")),
                text=text,
                words=[],
            ))

        text = data.get("text") or " ".join(s.text for s in segments)
        if not segments and not text.strip():
            raise EngineError(f"OpenAI returned an empty transcript: {json.dumps(data)[:300]}")

        return Result(
            segments=segments or None,
            language=ctx.language,
            model=MODEL,
            text=text,
        )


def _fit_under_cap(ctx: Context):
    """Compress hard enough to clear 25 MB, if we can."""
    from .. import audio

    if not audio.have_ffmpeg():
        if ctx.source.stat().st_size > MAX_BYTES:
            raise EngineError(
                "This file is over OpenAI's 25 MB limit and ffmpeg isn't installed to "
                "compress it. Run:  pkg install ffmpeg"
            )
        return ctx.source

    duration = ctx.duration or 0.0
    bitrate = "48k"
    if duration > 0:
        # bits/sec that lands the file at ~SAFE_BYTES, clamped to a range where
        # Opus is still clean for speech.
        target_bps = (SAFE_BYTES * 8) / duration
        kbps = max(12, min(48, int(target_bps / 1000)))
        bitrate = f"{kbps}k"
    ctx.log(f"Compressing to {bitrate} Opus to fit OpenAI's 25 MB cap")
    ctx.progress("converting", 0.0)
    try:
        return audio.to_compressed(ctx.source, ctx.workdir / f"{ctx.job_id}.oai.opus", bitrate=bitrate)
    except audio.AudioError as e:
        raise EngineError(str(e))
