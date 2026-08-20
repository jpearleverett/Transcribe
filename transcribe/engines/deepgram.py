"""Deepgram Nova-3.

Cheapest of the good options (~$0.26/hour, diarization included) and a single
synchronous POST with the audio as the raw body.

The one thing that matters here is the diarization parameter, and it has a trap
at both ends:

  * `diarize=true` is deprecated. It still returns 200 on batch, but always
    routes to the *v1* diarizer, so code copied from a 2025 tutorial quietly
    gets materially worse speaker labels.
  * `diarize_model` replaces it and, per Deepgram's docs, "both enables
    diarization and selects the model version" — it is not merely a selector.
  * Setting **both** is rejected outright: "requests that set both are
    rejected". So the tempting belt-and-braces approach 400s every request.

So: `diarize_model=latest` alone. We also check the result — if not a single
word comes back with a speaker we say so in the job log, rather than quietly
presenting a one-speaker transcript.
"""

from __future__ import annotations

import json
import urllib.parse

from ..align import Word
from .. import httpclient as http
from .base import Context, Engine, EngineError, Result, register, normalize_speaker

ENDPOINT = "https://api.deepgram.com/v1/listen"
MODEL = "nova-3"


@register
class Deepgram(Engine):
    name = "deepgram"
    label = "Deepgram Nova-3"
    description = "Fast and cheap (~$0.26/hour, diarization free). $200 of free credit, no card."
    signup_url = "https://console.deepgram.com/signup"
    key_help = "Deepgram gives $200 of free credit on signup — enough for hundreds of hours."
    speed_factor = 90.0
    # Deepgram's diarizer takes no speaker-count hint of any kind — there is no
    # num_speakers, min/max, or equivalent parameter on /v1/listen.
    supports_speaker_count = False

    def transcribe(self, ctx: Context) -> Result:
        path = _upload_copy(ctx)
        ctx.check_cancel()

        params = {
            "model": MODEL,
            # diarize_model alone: it enables diarization as well as selecting
            # v2, and sending the deprecated `diarize` alongside it is rejected.
            "diarize_model": "latest",
            "punctuate": "true",
            "smart_format": "true",
            "utterances": "true",
            "filler_words": "false",
        }
        if ctx.language and ctx.language != "auto":
            params["language"] = ctx.language
        else:
            params["detect_language"] = "true"

        url = ENDPOINT + "?" + urllib.parse.urlencode(params)
        ctx.log(f"Sending to Deepgram ({MODEL}, v2 diarizer)")
        ctx.progress("uploading", 0.05)

        data = http.upload_raw(
            url, path,
            headers={"Authorization": f"Token {self.key()}"},
            on_progress=lambda sent, total: ctx.progress("uploading", 0.05 + 0.55 * (sent / max(total, 1))),
            should_abort=ctx.cancelled,
            timeout=7200,
        )
        ctx.check_cancel()
        ctx.progress("transcribing", 0.85)

        try:
            channel = data["results"]["channels"][0]
            alt = channel["alternatives"][0]
        except (KeyError, IndexError, TypeError):
            raise EngineError(f"Unexpected response from Deepgram: {json.dumps(data)[:400]}")

        words = []
        for w in alt.get("words") or []:
            text = w.get("punctuated_word") or w.get("word") or ""
            if not text:
                continue
            words.append(Word(
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                text=text,
                speaker=normalize_speaker(w.get("speaker")),
                confidence=w.get("confidence"),
                # Reported per word on pre-recorded audio, and distinct from
                # `confidence`: this is how sure the diarizer is of the
                # *speaker*, which is what attribution errors turn on.
                speaker_confidence=w.get("speaker_confidence"),
            ))

        if not words and not (alt.get("transcript") or "").strip():
            raise EngineError("Deepgram returned an empty transcript.")

        if words and not any(w.speaker is not None for w in words):
            ctx.log("Deepgram returned no speaker labels for this recording — "
                    "the whole transcript will show as one speaker.")

        scored = [w for w in words if w.speaker_confidence is not None]
        if scored:
            shaky = sum(1 for w in scored if w.speaker_confidence < 0.5)
            ctx.log(f"Diarizer was unsure of the speaker on {shaky} of {len(scored)} words"
                    f" ({shaky / len(scored):.0%}); those are re-decided from their neighbours.")

        detected = ""
        try:
            detected = channel.get("detected_language") or data["results"].get("language") or ""
        except (KeyError, AttributeError):
            pass

        return Result(
            words=words,
            language=detected or ctx.language,
            model=MODEL,
            text=alt.get("transcript") or "",
        )


def _upload_copy(ctx: Context):
    from .. import audio
    if audio.have_ffmpeg():
        try:
            ctx.progress("converting", 0.0)
            return ctx.opus()
        except audio.AudioError as e:
            ctx.log(f"Could not compress audio ({e}); uploading the original.")
    return ctx.source
