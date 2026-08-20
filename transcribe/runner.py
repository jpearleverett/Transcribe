"""Job runner: audio in, diarized transcript on disk."""

from __future__ import annotations

import time
from pathlib import Path

from . import audio as audio_mod, config, jobs as jobs_mod
from .align import diarize_transcript
from .engines import base as engines


def _clean_intermediates(job_id: str) -> None:
    """Remove this job's working files.

    Every intermediate is named `<job id>.<something>`; the original upload is
    named `<timestamp>-<random>-<filename>` and so is never matched.
    """
    try:
        for tmp in config.UPLOAD_DIR.glob(f"{job_id}.*"):
            try:
                tmp.unlink()
            except OSError:
                pass
    except OSError:
        pass


def run_job(job) -> None:
    store = jobs_mod.store()
    cfg = config.load()
    source = Path(job.audio_file)
    if not source.exists():
        raise engines.EngineError("The uploaded audio is missing from disk.")

    engine = engines.get(job.engine or cfg["engine"])
    ok, reason = engine.available()
    if not ok:
        raise engines.EngineError(reason or f"{engine.label} is not available.")

    store.add_log(job.id, f"Engine: {engine.label}")

    # Probe first: duration drives every progress estimate and the UI's ETA.
    store.progress(job.id, "starting", 0.02)
    try:
        info = audio_mod.probe(source)
        if info.get("duration"):
            store.update(job.id, duration=info["duration"])
            store.add_log(job.id, f"Audio: {audio_mod.format_duration(info['duration'])}, "
                                  f"{info.get('sample_rate') or '?'} Hz, "
                                  f"{info.get('channels') or '?'} ch, {info.get('codec') or '?'}")
        else:
            store.add_log(job.id, "Could not read the duration; continuing anyway.")
    except audio_mod.AudioError as e:
        raise engines.EngineError(str(e))

    job = store.get(job.id)
    opts = job.options or {}
    ctx = engines.Context(
        job_id=job.id,
        source=source,
        duration=job.duration or 0.0,
        language=job.language or cfg["language"],
        num_speakers=int(opts.get("num_speakers") or cfg["num_speakers"] or 0),
        min_speakers=int(opts.get("min_speakers") or cfg["min_speakers"] or 0),
        max_speakers=int(opts.get("max_speakers") or cfg["max_speakers"] or 0),
        options=dict(opts),
        progress=lambda stage, frac: store.progress(job.id, stage, frac),
        log=lambda msg: store.add_log(job.id, msg),
        cancelled=lambda: store.is_cancelled(job.id),
        workdir=config.UPLOAD_DIR,
    )

    started = time.time()
    try:
        result = engine.transcribe(ctx)
    finally:
        # Intermediates are cleaned here, not after a successful save. A failed
        # or cancelled job is exactly when they pile up, and on a phone a few
        # abandoned WAV copies of an hour-long recording fills the disk fast.
        _clean_intermediates(job.id)
    ctx.check_cancel()

    if result.is_empty():
        raise engines.EngineError(
            "No speech was found in this recording. If it is very quiet or is "
            "music, try a different file."
        )

    store.progress(job.id, "assembling", 0.95)

    if result.segments and not result.words:
        # Segment-level engine: nothing to align, the speakers are already
        # attached to the segments.
        segments = result.segments
        store.add_log(job.id, "This engine returns speaker segments without word timings.")
    elif result.words:
        segments = diarize_transcript(
            result.words, result.turns,
            max_gap=cfg["max_gap"], max_dur=cfg["max_dur"], max_chars=cfg["max_chars"],
            min_run_words=cfg["min_run_words"], min_run_dur=cfg["min_run_dur"],
        )
    else:
        # An engine that returned only plain text still deserves to be shown.
        from .align import Segment
        segments = [Segment(0.0, job.duration or 0.0, None, result.text.strip(), [])]
        store.add_log(job.id, "This engine returned no word timings, so timestamps are approximate.")

    elapsed = time.time() - started
    speakers = sorted({s.speaker for s in segments if s.speaker is not None}, key=str)
    store.add_log(
        job.id,
        f"Done in {elapsed:.0f}s — {len(segments)} segments, "
        f"{len(result.words)} words, {len(speakers) or 1} speaker(s)",
    )

    store.save_result(job.id, {
        "meta": {
            "engine": engine.name,
            "engine_label": engine.label,
            "model": result.model,
            "language": result.language or job.language,
            "duration": job.duration,
            "words": len(result.words),
            "speakers": len(speakers),
            "elapsed": round(elapsed, 1),
            "created": time.time(),
        },
        "segments": [s.to_dict() for s in segments],
    })

    store.update(
        job.id,
        model=result.model,
        language=result.language or job.language,
        speakers=job.speakers or {},
    )

    if not cfg["keep_audio"]:
        try:
            source.unlink(missing_ok=True)
            store.update(job.id, audio_file="")
        except OSError:
            pass
