"""Job runner: audio in, diarized transcript on disk."""

from __future__ import annotations

import time
from pathlib import Path

from . import audio as audio_mod, config, jobs as jobs_mod, video as video_mod
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
    if job.kind == "extract":
        return run_extract(job)
    if job.kind == "compress":
        return run_compress(job)
    return run_transcription(job)


def run_extract(job) -> None:
    """Pull the audio track out of a local video, leaving the video alone."""
    from . import files as files_mod

    store = jobs_mod.store()
    cfg = config.load()
    source = Path(job.audio_file)
    if not source.exists():
        raise engines.EngineError("That video is no longer there.")

    store.progress(job.id, "converting", 0.01)
    info = audio_mod.probe(source)
    duration = info.get("duration") or 0.0
    if duration:
        store.update(job.id, duration=duration)

    mode = (job.options or {}).get("extract_mode") or cfg["extract_mode"]
    ext, _args, _ = audio_mod.plan_extract(source, mode)
    out_dir = files_mod.output_dir()
    dest = _unique_path(out_dir / (source.stem + ext))

    # Size the output from the audio track, not the container — the container
    # rate includes the video, which on a phone recording is over a hundred
    # times larger and would refuse the job for space it never needed.
    needed = audio_mod.estimate_extract_bytes(info, mode, duration) + (32 << 20)
    free = files_mod.free_bytes(out_dir)
    if free and free < needed:
        raise engines.EngineError(
            f"Not enough space in {out_dir}: about {needed / 1e6:.0f} MB needed for "
            f"the audio, {free / 1e6:.0f} MB free. Free some space, or pick the "
            "'Small — Opus' output which is a fraction of the size.")

    store.add_log(job.id, f"Source: {audio_mod.format_duration(duration)}, "
                          f"{source.stat().st_size / 1e9:.1f} GB, "
                          f"{info.get('codec') or 'unknown'} audio")
    store.add_log(job.id, f"Mode: {mode}"
                          + (" (no re-encoding)" if mode == "copy" else ""))
    store.add_log(job.id, f"Writing {dest}")

    started = time.time()
    try:
        audio_mod.extract_audio(
            source, dest, mode, duration=duration,
            on_progress=lambda f: store.progress(job.id, "converting", 0.01 + 0.98 * f),
            should_abort=lambda: store.is_cancelled(job.id),
        )
    except BaseException:
        # A half-written file helps nobody and, on a phone already low on
        # space, actively hurts. The estimate above is only an estimate.
        dest.unlink(missing_ok=True)
        raise
    if store.is_cancelled(job.id):
        dest.unlink(missing_ok=True)
        return

    size = dest.stat().st_size
    store.add_log(job.id, f"Done in {time.time() - started:.0f}s — "
                          f"{size / 1e6:.0f} MB ({size / max(source.stat().st_size, 1):.1%} "
                          "of the video)")
    store.update(job.id, output_file=str(dest), size=size,
                 model=mode, language=job.language)


def run_compress(job) -> None:
    """Re-encode a local video smaller, measuring first so the ETA is honest.

    The measurement is not a nicety. On a phone the same job can take forty
    minutes or fourteen hours depending on whether the hardware encoder works,
    and there is no way to know which without trying it — so we try it, on
    eight seconds of the actual file, before committing the user to hours.
    """
    from . import files as files_mod

    store = jobs_mod.store()
    cfg = config.load()
    source = Path(job.audio_file)
    if not source.exists():
        raise engines.EngineError("That video is no longer there.")

    store.progress(job.id, "inspecting", 0.01)
    try:
        info = video_mod.probe_video(source)
    except video_mod.VideoError as e:
        raise engines.EngineError(str(e))

    duration = info.get("duration") or 0.0
    if duration <= 0:
        raise engines.EngineError("This video has no readable duration, so it "
                                  "cannot be compressed safely.")
    store.update(job.id, duration=duration)

    opts = job.options or {}
    quality = opts.get("compress_quality") or cfg["compress_quality"]
    speed = opts.get("compress_speed") or cfg["compress_speed"]
    codec = opts.get("compress_codec") or cfg["compress_codec"]
    try:
        plans = video_mod.build_plans(
            info, quality=quality, speed=speed, codec=codec,
            hardware=bool(cfg["compress_hardware"]))
    except video_mod.VideoError as e:
        raise engines.EngineError(str(e))

    width, height = video_mod.display_size(info)
    store.add_log(job.id, f"Source: {width}x{height} {info.get('codec') or '?'}"
                          f"{' HDR' if info.get('hdr') else ''}, "
                          f"{audio_mod.format_duration(duration)}, "
                          f"{info['size'] / 1e9:.2f} GB")
    for note in plans[0].notes:
        store.add_log(job.id, note)
    store.add_log(job.id, plans[0].audio_note)

    out_dir = files_mod.output_dir()
    dest = _unique_path(out_dir / f"{source.stem} (compressed).mp4")

    # A rough check before spending anything: if there is not room for even the
    # optimistic estimate, say so now rather than after a trial encode.
    rough = video_mod.estimate_bytes(info, plans[0], duration)
    _require_space(out_dir, rough, files_mod)

    store.progress(job.id, "measuring", 0.02)
    if len(plans) > 1 or duration > video_mod.CALIBRATE_ABOVE:
        try:
            measured = video_mod.choose_plan(
                source, plans, duration=duration, workdir=config.UPLOAD_DIR,
                stem=job.id, log=lambda m: store.add_log(job.id, m),
                should_abort=lambda: store.is_cancelled(job.id))
        except video_mod.VideoError as e:
            if str(e) == "cancelled":
                return
            raise engines.EngineError(str(e))
    else:
        measured = video_mod.Measurement(plan=plans[0], ok=True)

    if store.is_cancelled(job.id):
        return

    plan = measured.plan
    predicted = measured.predict_bytes(duration) if measured.speed else rough
    if measured.speed:
        store.add_log(
            job.id,
            f"Measured {measured.speed:.2f}x real time with {plan.label} — "
            f"about {video_mod.format_eta(measured.eta(duration))} for the whole "
            f"video, ending near {predicted / 1e6:.0f} MB "
            f"({predicted / max(info['size'], 1):.0%} of the original).")

    if predicted >= info["size"]:
        if measured.speed:
            # Measured, so this is worth acting on: hours of encoding to end up
            # with a larger file is the worst outcome available here.
            raise engines.EngineError(
                f"This would not make the file smaller — about "
                f"{predicted / 1e6:.0f} MB against the original's "
                f"{info['size'] / 1e6:.0f} MB. The video is already efficiently "
                "encoded. Choose 'Small' or 'Balanced' quality, or leave it as it is.")
        store.add_log(job.id, "Warning: this may not end up smaller than the "
                              "original — it is already efficiently encoded.")
    _require_space(out_dir, predicted, files_mod)

    store.add_log(job.id, f"Writing {dest}")
    if predicted > (1 << 30):
        # +faststart moves the index to the front once encoding is finished,
        # which on a file this size is a further pass over it with nothing to
        # report. Better to say so than to look stalled at 99%.
        store.add_log(job.id, "It will sit near the end for a while at the "
                              "finish, moving the index to the front of the "
                              "file so it starts playing instantly.")
    started = time.time()
    try:
        video_mod.compress_video(
            source, dest, plan, duration=duration,
            on_progress=lambda f: store.progress(job.id, "compressing", 0.06 + 0.92 * f),
            should_abort=lambda: store.is_cancelled(job.id))
    except video_mod.VideoError as e:
        dest.unlink(missing_ok=True)
        if str(e) == "cancelled":
            return
        raise engines.EngineError(str(e))
    except BaseException:
        # Half an encode is worth nothing and, on a phone short of space,
        # costs plenty.
        dest.unlink(missing_ok=True)
        raise
    finally:
        _clean_intermediates(job.id)

    if store.is_cancelled(job.id):
        dest.unlink(missing_ok=True)
        return

    size = dest.stat().st_size
    elapsed = time.time() - started
    store.add_log(
        job.id,
        f"Done in {video_mod.format_eta(elapsed)} — {size / 1e6:.0f} MB, "
        f"{size / max(info['size'], 1):.0%} of the original "
        f"({(info['size'] - size) / 1e9:.2f} GB saved)")
    store.update(job.id, output_file=str(dest), size=size,
                 model=f"{plan.encoder} {quality}", language=job.language)


def _require_space(out_dir: Path, needed: int, files_mod) -> None:
    """Refuse before starting rather than fail hours in with a part-written file."""
    needed = int(needed) + (64 << 20)
    free = files_mod.free_bytes(out_dir)
    if free and free < needed:
        raise engines.EngineError(
            f"Not enough space in {out_dir}: about {needed / 1e6:.0f} MB needed, "
            f"{free / 1e6:.0f} MB free. Free some space, or choose a smaller "
            "quality setting.")


def _unique_path(path: Path) -> Path:
    """Never overwrite something already in the user's Downloads."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(2, 1000):
        candidate = path.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}-{int(time.time())}{suffix}")


def run_transcription(job) -> None:
    store = jobs_mod.store()
    cfg = config.load()
    source = Path(job.audio_file)
    if not source.exists():
        raise engines.EngineError("The audio for this job is missing from disk.")

    engine = engines.get(job.engine or cfg["engine"])
    ok, reason = engine.available()
    if not ok:
        raise engines.EngineError(reason or f"{engine.label} is not available.")

    store.add_log(job.id, f"Engine: {engine.label}")

    wanted = int((job.options or {}).get("num_speakers") or 0)
    if wanted and not engine.supports_speaker_count:
        store.add_log(
            job.id,
            f"Note: {engine.label} has no speaker-count setting, so '{wanted} "
            "speakers' cannot be passed to its diarizer. It is still applied "
            "afterwards, by merging any extra speakers it invents.")

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
        wanted = int((job.options or {}).get("num_speakers") or 0)
        before = len({w.speaker for w in result.words if w.speaker is not None})
        segments = diarize_transcript(
            result.words, result.turns,
            max_gap=cfg["max_gap"], max_dur=cfg["max_dur"], max_chars=cfg["max_chars"],
            min_run_words=cfg["min_run_words"], min_run_dur=cfg["min_run_dur"],
            num_speakers=wanted,
        )
        after = len({s.speaker for s in segments if s.speaker is not None})
        if wanted and before > after:
            store.add_log(
                job.id,
                f"The engine found {before} speakers; you said {wanted}, so the "
                f"{before - after} least-spoken were merged into whoever was "
                "talking around them.")
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

    if not cfg["keep_audio"] and job.owns_audio:
        try:
            source.unlink(missing_ok=True)
            store.update(job.id, audio_file="")
        except OSError:
            pass
