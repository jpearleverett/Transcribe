"""Audio inspection and normalisation via ffmpeg.

Every engine, cloud or local, does better on 16 kHz mono PCM than on whatever
the phone's recorder produced, and whisper.cpp accepts *only* 16 kHz mono WAV.
Converting also caps upload size: an hour of 16 kHz mono WAV is ~115 MB versus
several hundred for a high-bitrate stereo source.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# Formats Android's recorder and messaging apps actually produce, plus the
# usual desktop suspects.
AUDIO_EXTS = {
    ".mp3", ".m4a", ".mp4", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus",
    ".webm", ".amr", ".3gp", ".3gpp", ".aiff", ".aif", ".caf", ".wma", ".mkv", ".mov",
}


class AudioError(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return FFMPEG is not None


def probe(path: Path) -> dict:
    """Return {duration, sample_rate, channels, codec, bitrate, format}.

    Falls back to a WAV header read when ffprobe is missing, so the app still
    works (with reduced features) on a Termux install without ffmpeg.
    """
    if FFPROBE:
        try:
            out = subprocess.run(
                [FFPROBE, "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=120,
            )
            if out.returncode != 0:
                # ffprobe ran and rejected the file. Fail here rather than
                # falling back to a zero duration and discovering it after
                # uploading the file to a paid API.
                detail = (out.stderr or "").strip().splitlines()
                raise AudioError(
                    "This file could not be read as audio"
                    + (f": {detail[-1][:200]}" if detail else "."))
            if out.returncode == 0:
                data = json.loads(out.stdout or "{}")
                streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
                fmt = data.get("format", {})
                if streams:
                    s = streams[0]
                    dur = _f(s.get("duration")) or _f(fmt.get("duration")) or 0.0
                    return {
                        "duration": dur,
                        "sample_rate": int(s.get("sample_rate") or 0),
                        "channels": int(s.get("channels") or 0),
                        "codec": s.get("codec_name") or "",
                        "bitrate": int(_f(fmt.get("bit_rate")) or 0),
                        "format": (fmt.get("format_name") or "").split(",")[0],
                        "size": int(_f(fmt.get("size")) or path.stat().st_size),
                    }
                raise AudioError("no audio stream found in file")
        except AudioError:
            raise
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError, ValueError):
            pass
    return _probe_wav_fallback(path)


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _probe_wav_fallback(path: Path) -> dict:
    """Read a RIFF/WAVE header directly — no ffprobe required."""
    size = path.stat().st_size
    info = {"duration": 0.0, "sample_rate": 0, "channels": 0, "codec": "",
            "bitrate": 0, "format": path.suffix.lstrip("."), "size": size}
    try:
        with open(path, "rb") as fh:
            head = fh.read(44)
        if len(head) >= 44 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            channels, rate, byte_rate = struct.unpack("<HII", head[22:32])
            bits = struct.unpack("<H", head[34:36])[0]
            info.update({"sample_rate": rate, "channels": channels, "codec": "pcm",
                         "bitrate": byte_rate * 8, "format": "wav"})
            if byte_rate:
                info["duration"] = max(0.0, (size - 44) / byte_rate)
            elif rate and channels and bits:
                info["duration"] = max(0.0, (size - 44) / (rate * channels * bits / 8))
    except (OSError, struct.error):
        pass
    return info


def to_wav16k(src: Path, dst: Path, *, on_progress=None, duration: float = 0.0) -> Path:
    """Convert any input to 16 kHz mono 16-bit PCM WAV.

    `-vn` drops album art and video (a .mp4 or .webm voice memo often carries a
    video stream that would otherwise make ffmpeg fail or produce a huge file).
    """
    if not FFMPEG:
        raise AudioError(
            "ffmpeg is not installed. In Termux run:  pkg install ffmpeg"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-progress", "pipe:1", "-nostats",
        "-y", str(dst),
    ]
    # stderr goes to a file, not a pipe. Reading only stdout while ffmpeg writes
    # to a stderr PIPE deadlocks as soon as stderr exceeds the ~64 KiB pipe
    # buffer: ffmpeg blocks in write(), stops emitting progress on stdout, and
    # the read loop waits forever — hanging the job and stalling the queue
    # behind it. A file has no such limit and needs no second reader thread.
    err_file = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, text=True)
        try:
            for line in proc.stdout:
                if on_progress and line.startswith("out_time_ms=") and duration > 0:
                    try:
                        done = int(line.split("=", 1)[1].strip()) / 1_000_000.0
                        on_progress(min(1.0, done / duration))
                    except ValueError:
                        pass
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            raise AudioError("ffmpeg timed out")
        finally:
            if proc.stdout:
                proc.stdout.close()

        if proc.returncode != 0:
            err_file.seek(0)
            err = err_file.read()
            raise AudioError(f"ffmpeg failed to decode this file: {err.strip()[:500]}")
    finally:
        err_file.close()
    if not dst.exists() or dst.stat().st_size <= 44:
        raise AudioError("conversion produced an empty file — is there any audio in it?")
    return dst


def to_compressed(src: Path, dst: Path, *, bitrate: str = "48k") -> Path:
    """Downmix to a small mono Opus file for uploading over mobile data.

    16 kHz mono Opus at 48 kbps is transparent for speech and is roughly a
    twentieth the size of the WAV, which matters a lot on a metered connection.
    """
    if not FFMPEG:
        raise AudioError("ffmpeg is not installed. In Termux run:  pkg install ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", bitrate,
        "-y", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise AudioError(f"ffmpeg compression failed: {r.stderr.strip()[:400]}")
    return dst


def format_duration(seconds: float) -> str:
    total = int(seconds or 0)
    h, m, s = total // 3600, (total // 60) % 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --------------------------------------------------------------------------
# Extracting audio from video
# --------------------------------------------------------------------------

# Container each codec has to land in for a stream copy to be valid.
_COPY_CONTAINER = {
    "aac": ".m4a", "alac": ".m4a", "mp3": ".mp3", "opus": ".opus",
    "vorbis": ".ogg", "flac": ".flac", "pcm_s16le": ".wav", "pcm_s24le": ".wav",
    "ac3": ".ac3", "eac3": ".eac3", "dts": ".dts",
}

EXTRACT_MODES = {
    "copy": "Original quality — copies the audio untouched, no re-encoding",
    "opus": "Small — Opus, good for speech and a fraction of the size",
    "mp3":  "MP3 — plays anywhere",
    "wav":  "WAV 16 kHz mono — what transcription engines want",
}


def plan_extract(src: Path, mode: str = "copy") -> tuple:
    """Decide the output extension and ffmpeg arguments for an extraction.

    A stream copy is enormously the best option where it works: no quality
    loss, no CPU, and on a multi-gigabyte video the job becomes a read rather
    than a transcode. It is only valid if the codec has a container that can
    hold it, so an unknown codec falls back to re-encoding rather than
    producing a file that will not play.
    """
    info = probe(src)
    codec = (info.get("codec") or "").lower()

    if mode == "copy":
        ext = _COPY_CONTAINER.get(codec)
        if ext:
            return ext, ["-vn", "-sn", "-dn", "-c:a", "copy"], info
        mode = "opus"       # cannot copy this codec; compress instead

    if mode == "mp3":
        return ".mp3", ["-vn", "-sn", "-dn", "-c:a", "libmp3lame", "-q:a", "4"], info
    if mode == "wav":
        return ".wav", ["-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le"], info
    return ".opus", ["-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
                     "-c:a", "libopus", "-b:a", "48k"], info


def extract_audio(src: Path, dst: Path, mode: str = "copy", *,
                  on_progress=None, duration: float = 0.0,
                  should_abort=None) -> Path:
    """Pull the audio out of a video without ever copying the video itself."""
    if not FFMPEG:
        raise AudioError("ffmpeg is not installed. In Termux run:  pkg install ffmpeg")

    _, args, _info = plan_extract(src, mode)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(src), *args, "-progress", "pipe:1", "-nostats",
           "-y", str(dst)]

    err_file = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, text=True)
        try:
            for line in proc.stdout:
                if line.startswith("out_time_ms=") and duration > 0 and on_progress:
                    try:
                        done = int(line.split("=", 1)[1].strip()) / 1_000_000.0
                        on_progress(min(1.0, done / duration))
                    except ValueError:
                        pass
                if should_abort and should_abort():
                    proc.terminate()
                    raise AudioError("cancelled")
            # No timeout: a stream copy still has to read the whole file, and
            # tens of gigabytes off a phone's storage legitimately takes a while.
            proc.wait()
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.poll() is None:
                proc.kill()

        if proc.returncode != 0:
            err_file.seek(0)
            raise AudioError("ffmpeg could not extract the audio: "
                             + err_file.read().strip()[:500])
    finally:
        err_file.close()

    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioError("Extraction produced an empty file — does this video have "
                         "an audio track?")
    return dst
