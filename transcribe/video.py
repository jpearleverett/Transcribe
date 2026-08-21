"""Compressing video down to something a phone can hold, play and send.

Three facts shape this module, and all three were established by measurement
rather than assumed:

1. **A phone might have a hardware encoder, and might not be able to use it.**
   Termux's ffmpeg is built with `--enable-mediacodec --enable-jni`, and since
   ffmpeg 6.0 the MediaCodec *encoder* falls back to the NDK C API when there is
   no JavaVM — exactly the Termux case. But the bug tracker is full of devices
   where it configures fine and then freezes, leaving a 0-byte file. So the
   encoder is never trusted on the strength of appearing in `-encoders`; it has
   to produce a valid file from a real slice of the real video first.

2. **Nobody can predict how long a phone will take.** Encoder, decoder,
   resolution, thermal state and the specific silicon all matter. Rather than
   quote a table, we encode a few seconds and measure, then multiply. That same
   measurement sizes the output far better than any bits-per-pixel formula.

3. **Phone video is HDR now.** Re-encoding HDR to 8-bit H.264 without tone
   mapping is the single most common way to ruin a recording: the result is
   flat, washed-out grey. It is detected and converted properly, or not
   attempted at all.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class VideoError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# What this ffmpeg build can actually do
# --------------------------------------------------------------------------

_caps: dict = {}


def _capability(kind: str) -> set:
    """Names ffmpeg reports for `-encoders` or `-filters`, cached.

    Asking the binary beats hardcoding: the same app runs on a Termux build
    with MediaCodec and libplacebo, and on a minimal desktop build with
    neither.
    """
    if kind in _caps:
        return _caps[kind]
    names: set = set()
    if FFMPEG:
        try:
            out = subprocess.run([FFMPEG, "-hide_banner", "-" + kind],
                                 capture_output=True, text=True, timeout=30)
            for line in (out.stdout or "").splitlines():
                m = re.match(r"^\s*[A-Z.][A-Z.]{2,}\s+(\S+)", line)
                if m and m.group(1) not in ("=", "--"):
                    names.add(m.group(1))
        except (subprocess.SubprocessError, OSError):
            pass
    _caps[kind] = names
    return names


def encoders() -> set:
    return _capability("encoders")


def filters() -> set:
    return _capability("filters")


def reset_capabilities() -> None:
    """Forget the cached probe — for tests, and after installing ffmpeg."""
    _caps.clear()


def have_ffmpeg() -> bool:
    return FFMPEG is not None


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

# Transfer functions that mean "this is HDR and needs tone mapping".
HDR_TRANSFERS = {"smpte2084", "arib-std-b67", "smpte428", "bt2020-10", "bt2020-12"}


def probe_video(path: Path) -> dict:
    """Everything the planner needs about a video file.

    Reports the *video stream's* bitrate separately from the container's.
    Confusing the two is a large error on phone footage and has already cost
    this project one wrong "not enough space" refusal.
    """
    if not FFPROBE:
        raise VideoError("ffprobe is not installed. In Termux run:  pkg install ffmpeg")
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as e:
        raise VideoError(f"Could not read that video: {e}")
    if out.returncode != 0:
        detail = (out.stderr or "").strip().splitlines()
        raise VideoError("This file could not be read as video"
                         + (f": {detail[-1][:200]}" if detail else "."))
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        raise VideoError("ffprobe returned something unreadable for this file.")

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and (s.get("disposition") or {}).get("attached_pic") != 1), None)
    if video is None:
        raise VideoError("There is no video track in this file — only audio. "
                         "Use the Extract audio tab for audio files.")
    fmt = data.get("format") or {}
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _f(video.get("duration")) or _f(fmt.get("duration")) or 0.0
    transfer = (video.get("color_transfer") or "").lower()
    info = {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "codec": video.get("codec_name") or "",
        "profile": video.get("profile") or "",
        "pix_fmt": video.get("pix_fmt") or "",
        "fps": _fps(video),
        "duration": duration,
        "video_bitrate": int(_f(video.get("bit_rate")) or 0),
        "bitrate": int(_f(fmt.get("bit_rate")) or 0),
        "size": int(_f(fmt.get("size")) or _stat_size(path)),
        "rotation": _rotation(video),
        "color_transfer": transfer,
        "color_primaries": (video.get("color_primaries") or "").lower(),
        "color_space": (video.get("color_space") or "").lower(),
        "hdr": transfer in HDR_TRANSFERS,
        "audio": {
            "codec": (audio or {}).get("codec_name") or "",
            "bitrate": int(_f((audio or {}).get("bit_rate")) or 0),
            "channels": int((audio or {}).get("channels") or 0),
            "sample_rate": int((audio or {}).get("sample_rate") or 0),
        } if audio else None,
    }
    # A video stream that does not report its own rate: what is left of the
    # container after the audio track is a much better guess than the whole.
    if info["video_bitrate"] <= 0 and info["bitrate"] > 0:
        info["video_bitrate"] = max(0, info["bitrate"] - ((info["audio"] or {}).get("bitrate") or 0))
    return info


def _stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key) or ""
        if "/" in raw:
            num, _, den = raw.partition("/")
            n, d = _f(num) or 0.0, _f(den) or 0.0
            if d > 0 and n > 0:
                return n / d
    return 0.0


def _rotation(stream: dict) -> int:
    """Degrees of rotation, from either the tag or the display matrix.

    ffmpeg applies this itself when re-encoding, so this is reported rather
    than acted on — but the *displayed* orientation is what the user compares
    against, and portrait footage stored as landscape confuses every estimate
    that does not know about it.
    """
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    raw = (stream.get("tags") or {}).get("rotate")
    try:
        return int(round(float(raw))) % 360
    except (TypeError, ValueError):
        return 0


def display_size(info: dict) -> tuple:
    """Width and height as the video is actually shown, rotation applied."""
    w, h = info.get("width") or 0, info.get("height") or 0
    if info.get("rotation") in (90, 270):
        return h, w
    return w, h


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

# short side (0 = keep), x264 CRF, bits per pixel per frame, description.
QUALITY_PRESETS = {
    "small":    (720,  30, 0.055, "Small — 720p, sized for messaging"),
    "balanced": (1080, 26, 0.090, "Balanced — 1080p, much smaller and still sharp"),
    "high":     (1080, 22, 0.140, "High quality — 1080p, close to the original"),
    "original": (0,    24, 0.110, "Keep the original size — re-encode only"),
}

SPEED_PRESETS = {
    "fast":    {"libx264": "veryfast", "libx265": "faster",
                "label": "Fast — finishes sooner, file is a little larger"},
    "smaller": {"libx264": "medium", "libx265": "medium",
                "label": "Smaller — noticeably slower, squeezes harder"},
}

CODECS = {
    "h264": {"label": "H.264 — plays on everything", "sw": "libx264",
             "hw": "h264_mediacodec"},
    "hevc": {"label": "HEVC — about 40% smaller, newer devices only",
             "sw": "libx265", "hw": "hevc_mediacodec"},
}

# Audio codecs an .mp4 can carry untouched. Copying is free and lossless; the
# audio track of a phone video is a rounding error next to the picture.
_MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}


@dataclass
class Plan:
    """One concrete way to encode this file."""
    encoder: str
    codec: str = "h264"
    quality: str = "balanced"
    speed: str = "fast"
    short_side: int = 0            # target, 0 = keep the source size
    crf: int = 26
    bitrate: int = 0               # bits/s, for encoders with no CRF
    hardware: bool = False
    vf: list = field(default_factory=list)
    audio_args: list = field(default_factory=list)
    audio_note: str = ""
    notes: list = field(default_factory=list)

    @property
    def label(self) -> str:
        where = "hardware" if self.hardware else "software"
        return f"{self.encoder} ({where})"

    def args(self) -> list:
        """The ffmpeg arguments after the input, minus output path."""
        out = []
        if self.vf:
            out += ["-vf", ",".join(self.vf)]
        out += ["-c:v", self.encoder]
        if self.hardware:
            # MediaCodec has no CRF: it is a bitrate encoder, in VBR mode here
            # so quiet scenes give their bits back to busy ones.
            out += ["-b:v", str(self.bitrate), "-bitrate_mode", "1"]
        elif self.encoder == "libx264":
            # No -profile:v or -level. x264 picks both from what it is actually
            # encoding; pinning level 4.1 would be a lie about a 4K stream kept
            # at its original size, and High profile is the automatic choice on
            # anything that is not a decade old.
            out += ["-crf", str(self.crf), "-preset",
                    SPEED_PRESETS[self.speed]["libx264"]]
        elif self.encoder == "libx265":
            # x265's CRF scale runs about 5 points looser than x264's for the
            # same look, which is the whole point of using it.
            out += ["-crf", str(self.crf + 5), "-preset",
                    SPEED_PRESETS[self.speed]["libx265"],
                    "-tag:v", "hvc1"]
        else:
            out += ["-b:v", str(self.bitrate)]
        out += ["-g", "250", *self.audio_args, "-movflags", "+faststart"]
        return out


def build_plans(info: dict, *, quality: str = "balanced", speed: str = "fast",
                codec: str = "h264", hardware: bool = True) -> list:
    """Every encoder worth trying for this file, best first.

    More than one is returned on purpose. The first is whatever should be
    fastest; `choose_plan` demotes it the moment it fails to produce a valid
    file, which is how the documented MediaCodec freezes get handled without
    the user ever seeing them.
    """
    if quality not in QUALITY_PRESETS:
        quality = "balanced"
    if speed not in SPEED_PRESETS:
        speed = "fast"
    if codec not in CODECS:
        codec = "h264"

    short_side, crf, bpp, _desc = QUALITY_PRESETS[quality]
    width, height = display_size(info)
    source_short = min(width, height) if width and height else 0
    # Never upscale: asking for 1080p from a 720p source only wastes bits.
    if short_side and source_short and source_short <= short_side:
        short_side = 0

    notes = []
    vf, hdr_ok = _colour_filters(info, notes)
    if short_side:
        vf.insert(0, _scale_filter(short_side))
        notes.append(f"Scaling down to {short_side}p — the source is {width}x{height}.")
    elif source_short:
        notes.append(f"Keeping the original {width}x{height}.")

    audio_args, audio_note = _audio_plan(info)
    bitrate = target_bitrate(info, short_side, bpp)

    spec = CODECS[codec]
    candidates = []
    hw_name = spec["hw"]
    if hardware and hw_name in encoders():
        candidates.append(hw_name)
    candidates.append(spec["sw"])
    if spec["sw"] not in encoders() and len(candidates) == 1:
        raise VideoError(
            f"This ffmpeg has no {spec['sw']} encoder, so it cannot compress "
            "video. In Termux run:  pkg install ffmpeg")

    plans = []
    for name in candidates:
        if name not in encoders():
            continue
        is_hw = name.endswith("_mediacodec")
        # MediaCodec wants NV12; x264/x265 8-bit want yuv420p. Getting this
        # wrong is a hard configure failure on the hardware path.
        chain = list(vf) + ["format=nv12" if is_hw else "format=yuv420p"]
        plans.append(Plan(
            encoder=name, codec=codec, quality=quality, speed=speed,
            short_side=short_side, crf=crf, bitrate=bitrate, hardware=is_hw,
            vf=chain, audio_args=list(audio_args), audio_note=audio_note,
            notes=list(notes) + ([] if hdr_ok else [
                "This ffmpeg has no zscale filter, so HDR cannot be converted "
                "properly — the result may look flat."]),
        ))
    if not plans:
        raise VideoError("No usable video encoder was found in this ffmpeg build.")
    return plans


def _scale_filter(short_side: int) -> str:
    """Fit the *shorter* side to `short_side`, whichever way the video is turned.

    Written as expressions rather than fixed numbers so one filter handles
    landscape and portrait, and so it can never upscale. `-2` keeps the
    computed side even, which H.264 and HEVC both require.
    """
    return (f"scale=w='if(gt(iw,ih),-2,min(iw,{short_side}))'"
            f":h='if(gt(iw,ih),min(ih,{short_side}),-2)':flags=lanczos")


def _colour_filters(info: dict, notes: list) -> tuple:
    """Tone map HDR to SDR, or explain why we cannot.

    Dropping HDR tags without converting is what produces the washed-out grey
    everybody complains about: the picture is still encoded for a 1000-nit
    display and is then shown on a 100-nit one.
    """
    if not info.get("hdr"):
        return [], True
    if "zscale" not in filters():
        return [], False
    notes.append("This is HDR footage; converting it to standard range so the "
                 "colours survive H.264.")
    return ([
        "zscale=t=linear:npl=100",
        "format=gbrpf32le",
        "zscale=p=bt709",
        "tonemap=tonemap=hable:desat=0",
        "zscale=t=bt709:m=bt709:r=tv",
    ], True)


def _audio_plan(info: dict) -> tuple:
    """Copy the audio when the container allows it — it is free and lossless."""
    track = info.get("audio")
    if not track:
        return ["-an"], "No audio track."
    codec = (track.get("codec") or "").lower()
    rate = track.get("bitrate") or 0
    if codec in _MP4_AUDIO and track.get("channels", 0) <= 2 and rate <= 320_000:
        return ["-c:a", "copy"], f"Audio copied untouched ({codec or 'unknown'})."
    return (["-c:a", "aac", "-b:a", "160k", "-ac", "2"],
            f"Audio re-encoded to AAC 160k ({codec or 'unknown'} cannot be copied "
            "into an .mp4 as-is).")


def target_bitrate(info: dict, short_side: int, bpp: float) -> int:
    """Bits per second for encoders that have no quality mode.

    Bits-per-pixel-per-frame is crude but it is the right shape: a 4K frame
    genuinely needs several times the bits of a 720p one for the same look.
    """
    width, height = display_size(info)
    if short_side and width and height:
        scale = short_side / max(1, min(width, height))
        width, height = int(width * scale), int(height * scale)
    pixels = max(1, width * height) if width and height else 1920 * 1080
    fps = info.get("fps") or 30.0
    rate = int(pixels * min(fps, 60.0) * bpp)
    # Never propose a bitrate above the source's: that is not compression.
    source = info.get("video_bitrate") or 0
    if source > 0:
        rate = min(rate, source)
    return max(300_000, rate)


def estimate_bytes(info: dict, plan: Plan, duration: float) -> int:
    """A first guess at the output size, before anything has been measured."""
    duration = max(duration, 1.0)
    _, _, bpp, _ = QUALITY_PRESETS.get(plan.quality, QUALITY_PRESETS["balanced"])
    video = target_bitrate(info, plan.short_side, bpp)
    if plan.codec == "hevc":
        video = int(video * 0.6)
    audio = (info.get("audio") or {}).get("bitrate") or 128_000
    if "-an" in plan.audio_args:
        audio = 0
    return int((video + audio) / 8 * duration)


# --------------------------------------------------------------------------
# Measuring, rather than guessing
# --------------------------------------------------------------------------

@dataclass
class Measurement:
    """What a short trial encode revealed about a plan."""
    plan: Plan
    ok: bool = False
    speed: float = 0.0             # multiples of real time
    bytes_per_second: float = 0.0
    sampled: float = 0.0           # seconds of video actually encoded
    error: str = ""

    def eta(self, duration: float) -> float:
        return duration / self.speed if self.speed > 0 else 0.0

    def predict_bytes(self, duration: float) -> int:
        return int(self.bytes_per_second * max(duration, 1.0))


# Enough video to get past the first keyframe and into steady state, without
# spending real time on a measurement.
SAMPLE_SECONDS = 8.0
# A hardware encoder that has not written a valid clip by now is the freeze
# described in termux-packages#21264, not a slow one.
SAMPLE_TIMEOUT = 240.0
# Below this there is nothing to amortise a trial encode against; just run.
CALIBRATE_ABOVE = 180.0


def calibrate(src: Path, plan: Plan, *, duration: float,
              sample: float = SAMPLE_SECONDS,
              timeout: float = SAMPLE_TIMEOUT,
              workdir: Optional[Path] = None,
              stem: str = "calib",
              should_abort=None) -> Measurement:
    """Encode a slice of the real file and time it.

    Both halves matter. The timing turns "how long will this take?" from a
    guess into arithmetic, and the mere fact that a valid file came out is the
    only trustworthy evidence that this encoder works on this device — which
    for MediaCodec is a live question on every phone.
    """
    if not FFMPEG:
        raise VideoError("ffmpeg is not installed. In Termux run:  pkg install ffmpeg")
    sample = max(1.0, min(sample, duration or sample))
    # Start a little way in: the first seconds of a phone recording are often
    # a black frame and a hand moving, which encodes unrepresentatively fast.
    start = 0.0
    if duration > sample * 3:
        start = min(duration * 0.1, duration - sample)

    tmp_dir = workdir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe_file = tmp_dir / f"{stem}.trial-{int(time.time() * 1000)}.mp4"
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{sample:.3f}",
           *plan.args(), "-y", str(probe_file)]

    err_file = tempfile.TemporaryFile(mode="w+")
    try:
        began = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err_file)
        # Waited for in short slices rather than one long block: a trial that
        # hangs can burn the whole timeout, and somebody who has just read the
        # estimate and tapped Cancel should not sit through the rest of it.
        while True:
            try:
                proc.wait(timeout=0.4)
                break
            except subprocess.TimeoutExpired:
                pass
            if should_abort and should_abort():
                proc.kill()
                proc.wait(timeout=10)
                raise VideoError("cancelled")
            if time.monotonic() - began > timeout:
                proc.kill()
                proc.wait(timeout=10)
                return Measurement(plan=plan, error=(
                    f"produced nothing in {timeout:.0f}s — this encoder hangs on "
                    "this device"))
        elapsed = max(time.monotonic() - began, 1e-6)

        if proc.returncode != 0:
            err_file.seek(0)
            return Measurement(plan=plan, error=_last_line(err_file.read()))

        ok, reason, encoded = _validate_clip(probe_file, sample)
        if not ok:
            return Measurement(plan=plan, error=reason)

        size = probe_file.stat().st_size
        return Measurement(
            plan=plan, ok=True, sampled=encoded,
            speed=encoded / elapsed,
            # Biased slightly high by the container header on a short clip,
            # which is the safe direction for a free-space check.
            bytes_per_second=size / encoded,
        )
    finally:
        err_file.close()
        probe_file.unlink(missing_ok=True)


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:200] if lines else "failed with no explanation"


def _validate_clip(path: Path, expected: float) -> tuple:
    """Is this a real, decodable video — or the 0-byte file MediaCodec leaves?"""
    if not path.exists() or path.stat().st_size < 1024:
        return False, "wrote an empty file", 0.0
    if not FFPROBE:
        return True, "", expected
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames,duration",
             "-show_entries", "format=duration",
             "-print_format", "json", str(path)],
            capture_output=True, text=True, timeout=120)
        data = json.loads(out.stdout or "{}")
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return False, "wrote a file that could not be read back", 0.0
    if out.returncode != 0:
        return False, "wrote a file that could not be read back", 0.0

    streams = data.get("streams") or []
    if not streams:
        return False, "wrote a file with no video in it", 0.0
    frames = int(_f(streams[0].get("nb_read_frames")) or 0)
    encoded = (_f(streams[0].get("duration"))
               or _f((data.get("format") or {}).get("duration")) or 0.0)
    if frames < 1:
        return False, "wrote a file with no decodable frames", 0.0
    if encoded < expected * 0.5:
        return False, (f"stopped after {encoded:.1f}s of a {expected:.0f}s sample"), 0.0
    return True, "", encoded


def choose_plan(src: Path, plans: list, *, duration: float,
                sample: float = SAMPLE_SECONDS, timeout: float = SAMPLE_TIMEOUT,
                workdir: Optional[Path] = None, stem: str = "calib",
                log=None, should_abort=None) -> Measurement:
    """Trial-encode candidates in order and keep the first that really works."""
    failures = []
    for plan in plans:
        if should_abort and should_abort():
            raise VideoError("cancelled")
        if log:
            log(f"Testing {plan.label} on a {sample:.0f}s sample…")
        result = calibrate(src, plan, duration=duration, sample=sample,
                           timeout=timeout, workdir=workdir, stem=stem,
                           should_abort=should_abort)
        if result.ok:
            if log:
                log(f"{plan.label}: {result.speed:.2f}x real time.")
            return result
        failures.append(f"{plan.label} {result.error}")
        if log:
            log(f"{plan.label} did not work — {result.error}. Trying the next one.")
    raise VideoError("No encoder on this device could compress this video. "
                     + "; ".join(failures))


def format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    hours = int(minutes // 60)
    rest = int(minutes % 60)
    return f"{hours}h {rest:02d}m"


# --------------------------------------------------------------------------
# The real thing
# --------------------------------------------------------------------------

def compress_video(src: Path, dst: Path, plan: Plan, *, duration: float = 0.0,
                   on_progress=None, should_abort=None) -> Path:
    """Re-encode `src` into `dst`, reporting progress as it goes."""
    if not FFMPEG:
        raise VideoError("ffmpeg is not installed. In Termux run:  pkg install ffmpeg")
    if src.resolve() == dst.resolve():
        raise VideoError("The compressed file would overwrite the original.")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(src), *plan.args(), "-progress", "pipe:1", "-nostats",
           "-y", str(dst)]

    # stderr to a file, never a pipe. A long encode emits far more than the
    # ~64 KiB a pipe buffers, and ffmpeg would block writing it while we sat
    # reading stdout — hanging the job forever.
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
                    raise VideoError("cancelled")
            proc.wait()          # no timeout: hours is a legitimate runtime here
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.poll() is None:
                proc.kill()

        if proc.returncode != 0:
            err_file.seek(0)
            raise VideoError("ffmpeg could not compress this video: "
                             + _last_line(err_file.read()))
    finally:
        err_file.close()

    ok, reason, _ = _validate_clip(dst, min(duration, 1.0) if duration else 1.0)
    if not ok:
        raise VideoError(f"The compressed file is not usable — it {reason}.")
    return dst
