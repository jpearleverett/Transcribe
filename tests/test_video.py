"""Compressing video: planning, measuring, and surviving a broken encoder.

The behaviour that matters most here is not the happy path. Android's
MediaCodec encoders are reported present by ffmpeg on devices where they then
hang, or exit cleanly having written nothing at all — so these tests spend most
of their effort on what happens when an encoder lies about working.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-video-"))

from transcribe import video                                        # noqa: E402

HAVE_FFMPEG = bool(video.FFMPEG and video.FFPROBE)
skip_no_ffmpeg = unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")

TMP = Path(tempfile.mkdtemp(prefix="transcribe-video-fixtures-"))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def _run(args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    *args], check=True, capture_output=True)


def make_video(name, *, size="640x360", seconds=6.0, fps=30, audio=True,
               bitrate="4M", codec="libx264"):
    path = TMP / name
    if path.exists():
        return path
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-c:v", codec, "-b:v", bitrate, "-preset", "ultrafast"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    args += ["-y", str(path)]
    _run(args)
    return path


def make_hdr_video(name, *, size="640x360", seconds=4.0):
    """A PQ/BT.2020 clip — what a modern phone camera actually records."""
    path = TMP / name
    if path.exists():
        return path
    _run(["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
          "-vf", "format=yuv420p10le", "-c:v", "libx265",
          "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
          "-color_primaries", "bt2020", "-color_trc", "smpte2084",
          "-colorspace", "bt2020nc", "-y", str(path)])
    return path


def fake_ffmpeg(script_body):
    """A stand-in ffmpeg, for failures a real one will not reproduce on demand.

    `$OUT` is the last argument, which is where ffmpeg's output path always
    sits. Resolved with a loop rather than `${@: -1}`, which is a bashism and
    /bin/sh here is dash.
    """
    path = TMP / f"fake-ffmpeg-{abs(hash(script_body)) % 10**8}"
    path.write_text('#!/bin/sh\nfor a in "$@"; do OUT="$a"; done\n'
                    + script_body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


# --------------------------------------------------------------------------


@skip_no_ffmpeg
class ProbeTest(unittest.TestCase):
    def test_reads_the_shape_of_a_video(self):
        info = video.probe_video(make_video("probe.mp4"))
        self.assertEqual((info["width"], info["height"]), (640, 360))
        self.assertAlmostEqual(info["fps"], 30.0, delta=0.5)
        self.assertEqual(info["codec"], "h264")
        self.assertAlmostEqual(info["duration"], 6.0, delta=0.5)
        self.assertFalse(info["hdr"])
        self.assertEqual(info["audio"]["codec"], "aac")

    def test_video_bitrate_is_not_the_container_bitrate(self):
        """The distinction that once produced a 28 GB space estimate for audio."""
        info = video.probe_video(make_video("rates.mp4", bitrate="8M"))
        self.assertGreater(info["bitrate"], 0)
        self.assertGreater(info["video_bitrate"], 0)
        # The container carries the audio too, so it is the larger of the pair.
        self.assertLess(info["video_bitrate"], info["bitrate"])

    def test_hdr_is_detected(self):
        info = video.probe_video(make_hdr_video("hdr.mp4"))
        self.assertTrue(info["hdr"])
        self.assertEqual(info["color_transfer"], "smpte2084")

    def test_audio_only_file_is_refused_by_name(self):
        path = TMP / "audio-only.m4a"
        _run(["-f", "lavfi", "-i", "sine=frequency=440:duration=2",
              "-c:a", "aac", "-y", str(path)])
        with self.assertRaises(video.VideoError) as caught:
            video.probe_video(path)
        self.assertIn("no video track", str(caught.exception))

    def test_display_size_follows_rotation(self):
        landscape = {"width": 1920, "height": 1080, "rotation": 0}
        self.assertEqual(video.display_size(landscape), (1920, 1080))
        turned = {"width": 1920, "height": 1080, "rotation": 90}
        self.assertEqual(video.display_size(turned), (1080, 1920),
                         "footage shot upright is 1080 wide, not 1920")


class PlanTest(unittest.TestCase):
    """Planning needs no ffmpeg beyond the encoder list, so it is always run."""

    def base(self, **over):
        info = {"width": 3840, "height": 2160, "fps": 30.0, "rotation": 0,
                "hdr": False, "video_bitrate": 50_000_000, "bitrate": 50_200_000,
                "size": 26_000_000_000, "duration": 7989.0,
                "audio": {"codec": "aac", "bitrate": 192_000, "channels": 2}}
        info.update(over)
        return info

    @skip_no_ffmpeg
    def test_downscaling_uses_lanczos(self):
        """Bicubic softens on the way down; lanczos is the standard choice."""
        plan = video.build_plans(self.base(), quality="balanced")[-1]
        self.assertIn("flags=lanczos", ",".join(plan.vf))

    @skip_no_ffmpeg
    def test_downscales_a_4k_source(self):
        plan = video.build_plans(self.base(), quality="balanced")[-1]
        self.assertTrue(any("scale=" in f for f in plan.vf))
        self.assertEqual(plan.short_side, 1080)

    @skip_no_ffmpeg
    def test_never_upscales(self):
        """Asking 1080p of a 720p source must not add pixels that are not there."""
        plan = video.build_plans(self.base(width=1280, height=720),
                                 quality="balanced")[-1]
        self.assertEqual(plan.short_side, 0)
        self.assertFalse(any("scale=" in f for f in plan.vf), plan.vf)

    @skip_no_ffmpeg
    def test_portrait_video_is_measured_on_its_short_side(self):
        plan = video.build_plans(self.base(width=2160, height=3840),
                                 quality="balanced")[-1]
        self.assertEqual(plan.short_side, 1080)

    @skip_no_ffmpeg
    def test_hdr_gets_a_tone_map_not_just_a_pixel_format(self):
        plan = video.build_plans(self.base(hdr=True), quality="balanced")[-1]
        chain = ",".join(plan.vf)
        if "zscale" in video.filters():
            self.assertIn("tonemap", chain)
            self.assertIn("zscale=t=linear", chain)
            self.assertTrue(any("standard range" in n for n in plan.notes))
        else:
            self.assertTrue(any("flat" in n for n in plan.notes))

    @skip_no_ffmpeg
    def test_software_and_hardware_want_different_pixel_formats(self):
        plans = video.build_plans(self.base())
        for plan in plans:
            expected = "format=nv12" if plan.hardware else "format=yuv420p"
            self.assertEqual(plan.vf[-1], expected, plan.encoder)

    @skip_no_ffmpeg
    def test_aac_audio_is_copied_and_odd_audio_is_not(self):
        copied = video.build_plans(self.base())[-1]
        self.assertIn("copy", copied.audio_args)
        recoded = video.build_plans(
            self.base(audio={"codec": "vorbis", "bitrate": 128_000, "channels": 2}))[-1]
        self.assertIn("aac", recoded.audio_args)
        silent = video.build_plans(self.base(audio=None))[-1]
        self.assertIn("-an", silent.audio_args)

    def test_target_bitrate_never_exceeds_the_source(self):
        """Re-encoding above the source rate is not compression."""
        info = self.base(width=1280, height=720, video_bitrate=900_000)
        self.assertLessEqual(video.target_bitrate(info, 0, 0.14), 900_000)

    def test_target_bitrate_scales_with_resolution(self):
        info = self.base()
        self.assertGreater(video.target_bitrate(info, 1080, 0.09),
                           video.target_bitrate(info, 720, 0.09))

    @skip_no_ffmpeg
    def test_x264_level_is_left_to_x264(self):
        """Pinning a level stamps a lie into the bitstream on a large frame.

        `-level 4.1` on a 4K encode makes x264 warn that the frame size and MB
        rate exceed the limit — and then write 4.1 into the headers anyway, so
        a strict hardware decoder sizes its buffers for 1080p and meets 4K.
        """
        args = video.build_plans(self.base(quality="original"),
                                 quality="original", hardware=False)[0].args()
        self.assertNotIn("-level", args)
        self.assertNotIn("-profile:v", args)

    @skip_no_ffmpeg
    def test_unknown_preset_names_fall_back_rather_than_reaching_ffmpeg(self):
        plan = video.build_plans(self.base(), quality="enormous", speed="warp",
                                 codec="dirac")[-1]
        self.assertEqual(plan.quality, "balanced")
        self.assertEqual(plan.speed, "fast")
        self.assertEqual(plan.codec, "h264")

    @skip_no_ffmpeg
    def test_hardware_is_offered_first_when_present_and_never_when_refused(self):
        info = self.base()
        # Pretend this build has Android's encoder, as Termux's does.
        original = video._caps.get("encoders")
        video._caps["encoders"] = set(original or set()) | {"h264_mediacodec"}
        try:
            first = video.build_plans(info, hardware=True)[0]
            self.assertEqual(first.encoder, "h264_mediacodec")
            self.assertTrue(first.hardware)
            self.assertIn("-bitrate_mode", first.args(),
                          "MediaCodec has no CRF; it must be given a bitrate")
            self.assertNotIn("-crf", first.args())

            names = [p.encoder for p in video.build_plans(info, hardware=False)]
            self.assertNotIn("h264_mediacodec", names)
        finally:
            if original is None:
                video._caps.pop("encoders", None)
            else:
                video._caps["encoders"] = original


@skip_no_ffmpeg
class CalibrationTest(unittest.TestCase):
    def test_measures_speed_and_predicts_the_final_size(self):
        src = make_video("calib.mp4", seconds=20.0, size="854x480", bitrate="6M")
        info = video.probe_video(src)
        plan = video.build_plans(info, quality="balanced", hardware=False)[0]
        measured = video.calibrate(src, plan, duration=info["duration"],
                                   sample=4.0, workdir=TMP)
        self.assertTrue(measured.ok, measured.error)
        self.assertGreater(measured.speed, 0)
        self.assertGreater(measured.bytes_per_second, 0)

        dst = TMP / "calib-out.mp4"
        video.compress_video(src, dst, plan, duration=info["duration"])
        predicted = measured.predict_bytes(info["duration"])
        actual = dst.stat().st_size
        # A four-second sample carries the container header, so it reads high;
        # what matters is that it is the right order of magnitude, not exact.
        self.assertLess(abs(predicted - actual) / actual, 0.6,
                        f"predicted {predicted}, got {actual}")

    def test_no_trial_files_are_left_behind(self):
        src = make_video("calib-clean.mp4", seconds=6.0)
        info = video.probe_video(src)
        plan = video.build_plans(info, hardware=False)[0]
        work = TMP / "workdir"
        work.mkdir(exist_ok=True)
        video.calibrate(src, plan, duration=info["duration"], sample=2.0,
                        workdir=work, stem="job123")
        self.assertEqual(list(work.iterdir()), [])

    def test_an_encoder_that_writes_nothing_is_caught(self):
        """Exit code 0 and a 0-byte file: the Pixel 9 MediaCodec failure."""
        src = make_video("calib2.mp4", seconds=6.0)
        info = video.probe_video(src)
        plan = video.build_plans(info, hardware=False)[0]
        liar = fake_ffmpeg(': > "$OUT"; exit 0')
        real, video.FFMPEG = video.FFMPEG, str(liar)
        try:
            measured = video.calibrate(src, plan, duration=info["duration"],
                                       sample=2.0, workdir=TMP)
        finally:
            video.FFMPEG = real
        self.assertFalse(measured.ok)
        self.assertIn("empty file", measured.error)

    def test_an_encoder_that_hangs_is_caught_rather_than_waited_on(self):
        """The other documented MediaCodec failure: it never returns at all."""
        src = make_video("calib3.mp4", seconds=6.0)
        info = video.probe_video(src)
        plan = video.build_plans(info, hardware=False)[0]
        hanger = fake_ffmpeg("sleep 600")
        real, video.FFMPEG = video.FFMPEG, str(hanger)
        try:
            measured = video.calibrate(src, plan, duration=info["duration"],
                                       sample=2.0, timeout=2.0, workdir=TMP)
        finally:
            video.FFMPEG = real
        self.assertFalse(measured.ok)
        self.assertIn("hangs", measured.error)

    def test_a_truncated_clip_is_not_accepted(self):
        src = make_video("calib4.mp4", seconds=10.0)
        info = video.probe_video(src)
        plan = video.build_plans(info, hardware=False)[0]
        # Encodes a tenth of what was asked for, then stops — as a codec that
        # dies partway through a GOP does.
        stopper = fake_ffmpeg(
            f'"{video.FFMPEG}" -nostdin -loglevel error -i "{src}" -t 0.2 '
            '-c:v libx264 -an -y "$OUT"')
        real, video.FFMPEG = video.FFMPEG, str(stopper)
        try:
            measured = video.calibrate(src, plan, duration=info["duration"],
                                       sample=8.0, workdir=TMP)
        finally:
            video.FFMPEG = real
        self.assertFalse(measured.ok)
        self.assertIn("stopped after", measured.error)


@skip_no_ffmpeg
class FallbackTest(unittest.TestCase):
    def test_a_broken_first_choice_is_abandoned_for_a_working_one(self):
        """The whole reason candidates are a list: hardware may simply not work."""
        src = make_video("fallback.mp4", seconds=6.0)
        info = video.probe_video(src)
        good = video.build_plans(info, hardware=False)[0]
        broken = video.Plan(encoder="definitely_not_an_encoder", vf=list(good.vf),
                            audio_args=list(good.audio_args), hardware=True,
                            bitrate=1_000_000)

        logged = []
        measured = video.choose_plan(src, [broken, good], duration=info["duration"],
                                     sample=2.0, workdir=TMP, log=logged.append)
        self.assertTrue(measured.ok)
        self.assertEqual(measured.plan.encoder, good.encoder)
        self.assertTrue(any("did not work" in line for line in logged), logged)

    def test_when_nothing_works_the_reasons_are_reported(self):
        src = make_video("fallback2.mp4", seconds=6.0)
        info = video.probe_video(src)
        broken = video.Plan(encoder="definitely_not_an_encoder", bitrate=1_000_000)
        with self.assertRaises(video.VideoError) as caught:
            video.choose_plan(src, [broken], duration=info["duration"],
                              sample=2.0, workdir=TMP)
        self.assertIn("definitely_not_an_encoder", str(caught.exception))


@skip_no_ffmpeg
class CompressTest(unittest.TestCase):
    def test_it_actually_makes_the_file_smaller_and_leaves_the_source_alone(self):
        src = make_video("shrink.mp4", size="1280x720", seconds=8.0, bitrate="12M")
        before = src.stat().st_size
        info = video.probe_video(src)
        plan = video.build_plans(info, quality="small", hardware=False)[0]
        dst = TMP / "shrink-out.mp4"
        video.compress_video(src, dst, plan, duration=info["duration"])

        self.assertLess(dst.stat().st_size, before)
        self.assertEqual(src.stat().st_size, before, "the original must not change")
        out = video.probe_video(dst)
        self.assertEqual(min(out["width"], out["height"]), 720)
        self.assertAlmostEqual(out["duration"], info["duration"], delta=1.0)

    def test_hdr_becomes_properly_tagged_sdr(self):
        src = make_hdr_video("hdr-in.mp4")
        info = video.probe_video(src)
        plan = video.build_plans(info, quality="original", hardware=False)[0]
        if "zscale" not in video.filters():
            self.skipTest("this ffmpeg has no zscale")
        dst = TMP / "hdr-out.mp4"
        video.compress_video(src, dst, plan, duration=info["duration"])
        out = video.probe_video(dst)
        self.assertFalse(out["hdr"])
        self.assertEqual(out["color_transfer"], "bt709")
        self.assertEqual(out["pix_fmt"], "yuv420p")

    def test_tone_mapping_is_not_cosmetic(self):
        """Without it the picture is the washed-out grey everyone complains of."""
        if "zscale" not in video.filters():
            self.skipTest("this ffmpeg has no zscale")
        src = make_hdr_video("hdr-compare.mp4")
        info = video.probe_video(src)
        plan = video.build_plans(info, quality="original", hardware=False)[0]
        mapped = TMP / "mapped.mp4"
        video.compress_video(src, mapped, plan, duration=info["duration"])

        naive = TMP / "naive.mp4"
        flat = video.Plan(encoder="libx264", vf=["format=yuv420p"],
                          audio_args=["-an"], crf=plan.crf)
        video.compress_video(src, naive, flat, duration=info["duration"])

        self.assertLess(_luma(mapped), _luma(naive) - 15,
                        "the tone-mapped version should have real blacks")

    def test_it_refuses_to_overwrite_the_original(self):
        src = make_video("selfsame.mp4", seconds=4.0)
        info = video.probe_video(src)
        plan = video.build_plans(info, hardware=False)[0]
        with self.assertRaises(video.VideoError) as caught:
            video.compress_video(src, src, plan, duration=info["duration"])
        self.assertIn("overwrite", str(caught.exception))
        self.assertGreater(src.stat().st_size, 0)

    def test_progress_climbs_to_the_end(self):
        src = make_video("progress.mp4", seconds=8.0)
        info = video.probe_video(src)
        plan = video.build_plans(info, hardware=False)[0]
        seen = []
        video.compress_video(src, TMP / "progress-out.mp4", plan,
                             duration=info["duration"], on_progress=seen.append)
        self.assertTrue(seen)
        self.assertEqual(seen, sorted(seen), "progress must not go backwards")
        self.assertGreater(seen[-1], 0.5)
        self.assertLessEqual(max(seen), 1.0)


def _luma(path):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path),
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
         "-f", "null", "-"], capture_output=True, text=True)
    values = [float(line.split("=")[-1]) for line in out.stderr.splitlines()
              if "YAVG=" in line]
    assert values, out.stderr[-500:]
    return sum(values) / len(values)


class FormattingTest(unittest.TestCase):
    def test_eta_reads_like_a_person_wrote_it(self):
        self.assertEqual(video.format_eta(45), "45 seconds")
        self.assertEqual(video.format_eta(600), "10 minutes")
        self.assertEqual(video.format_eta(9000), "2h 30m")


if __name__ == "__main__":
    unittest.main()
