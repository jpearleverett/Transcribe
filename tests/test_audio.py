"""Real ffmpeg conversions.

Every recording goes through here before it reaches any engine, and the
conversion is what makes an hour of phone audio small enough to upload. These
tests run actual ffmpeg; they skip cleanly where it is absent.
"""

import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe import audio                                # noqa: E402

HAVE_FFMPEG = audio.have_ffmpeg()
skip = unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")


def tone_wav(path, seconds=2.0, rate=44100, channels=2, freq=440.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = []
        for i in range(int(rate * seconds)):
            v = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            frames.append(struct.pack("<" + "h" * channels, *([v] * channels)))
        w.writeframes(b"".join(frames))
    return path


def encode(src, dst, *args):
    subprocess.run([audio.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", str(src), *args, "-y", str(dst)], check=True,
                   capture_output=True)
    return dst


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="transcribe-audio-"))
        cls.wav = tone_wav(cls.dir / "source.wav")


@skip
class ProbeTest(Fixtures):
    def test_probe_wav(self):
        info = audio.probe(self.wav)
        self.assertAlmostEqual(info["duration"], 2.0, places=1)
        self.assertEqual(info["sample_rate"], 44100)
        self.assertEqual(info["channels"], 2)

    def test_probe_compressed_formats(self):
        for name, args in (("a.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
                           ("a.m4a", ["-c:a", "aac", "-b:a", "128k"]),
                           ("a.flac", ["-c:a", "flac"])):
            dst = encode(self.wav, self.dir / name, *args)
            info = audio.probe(dst)
            self.assertAlmostEqual(info["duration"], 2.0, delta=0.2, msg=name)
            self.assertGreater(info["sample_rate"], 0, name)

    def test_probe_rejects_a_file_with_no_audio(self):
        junk = self.dir / "junk.bin"
        junk.write_bytes(os.urandom(4096))
        with self.assertRaises(audio.AudioError):
            audio.probe(junk)


@skip
class ConversionTest(Fixtures):
    def test_to_wav16k_normalises_everything(self):
        """Every engine wants 16 kHz mono; whisper.cpp accepts nothing else."""
        for name, args in (("b.mp3", ["-c:a", "libmp3lame"]),
                           ("b.m4a", ["-c:a", "aac"]),
                           ("b.ogg", ["-c:a", "libvorbis"])):
            src = encode(self.wav, self.dir / name, *args)
            out = audio.to_wav16k(src, self.dir / f"{name}.16k.wav")
            with wave.open(str(out), "rb") as w:
                self.assertEqual(w.getframerate(), 16000, name)
                self.assertEqual(w.getnchannels(), 1, name)
                self.assertEqual(w.getsampwidth(), 2, name)
                self.assertGreater(w.getnframes(), 16000, name)

    def test_progress_is_reported(self):
        seen = []
        audio.to_wav16k(self.wav, self.dir / "prog.wav",
                        duration=2.0, on_progress=seen.append)
        self.assertTrue(seen, "ffmpeg -progress output should drive the bar")
        self.assertLessEqual(max(seen), 1.0)
        self.assertGreater(max(seen), 0.0)

    def test_video_container_audio_is_extracted(self):
        """A phone 'voice memo' is often an mp4 with a video stream attached."""
        mp4 = self.dir / "clip.mp4"
        subprocess.run([audio.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2",
                        "-i", str(self.wav), "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", "-y", str(mp4)], check=True, capture_output=True)
        out = audio.to_wav16k(mp4, self.dir / "clip.wav")
        with wave.open(str(out), "rb") as w:
            self.assertEqual(w.getframerate(), 16000)
            self.assertEqual(w.getnchannels(), 1)

    def test_undecodable_input_raises_with_ffmpeg_s_reason(self):
        junk = self.dir / "junk2.bin"
        junk.write_bytes(os.urandom(8192))
        with self.assertRaises(audio.AudioError) as cm:
            audio.to_wav16k(junk, self.dir / "junk2.wav")
        self.assertIn("decode", str(cm.exception).lower())

    def test_conversion_does_not_deadlock_on_noisy_stderr(self):
        """Regression: stderr on a pipe blocks ffmpeg past the pipe buffer.

        With stderr piped and only stdout read, ffmpeg stalls in write() once it
        exceeds ~64 KiB of stderr, and the job hangs forever. Decoding a damaged
        file at debug verbosity produces exactly that volume of output.
        """
        # Sized deliberately: a 60-second source corrupted every 17 bytes makes
        # ffmpeg emit ~160 KB of decode errors even at -loglevel error, which is
        # comfortably past the ~64 KiB pipe buffer. A shorter or more lightly
        # damaged file stays under it and the test would pass either way.
        long_src = tone_wav(self.dir / "long_src.wav", seconds=60.0, rate=44100)
        good = encode(long_src, self.dir / "good_long.mp3", "-c:a", "libmp3lame")
        data = bytearray(good.read_bytes())
        for i in range(100, len(data), 17):
            data[i] ^= 0xFF
        broken = self.dir / "broken.mp3"
        broken.write_bytes(bytes(data))

        result = {}

        def go():
            try:
                audio.to_wav16k(broken, self.dir / "broken.wav")
                result["ok"] = True
            except audio.AudioError:
                result["ok"] = True              # failing cleanly is fine
            except Exception as e:               # noqa: BLE001
                result["ok"] = False
                result["err"] = e

        t = threading.Thread(target=go, daemon=True)
        t.start()
        t.join(timeout=90)
        self.assertFalse(t.is_alive(), "conversion deadlocked — stderr is blocking again")
        self.assertTrue(result.get("ok"), result.get("err"))


@skip
class CompressionTest(Fixtures):
    def test_opus_is_far_smaller_and_still_decodable(self):
        """This is what makes uploading an hour of audio viable on mobile data."""
        big = tone_wav(self.dir / "long.wav", seconds=20.0)
        out = audio.to_compressed(big, self.dir / "long.opus", bitrate="24k")
        ratio = big.stat().st_size / out.stat().st_size
        self.assertGreater(ratio, 15, f"only compressed {ratio:.1f}x")
        info = audio.probe(out)
        self.assertAlmostEqual(info["duration"], 20.0, delta=1.0)
        self.assertEqual(info["channels"], 1)

    def test_bitrate_choice_changes_the_size(self):
        big = tone_wav(self.dir / "long2.wav", seconds=20.0)
        small = audio.to_compressed(big, self.dir / "lo.opus", bitrate="12k")
        large = audio.to_compressed(big, self.dir / "hi.opus", bitrate="48k")
        self.assertLess(small.stat().st_size, large.stat().st_size)


class NoFfmpegTest(unittest.TestCase):
    def test_message_names_the_fix(self):
        saved = audio.FFMPEG
        audio.FFMPEG = None
        try:
            with self.assertRaises(audio.AudioError) as cm:
                audio.to_wav16k(Path("/tmp/x"), Path("/tmp/y"))
            self.assertIn("pkg install ffmpeg", str(cm.exception))
        finally:
            audio.FFMPEG = saved

    def test_format_duration(self):
        self.assertEqual(audio.format_duration(0), "0:00")
        self.assertEqual(audio.format_duration(65), "1:05")
        self.assertEqual(audio.format_duration(3725), "1:02:05")


if __name__ == "__main__":
    unittest.main(verbosity=2)
