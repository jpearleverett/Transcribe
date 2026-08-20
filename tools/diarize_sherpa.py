#!/usr/bin/env python3
"""Speaker diarization via sherpa-onnx. Prints {"turns": [...]} as JSON on stdout.

Runs as a subprocess rather than in-process so that a segfault in a native ONNX
library costs you the speaker labels, not the whole server and the transcript
you just spent half an hour producing on a phone CPU.

Reads WAV with the stdlib `wave` module rather than soundfile/librosa: neither
installs cleanly in Termux, and the input is always 16 kHz mono PCM by the time
it reaches here.
"""

import argparse
import audioop
import json
import sys
import wave


def read_wav_mono_f32(path, want_rate):
    """Return float32 samples in [-1, 1], mono, at `want_rate`.

    Uses numpy when it is available (Termux ships `python-numpy`) and falls back
    to array.array so a leaner install still works — the fallback is slower but
    both expose the buffer protocol pybind11 wants.
    """
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
        width = 2
    if channels == 2:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
        channels = 1
    elif channels > 2:
        frames = _downmix(frames, channels)
        channels = 1
    if rate != want_rate:
        # audioop.ratecv is not the world's finest resampler, but the input is
        # already 16 kHz in normal operation; this is a safety net.
        frames, _ = audioop.ratecv(frames, width, 1, rate, want_rate, None)

    try:
        import numpy as np
    except ImportError:
        import array
        pcm = array.array("h")
        pcm.frombytes(frames[: len(frames) // 2 * 2])
        if sys.byteorder == "big":
            pcm.byteswap()
        return array.array("f", (s / 32768.0 for s in pcm))

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return np.ascontiguousarray(samples)


def _downmix(frames, channels):
    """Average N channels down to one, without numpy."""
    import array
    a = array.array("h")
    a.frombytes(frames[: len(frames) // (2 * channels) * 2 * channels])
    if sys.byteorder == "big":
        a.byteswap()
    out = array.array("h", (sum(a[i:i + channels]) // channels
                            for i in range(0, len(a), channels)))
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segmentation", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--num-speakers", type=int, default=0,
                    help="exact speaker count if you know it; 0 means auto")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="clustering threshold when the count is unknown; "
                         "smaller finds more speakers")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-duration-on", type=float, default=0.3)
    ap.add_argument("--min-duration-off", type=float, default=0.5)
    ap.add_argument("--out", default="", help="write JSON here instead of stdout")
    args = ap.parse_args()

    try:
        import sherpa_onnx
    except ImportError:
        die("sherpa-onnx is not installed. Run ./install.sh --local in Termux.")

    try:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=args.segmentation,
                ),
                num_threads=args.threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=args.embedding,
                num_threads=args.threads,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                # -1 means "unknown": the threshold decides how many speakers.
                # A positive count makes the threshold irrelevant, which is why
                # telling it the real number is such a large accuracy win.
                num_clusters=args.num_speakers if args.num_speakers > 0 else -1,
                threshold=args.threshold,
            ),
            min_duration_on=args.min_duration_on,
            min_duration_off=args.min_duration_off,
        )
    except (AttributeError, TypeError) as e:
        die(f"This sherpa-onnx build has a different API than expected: {e}")

    if not config.validate():
        die("sherpa-onnx rejected the diarization config — are both model files present?")

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    try:
        samples = read_wav_mono_f32(args.wav, sd.sample_rate)
    except (wave.Error, OSError, ImportError) as e:
        die(f"Could not read {args.wav}: {e}")

    if len(samples) == 0:
        emit(args.out, {"turns": []})
        return

    def on_progress(done, total):
        # Progress goes to stderr so stdout stays pure JSON.
        if total:
            print(f"progress = {int(100 * done / total)}%", file=sys.stderr, flush=True)
        return 0        # non-zero would abort

    try:
        result = sd.process(samples, callback=on_progress)
    except TypeError:
        result = sd.process(samples)        # older builds without the callback

    turns = [
        {"start": round(float(seg.start), 3),
         "end": round(float(seg.end), 3),
         "speaker": str(seg.speaker)}
        for seg in result.sort_by_start_time()
    ]
    emit(args.out, {"turns": turns,
                    "num_speakers": len({t["speaker"] for t in turns})})


def emit(out_path, payload):
    text = json.dumps(payload)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
    else:
        print(text)


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
