"""Download model weights at image-build time so cold starts are fast."""

import os
import sys

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DIARIZER_MODEL = os.environ.get("DIARIZER_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def prefetch_whisper():
    from faster_whisper import WhisperModel
    print(f"[prefetch] {WHISPER_MODEL}")
    # Instantiating on CPU downloads and converts the weights without needing a
    # GPU during the build.
    WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("[prefetch] whisper ready")


def prefetch_diarizer():
    if not HF_TOKEN:
        print("[prefetch] WARNING: no HF_TOKEN given, so the diarizer is not baked in.\n"
              "           The worker will try to download it on first use, which\n"
              "           needs HF_TOKEN set on the endpoint and makes the first\n"
              "           request slow. Rebuild with --build-arg HF_TOKEN=hf_... .",
              file=sys.stderr)
        return
    from pyannote.audio import Pipeline
    print(f"[prefetch] {DIARIZER_MODEL}")
    pipeline = Pipeline.from_pretrained(DIARIZER_MODEL, token=HF_TOKEN)
    if pipeline is None:
        raise SystemExit(
            f"Could not fetch {DIARIZER_MODEL}. That model is gated — visit\n"
            f"  https://huggingface.co/{DIARIZER_MODEL}\n"
            "and accept the conditions with the same account as your token."
        )
    print("[prefetch] diarizer ready")


if __name__ == "__main__":
    prefetch_whisper()
    prefetch_diarizer()
