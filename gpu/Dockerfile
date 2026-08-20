# CUDA 12.6 + cuDNN 9 runtime. Two constraints pin this:
#   * CTranslate2 4.x links against cuDNN 9, and a mismatched pairing fails at
#     load time with an unhelpful "cannot open shared object file".
#   * pyannote.audio 4.x requires torch >= 2.8, and PyTorch's cu124 index stops
#     at torch 2.6 — so CUDA 12.4 cannot satisfy the diarizer at all. cu126 is
#     the earliest index carrying 2.8.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

# libpython3.10 is not optional here, despite looking like it. pyannote 4.x
# decodes audio through torchcodec, whose FFmpeg backend dlopens
# libpython3.10.so.1.0 — a file this base image does not ship. Without it
# torchcodec fails to load, pyannote cannot read the audio, and the worker
# returns transcripts with no speaker labels. It fails as a warning at import
# time rather than an error, so it is easy to miss until diarization is
# silently gone.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libpython3.10 ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Torch first and explicitly: pyannote would otherwise pull whatever CUDA build
# pip happens to resolve, which is frequently the wrong one for this base image.
# torchcodec is pinned alongside because pyannote only asks for >=0.7.0, and the
# latest release expects a much newer torch — letting pip pick would drag the
# whole CUDA stack out from under this image.
RUN pip install --no-cache-dir \
        torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir torchcodec==0.7.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image. A cold start that has to pull 3 GB from
# Hugging Face is the single worst part of serverless GPU inference; baking them
# turns a multi-minute cold start into seconds.
#
# speaker-diarization-community-1 is GATED: accept its conditions at
# https://huggingface.co/pyannote/speaker-diarization-community-1 and pass your
# token at build time:
#     docker build --build-arg HF_TOKEN=hf_xxx -t you/transcribe-worker .
ARG HF_TOKEN=""
ARG WHISPER_MODEL=large-v3
ARG DIARIZER_MODEL=pyannote/speaker-diarization-community-1
ENV WHISPER_MODEL=${WHISPER_MODEL} \
    DIARIZER_MODEL=${DIARIZER_MODEL}

# SKIP_PREFETCH=1 builds the image without baking in the weights. Useful for
# validating the build quickly, or if you would rather the worker download them
# on first use (slower first request, smaller image).
ARG SKIP_PREFETCH=0
COPY prefetch.py .
RUN if [ "$SKIP_PREFETCH" = "1" ]; then \
        echo "[build] skipping model prefetch"; \
    else \
        HF_TOKEN="${HF_TOKEN}" python prefetch.py; \
    fi

COPY handler.py .

CMD ["python", "-u", "handler.py"]
