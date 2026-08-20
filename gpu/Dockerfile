# CUDA 12.4 + cuDNN 9 runtime: CTranslate2 4.x links against cuDNN 9, and a
# mismatched pairing fails at load time with an unhelpful "cannot open shared
# object file" rather than anything that names the real problem.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Torch first and explicitly: pyannote would otherwise pull whatever CUDA build
# pip happens to resolve, which is frequently the wrong one for this base image.
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu124

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

COPY prefetch.py .
RUN HF_TOKEN="${HF_TOKEN}" python prefetch.py

COPY handler.py .

CMD ["python", "-u", "handler.py"]
