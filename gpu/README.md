# GPU worker (RunPod Serverless)

This is the fastest and most accurate engine in the app. It runs
`faster-whisper large-v3` for the transcript and word timings, and
`pyannote/speaker-diarization-community-1` for the speaker timeline. A one-hour
recording takes roughly two to three minutes on a 24 GB card.

The phone sends audio, the worker returns words and speaker turns, and the
word→speaker attribution happens back in the app so it matches every other
engine.

## 1. Get access to the diarization model

`pyannote/speaker-diarization-community-1` is CC-BY-4.0 but **gated**. You must:

1. Sign in at <https://huggingface.co/pyannote/speaker-diarization-community-1>
2. Accept the conditions on that page.
3. Create a read token at <https://huggingface.co/settings/tokens>.

Without this the worker still transcribes, but returns no speaker labels.

## 2. Build and push the image

```bash
cd gpu
docker build --build-arg HF_TOKEN=hf_your_token_here \
             -t YOUR_DOCKERHUB_USER/transcribe-worker:1 .
docker push YOUR_DOCKERHUB_USER/transcribe-worker:1
```

The build bakes ~3 GB of weights into the image on purpose: a cold start that
has to pull them from Hugging Face is the slowest part of serverless GPU work.
To skip that and have the worker fetch them on first use instead — a much
faster build and a smaller image, at the cost of a slow first request:

```bash
docker build --build-arg SKIP_PREFETCH=1 -t you/transcribe-worker:1 .
```

Building on a machine without a GPU is fine — `prefetch.py` downloads on CPU.

**On the CUDA version:** the image is pinned to CUDA 12.6 rather than 12.4 for
a specific reason. pyannote.audio 4.x requires torch ≥ 2.8, and PyTorch's cu124
wheel index stops at torch 2.6 — so a 12.4 image cannot satisfy the diarizer at
all. CTranslate2 also links against cuDNN 9, which this base image provides. If
you change either pin, check both constraints still hold.

The dependency set has been verified to install: torch 2.8.0+cu126,
torchaudio 2.8.0+cu126, torchcodec 0.7.0, pyannote.audio 4.0.7,
faster-whisper 1.2.1, ctranslate2 4.8.1.

## 3. Create the RunPod endpoint

In the [RunPod Serverless console](https://www.runpod.io/console/serverless):

- **Container image**: the image you just pushed
- **GPU**: any 24 GB card (A5000 / 4090 / L4). Peak VRAM is about 6 GB, so
  smaller cards work; 24 GB just gives headroom for batching.
- **Container disk**: 25 GB or more
- **Environment variables**:
  - `HF_TOKEN` — your Hugging Face token (also needed at runtime if you did not
    bake the diarizer in)
  - `WHISPER_MODEL` — optional, defaults to `large-v3`
- **Max workers**: 1 is plenty for personal use
- **Idle timeout**: 5-10 s keeps costs down; raise it if you transcribe in bursts
- **Execution timeout**: the app overrides this per request, but set it
  generously anyway

Copy the **Endpoint ID** from the console.

## 4. Point the app at it

In the app's Settings, paste your RunPod **API key** and the **Endpoint ID**.
Or set them in the environment before starting the server:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
```

## Payload limits, and what they mean for long recordings

RunPod caps request bodies at 10 MB on `/run` and 20 MB on `/runsync`, and
these cannot be raised. Your phone has no public URL, so the audio travels
inside the request as base64 — which inflates it by a third.

The app handles this by compressing to Opus at a bitrate chosen from the
recording's length:

| Recording length | Opus bitrate used | Notes                       |
|------------------|-------------------|-----------------------------|
| up to ~1 hour    | 48 kbps           | transparent for speech      |
| 1-2 hours        | 24-48 kbps        | still clean                 |
| 2-3 hours        | 12-24 kbps        | audibly compressed, usable  |
| over ~3.5 hours  | —                 | rejected with an explanation |

If you have somewhere to host audio the worker can reach (S3, R2, a VPS), set
`runpod_audio_url` in the config and the cap disappears entirely — the worker
downloads the file itself.

For a recording too long to fit, use Deepgram or ElevenLabs for that one file;
both accept multi-gigabyte uploads.

## Input and output

The app sends:

```json
{"input": {"audio_base64": "...", "audio_format": "opus", "language": "en",
           "model": "large-v3", "diarize": true, "num_speakers": 3}}
```

The worker returns:

```json
{"words": [{"word": "Hello", "start": 0.12, "end": 0.44, "score": 0.98}],
 "turns": [{"start": 0.0, "end": 10.2, "speaker": "SPEAKER_00"}],
 "text": "Hello ...", "language": "en", "duration": 3600.0,
 "model": "large-v3", "elapsed": 142.3}
```

`audio_url` is accepted in place of `audio_base64`.

## Testing the image locally

With an NVIDIA GPU and the container toolkit:

```bash
docker run --rm --gpus all -v "$PWD:/data" \
  YOUR_IMAGE python handler.py --selftest /data/sample.wav
```
