# Transcribe

Upload an audio file from your phone, get back a transcript that says **who
said what, and when**. Runs as a small server in Termux; you use it from Chrome
on the same phone.

```
┌──────────────┐        ┌─────────────────┐        ┌──────────────────┐
│ Chrome       │  HTTP  │ Termux          │  API   │ Engine           │
│ on your      │ ─────► │ python server   │ ─────► │ GPU pod / cloud  │
│ phone        │ ◄───── │ (stdlib only)   │ ◄───── │ / this phone     │
└──────────────┘  SSE   └─────────────────┘        └──────────────────┘
```

The server does the work and writes every state change to disk, so you can lock
the phone, switch apps, or kill Chrome mid-job and come back to a finished
transcript.

---

## Quick start

```bash
pkg install git
git clone <this repo> Transcribe
cd Transcribe
./install.sh
./run.sh
```

Then open **http://127.0.0.1:8756/** in Chrome, tap the settings gear, and add
an API key for one of the engines below.

If anything misbehaves, run the setup check first — it names the problem and
the exact command that fixes it:

```bash
./run.sh --check              # environment, engines, storage, self-test
./run.sh --check --network    # also proves your API keys actually work
```

The base install needs nothing but `python` and `ffmpeg` — the server is pure
standard library, so there is no pip step to fail on a phone.

---

## Extracting audio from video

The **Extract audio** tab browses the media already on your phone and pulls the
audio out of a video, so you never upload it. That distinction matters: a
two-hour 4K recording can be 30 GB, and putting that through the browser would
have Chrome read all of it and the server write a *second* copy to the same
phone — needing twice the space to accomplish nothing. ffmpeg reads the original
where it sits and writes only the audio.

Four output modes:

| Mode | What it does | An hour of video |
|---|---|---|
| **Original quality** (default) | Copies the audio stream untouched — no re-encoding, no quality loss, barely any CPU | ~60-100 MB |
| Small | Opus, 16 kHz mono — ideal for speech | ~20 MB |
| MP3 | Plays anywhere | ~30 MB |
| WAV | 16 kHz mono, what engines want | ~115 MB |

Extracted audio lands in your **Downloads** folder, and an existing file is never
overwritten. From the result you can transcribe it in one tap.

Any media file in the browser can also be **transcribed directly**, skipping the
upload. For anything large, prefer that over the file picker.

Termux needs storage permission before it can see your videos:

```bash
termux-setup-storage      # then allow the Android permission
```

Nothing the app did not create is ever deleted — deleting a job removes only
files it made, never your originals.

## Which engine should you use?

| Engine | Speed for 1 hour of audio | Word-level speakers | Cost | Audio leaves the phone |
|---|---|---|---|---|
| **RunPod GPU** (your own pod) | ~2-3 min | yes | your GPU time (~$0.02) | to your pod only |
| **AssemblyAI** | ~1-2 min | yes | ~$0.23/hr, $50 free | yes |
| **Deepgram Nova-3** | ~40 s | yes | ~$0.26/hr, $200 free | yes |
| **ElevenLabs Scribe v2** | ~1 min | yes | ~$0.22/hr | yes |
| **OpenAI** | ~2 min | segments only | ~$0.36/hr | yes |
| **On this phone** | 20-45 min | yes | free | **no** |


**If you have the RunPod pod set up, use it.** It is the most accurate option
and the audio only ever touches infrastructure you control. See
[`gpu/README.md`](gpu/README.md) for the deploy steps — it is a Docker build,
a push, and pasting an endpoint ID into Settings.

**If you want to start in two minutes**, use Deepgram: $200 of free credit with
no card, and a single API call.

**If accuracy is all that matters and you don't mind waiting**, AssemblyAI has
the best published diarization numbers of the managed services.

**Avoid OpenAI for this specific job.** No OpenAI model returns speaker labels
*and* word-level timestamps in one response, and the 25 MB upload cap means long
recordings get chopped up, which breaks speaker continuity. It is included for
completeness and clearly labelled in the UI.

---

## Getting accurate results

A few things matter far more than which engine you pick:

**Tell it how many speakers there are.** Under *Options*, set the speaker count
if you know it. Diarizers spend most of their error budget guessing this number.

Engines differ in what they can do with it. AssemblyAI and ElevenLabs pass it to
their diarizer, which is the better outcome. Deepgram's API has no speaker-count
parameter at all, so the app applies it afterwards instead, merging any extra
speakers the engine invented into whoever was talking around them. On simulated
over-segmentation that takes word diarization error from 15.2% to 4.7%.

**Record in a quiet room, one mic, close to the speakers.** Diarization degrades
much faster than transcription does with distance and background noise.

**Overlapping speech is the hard case.** When two people talk at once, every
system available today attributes the overlap to one of them. Expect a word or
two to land on the wrong speaker around interruptions.

**Don't pre-process the audio.** The app converts to 16 kHz mono internally,
which is what every engine wants. Noise reduction applied beforehand usually
hurts more than it helps — it removes cues the diarizer uses.

---

## Keeping it alive on Android

This is the part that trips people up, and it is not the app's fault.

**Android 12+ kills background processes.** There is a "phantom process" limit
of 32; Termux's children count against it, and Android will `SIGKILL` a long
transcription without warning.

1. `./run.sh` takes a **wake lock** automatically (via `termux-wake-lock`), which
   handles most cases.
2. Turn off battery optimisation for Termux: *Settings → Apps → Termux → Battery
   → Unrestricted*.
3. On **Android 14 and newer**, there is a developer-options toggle: enable
   *Developer options → Disable child process restrictions*.
4. On some OEM builds (Samsung and Xiaomi especially) you may also need to lock
   Termux in the recent-apps list.

If a job dies anyway, it reappears as **Interrupted** with a Retry button —
nothing is lost except the time.

Keep the phone plugged in for anything long, particularly on the offline engine.

---

## The offline engine

```bash
./install.sh --local            # ~15-40 min: compiles from source
./install.sh --local --model=large-v3-turbo-q5_0   # bigger, better, ~3x slower
```

This builds **whisper.cpp** for transcription and **sherpa-onnx** for speaker
diarization, both native ARM64. Nothing is downloaded at runtime and nothing
leaves the phone.

Honest expectations for one hour of audio on a modern phone:

| Model | Size | Peak RAM | Transcription time | Quality |
|---|---|---|---|---|
| `small.en-q5_1` (default) | 190 MB | ~600 MB | 12-25 min | good, **English only** |
| `large-v3-turbo-q5_0` | 574 MB | ~1.2 GB | 35-60 min | best, multilingual |
| `base.en-q5_1` | 60 MB | ~400 MB | 5-8 min | rough |

Diarization adds another 6-18 minutes on top.

The default is the small model deliberately. Peak RAM, not file size, is what
gets a Termux process silently killed by Android, and `large-v3-turbo` wants
about 1.2 GB resident for the whole run. Move up to it if you need other
languages or the extra accuracy, ideally with the phone plugged in and
everything else closed.

Two things worth knowing, because the obvious approach does not work:

- whisper.cpp's own `--diarize` flag is **stereo** diarization — it compares
  left/right channel energy. On a phone recording it does nothing. Speaker
  identity comes from sherpa-onnx instead.
- PyTorch has no Termux build, so pyannote and WhisperX cannot run natively on
  the phone at all. That is why the offline path uses ONNX Runtime, which Termux
  does package.

---

## Using it from another device

```bash
./run.sh --lan
```

This binds to your local network and prints a URL containing an access token.
The token is required — without it, anyone else on the same wifi could read your
recordings. Loopback-only (the default) needs no token.

Note that `http://127.0.0.1` is a secure context in Chrome, but a LAN IP is not,
so some browser features are unavailable over `--lan`.

---

## Exports

Every transcript exports as plain text, Markdown, SRT, WebVTT, CSV, or JSON.
The JSON keeps per-word timings and confidences if the engine provided them.

Tap a speaker chip to rename them; the name flows through to every export.

---

## How it works

```
upload → probe duration → convert/compress (ffmpeg)
       → engine (words + speaker turns)
       → 1. assign each WORD a speaker by temporal overlap
       → 2. absorb implausible one-word flips
       → 3. snap sentences to their dominant speaker
       → 4. group into readable segments
       → save to disk
```

Steps 1-3 are where the accuracy lives.

**Why not just use the engine's segments?** Most tools assign a whole ASR
segment to whichever speaker covers most of it. Whisper-style segments are ~30 s
blocks whose boundaries are decoder artifacts, not acoustic ones, so this loses
every short interjection and smears turn boundaries by seconds.

**1. Per-word overlap.** Each word is attributed independently by summing how
much it overlaps each speaker's turns and taking the winner. A word straddling a
boundary goes to whichever side holds more of it. Words landing in a diarizer's
gap fall back to the nearest turn.

**2. Run-length smoothing.** A run that is short in *both* time and word count,
with the same speaker on both sides, is absorbed — that combination is
overwhelmingly a diarizer boundary error. A run forming a complete sentence of
its own ("Right.") is kept regardless of length, because those are real
interjections and are frequently under 200 ms.

**3. Sentence-scoped majority vote.** Within each `.?!`-delimited sentence, if
one speaker holds 60% of the talk time, stray words are snapped to them — real
speaker changes almost never happen mid-sentence. This is applied conservatively,
skipping long unpunctuated runs (which genuinely do span turns), minority
stretches over 1.5 s (real turns), and runs at a sentence's *edge* (ambiguous:
just as likely a real turn the punctuation lags by a word).

See [`transcribe/align.py`](transcribe/align.py). It is the part most worth
reading, and `tests/test_align.py` covers each of these cases.

**Does it actually help?** `tests/test_accuracy.py` builds a synthetic
conversation with known ground truth, runs a diarizer simulation that jitters
turn boundaries and drops short turns the way real ones do, and measures word
diarization error rate:

| Method | WDER |
|---|---|
| Naive segment-level majority | 36.4% |
| This pipeline | 1.5% |

With an exact diarization timeline the pipeline is lossless (0% WDER), and it
holds under ±0.8 s of boundary jitter at 6.1%.

---

## Layout

```
transcribe/          the server
  align.py           word→speaker attribution and segmentation
  files.py           browsing device media, fenced to media folders
  storage.py         disk accounting and reclaiming orphaned uploads
  doctor.py          the --check setup diagnosis
  exporters.py       SRT / VTT / TXT / MD / CSV / JSON
  server.py          HTTP, SSE, Range requests — stdlib only
  jobs.py            job store, worker thread, crash recovery
  runner.py          pipeline orchestration
  audio.py           ffmpeg probe and conversion
  httpclient.py      streaming uploads without loading files into RAM
  engines/           one adapter per provider
web/                 the front end (no build step, no framework)
gpu/                 the RunPod GPU worker
tools/               diarize_sherpa.py — offline diarization helper
tests/               283 tests, no network needed
```

Run the tests with `python3 -m unittest discover -s tests`. They need no
network and no API keys. Two groups skip themselves when their tooling is
absent: the ffmpeg conversion tests, and `tests/test_browser.py`, which drives
the real app in Chromium on an emulated Pixel and needs Playwright:

```bash
pip install playwright && playwright install chromium
python3 -m unittest tests.test_browser
```

That one earns its keep — it caught a CSS rule that overrode the `hidden`
attribute and left both dropdown menus permanently open on screen, which no
amount of server-side testing would have found.

---

## Troubleshooting

**"Termux is using tens of gigabytes"** — almost certainly an interrupted
upload. The app streams an upload to disk *before* creating the job that points
at it, so a process killed in between (Android does kill Termux) leaves a file
nothing references. A partly-uploaded video can be enormous.

```bash
./run.sh --disk       # where the space went — all of Termux, not just this app
./run.sh --clean      # delete files no transcript uses
```

`--disk` accounts for the whole Termux install, which is what Android's app-size
figure covers: the app's own storage, your home directory's biggest folders,
every individual file over 64 MB, and the package caches. `~/storage/*` are
symlinks to your shared storage, so they are deliberately excluded — your photos
and videos are not part of Termux's size.

The server now also reclaims these automatically each time it starts. For very
large media, prefer the **Extract audio** tab or transcribing from the file
browser — neither uploads anything, so neither can leave this behind.

**Start here:** `./run.sh --check`. It walks the whole setup — Python, ffmpeg
and whether it has the Opus encoder, wake locks, disk space, config file
permissions, every engine's readiness, the offline build, and an end-to-end
self-test of the transcript pipeline — and prints the exact fix for anything
wrong. Add `--network` and it verifies each API key against the provider and
tells you, for example, that your RunPod key is fine but the endpoint ID does
not exist on that account.

**"Port 8756 is already in use"** — it is probably already running. Open
http://127.0.0.1:8756/, or use `./run.sh --port 8757`.

**The file picker won't show my recording** — some Android file managers hide
files behind an `audio/*` filter. Use the *Files* app, or switch the picker to
"Browse" and select from Downloads.

**Upload gets to 100% then fails** — the phone ran out of storage for the
converted copy. Delete some old transcripts (which deletes their audio too).

**"No speech was found"** — the file is silent, is music, or ffmpeg couldn't
decode it. Check with `ffprobe yourfile.m4a`.

**Everything is attributed to one speaker** — either only one person is audible,
or diarization was skipped. Open the job's *Details* panel; it says which.

**The offline engine build fails** — the usual cause is running out of RAM
during compilation. Retry with fewer cores:
`cd ~/.transcribe/build/whisper.cpp && cmake --build build -j2`.

---

## Privacy

Cloud engines upload your audio to that provider — that is inherent to using
them. The RunPod engine uploads only to your own pod. The offline engine sends
nothing anywhere.

API keys are stored in `~/.transcribe/config.json` with `0600` permissions,
never in the repo, and the server never sends a key back to the browser — the
settings screen only ever shows whether one is set.

Keys can also come from the environment (`DEEPGRAM_API_KEY`, `RUNPOD_API_KEY`,
and so on). A key entered in Settings takes precedence over the environment, and
Settings says when a key came from the environment rather than from you.

Uploaded audio stays in `~/.transcribe/uploads` so the player can seek through
it. Deleting a transcript deletes its audio.

The server binds to loopback only unless you pass `--lan`, and it refuses
cross-site POST/PATCH/DELETE requests. That matters more than it sounds: a
loopback server is reachable from any page open in the same browser, and while
CORS hides the *response* from that page, the request still lands — so without
the check, a website you visited could quietly delete your transcripts.
