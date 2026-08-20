#!/data/data/com.termux/files/usr/bin/bash
#
# Transcribe — Termux installer.
#
#   ./install.sh            set up the cloud/GPU engines (fast, ~1 minute)
#   ./install.sh --local    also build the fully offline on-device engine
#
# The base install is deliberately tiny: the server is pure Python standard
# library, so there is nothing to pip-install and nothing to compile. --local
# is the slow path, because whisper.cpp and sherpa-onnx have to be built from
# source for Android.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${TRANSCRIBE_HOME:-$HOME/.transcribe}"
MODEL_DIR="$HOME_DIR/models"
BIN_DIR="$HOME_DIR/bin"
BUILD_DIR="$HOME_DIR/build"

WHISPER_MODEL="${WHISPER_MODEL:-small.en-q5_1}"
DO_LOCAL=0

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
die()  { printf '\033[31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }

for arg in "$@"; do
  case "$arg" in
    --local) DO_LOCAL=1 ;;
    --model=*) WHISPER_MODEL="${arg#*=}" ;;
    -h|--help)
      sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "Unknown option: $arg" ;;
  esac
done

IS_TERMUX=0
[ -n "${PREFIX:-}" ] && [ -d "${PREFIX:-/nonexistent}/bin" ] && IS_TERMUX=1

bold "Transcribe — installer"

# ---------------------------------------------------------------- base

step "Checking the basics"

if [ "$IS_TERMUX" = "1" ]; then
  info "Termux detected ($PREFIX)"
  pkg update -y >/dev/null 2>&1 || warn "pkg update failed; continuing with what's cached"
  # termux-tools carries termux-wake-lock, which is what stops Android from
  # freezing a transcription the moment the screen goes off.
  for p in python ffmpeg termux-tools; do
    if pkg list-installed 2>/dev/null | grep -q "^$p/"; then
      info "$p already installed"
    else
      info "installing $p"
      pkg install -y "$p" >/dev/null || warn "could not install $p"
    fi
  done
else
  warn "This does not look like Termux. Installing anyway — the server is portable."
fi

command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 \
  || die "Python is missing. Run: pkg install python"

PY="$(command -v python3 || command -v python)"
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "python $PYV at $PY"
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,8) else 1)' \
  || die "Python 3.8 or newer is required (found $PYV)."

if command -v ffmpeg >/dev/null 2>&1; then
  info "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)"
else
  warn "ffmpeg is missing. Audio conversion and compression will be skipped."
  warn "Install it with:  pkg install ffmpeg"
fi

mkdir -p "$HOME_DIR" "$MODEL_DIR" "$BIN_DIR"
chmod 700 "$HOME_DIR" 2>/dev/null || true
info "data directory: $HOME_DIR"

step "Checking the app"
( cd "$HERE" && "$PY" -m unittest discover -s tests -q >/dev/null 2>&1 ) \
  && info "self-test passed" \
  || warn "self-test did not pass — the app may still work, but something is off"

if [ "$DO_LOCAL" != "1" ]; then
  cat <<EOF

$(bold "Done.")

  Start it with:   ./run.sh
  Then open        http://127.0.0.1:8756/   in Chrome.

  Add an API key in Settings, or deploy the GPU worker in gpu/ if you have a
  RunPod account (that's the fastest and most accurate option).

  To also install the fully offline on-device engine:  ./install.sh --local

EOF
  exit 0
fi

# ---------------------------------------------------------------- local

[ "$IS_TERMUX" = "1" ] || die "--local is only supported on Termux (it builds native ARM64 binaries)."

bold ""
bold "Offline engine — this compiles from source and will take a while."
info "Expect 15-40 minutes and about 2 GB of storage."
info "Once built, a 1-hour recording takes roughly 20-45 minutes to process."
echo

step "Installing build tools"
for p in git cmake clang make binutils libopenblas onnxruntime python-onnxruntime python-numpy; do
  if pkg list-installed 2>/dev/null | grep -q "^$p/"; then
    info "$p already installed"
  else
    info "installing $p"
    pkg install -y "$p" >/dev/null 2>&1 || warn "could not install $p (continuing)"
  fi
done

mkdir -p "$BUILD_DIR"

# ---- whisper.cpp ----
step "Building whisper.cpp"
if [ -x "$BIN_DIR/whisper-cli" ]; then
  info "already built at $BIN_DIR/whisper-cli"
else
  cd "$BUILD_DIR"
  if [ -d whisper.cpp/.git ]; then
    info "updating existing checkout"
    ( cd whisper.cpp && git pull --ff-only >/dev/null 2>&1 || true )
  else
    info "cloning"
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp >/dev/null 2>&1 \
      || git clone --depth 1 https://github.com/ggerganov/whisper.cpp >/dev/null 2>&1 \
      || die "could not clone whisper.cpp"
  fi
  cd whisper.cpp
  info "configuring (this is the slow part)"
  # GGML_OPENMP=OFF is mandatory: Termux has no libomp and the build fails
  # confusingly at link time otherwise. GGML_NATIVE=OFF because it means
  # -march=native, which misdetects under Android's clang; we name the ARM
  # baseline explicitly instead.
  cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    >/dev/null 2>&1 \
    || cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF \
         -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF >/dev/null 2>&1 \
    || die "cmake configure failed for whisper.cpp"
  info "compiling with $(nproc) cores"
  cmake --build build --config Release -j "$(nproc)" >/dev/null 2>&1 \
    || die "whisper.cpp build failed. Try: cd $BUILD_DIR/whisper.cpp && cmake --build build -j2"
  found="$(find build -name 'whisper-cli' -type f 2>/dev/null | head -1)"
  [ -n "$found" ] || die "build finished but whisper-cli was not produced"
  cp "$found" "$BIN_DIR/whisper-cli"
  chmod +x "$BIN_DIR/whisper-cli"
  info "installed $BIN_DIR/whisper-cli"
fi

# ---- speech model ----
step "Downloading the speech model"
MODEL_FILE="$MODEL_DIR/ggml-$WHISPER_MODEL.bin"
if [ -s "$MODEL_FILE" ]; then
  info "already have $(basename "$MODEL_FILE")"
else
  URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$WHISPER_MODEL.bin"
  info "fetching ggml-$WHISPER_MODEL.bin"
  info "(small.en-q5_1 is 190 MB and English-only. For other languages or the"
  info " best accuracy: ./install.sh --local --model=large-v3-turbo-q5_0 — 574 MB,"
  info " and about 3x slower.)"
  curl -fL --progress-bar "$URL" -o "$MODEL_FILE.part" \
    || die "download failed. Check the model name at https://huggingface.co/ggerganov/whisper.cpp"
  mv "$MODEL_FILE.part" "$MODEL_FILE"
  info "saved $(du -h "$MODEL_FILE" | cut -f1)"
fi

# ---- sherpa-onnx (diarization) ----
step "Building sherpa-onnx for speaker diarization"
if "$PY" -c 'import sherpa_onnx' >/dev/null 2>&1; then
  info "sherpa-onnx already importable"
else
  # Without these two variables sherpa-onnx downloads Microsoft's glibc-linked
  # onnxruntime, which cannot load under Android's Bionic libc. Pointing them at
  # Termux's own onnxruntime package is the whole trick.
  # The header lands in one of two places depending on the package version,
  # and sherpa-onnx's fallback search only looks in /usr/include, which does
  # not exist in Termux. Find it rather than guessing.
  ORT_INC=""
  for d in "$PREFIX/include" "$PREFIX/include/onnxruntime" \
           "$PREFIX/include/onnxruntime/core/session"; do
    [ -f "$d/onnxruntime_cxx_api.h" ] && ORT_INC="$d" && break
  done
  if [ -z "$ORT_INC" ]; then
    ORT_INC="$(find "$PREFIX/include" -name onnxruntime_cxx_api.h -print -quit 2>/dev/null | xargs -r dirname)"
  fi
  if [ -z "$ORT_INC" ]; then
    warn "onnxruntime headers not found under \$PREFIX/include."
    warn "Run: pkg install onnxruntime  — then re-run ./install.sh --local"
    ORT_INC="$PREFIX/include"
  else
    info "onnxruntime headers: $ORT_INC"
  fi
  export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ORT_INC"
  export SHERPA_ONNXRUNTIME_LIB_DIR="$PREFIX/lib"
  cd "$BUILD_DIR"
  if [ -d sherpa-onnx/.git ]; then
    info "updating existing checkout"
    ( cd sherpa-onnx && git pull --ff-only >/dev/null 2>&1 || true )
  else
    info "cloning"
    git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx >/dev/null 2>&1 \
      || die "could not clone sherpa-onnx"
  fi
  cd sherpa-onnx
  info "compiling the Python bindings (15-30 minutes — go make tea)"
  export CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DSHERPA_ONNX_ENABLE_TTS=OFF -DSHERPA_ONNX_ENABLE_BINARY=OFF"
  "$PY" -m pip install --no-build-isolation . 2>&1 | tail -3 \
    || warn "sherpa-onnx build failed — the app will still work, just without speaker labels on the offline engine"
fi

# ---- diarization models ----
step "Downloading the speaker models"
SEG_TAR="$MODEL_DIR/pyannote-segmentation.tar.bz2"
if ls "$MODEL_DIR"/*segmentation*.onnx >/dev/null 2>&1; then
  info "segmentation model already present"
else
  info "fetching pyannote segmentation (6 MB)"
  curl -fL --progress-bar \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" \
    -o "$SEG_TAR" || die "segmentation model download failed"
  tar xjf "$SEG_TAR" -C "$MODEL_DIR" >/dev/null 2>&1 || die "could not extract the segmentation model"
  rm -f "$SEG_TAR"
  # Flatten and name it so the engine's model discovery finds it.
  found="$(find "$MODEL_DIR" -name 'model.int8.onnx' -path '*segmentation*' | head -1)"
  [ -n "$found" ] || found="$(find "$MODEL_DIR" -name 'model.onnx' -path '*segmentation*' | head -1)"
  [ -n "$found" ] && cp "$found" "$MODEL_DIR/pyannote-segmentation.onnx"
  info "installed pyannote-segmentation.onnx"
fi

EMB="$MODEL_DIR/speaker-embedding.onnx"
if [ -s "$EMB" ]; then
  info "speaker embedding model already present"
else
  info "fetching speaker embedding model (38 MB)"
  # Note the upstream release tag really is misspelled 'recongition'.
  curl -fL --progress-bar \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx" \
    -o "$EMB.part" || die "embedding model download failed"
  mv "$EMB.part" "$EMB"
  info "installed speaker-embedding.onnx"
fi

# Point the config at what we just installed.
"$PY" - <<PYEOF
import sys, os
sys.path.insert(0, "$HERE")
os.environ.setdefault("TRANSCRIBE_HOME", "$HOME_DIR")
from transcribe import config
config.save({"local_model": "ggml-$WHISPER_MODEL.bin"})
print("  config updated: local_model = ggml-$WHISPER_MODEL.bin")
PYEOF

cat <<EOF

$(bold "Offline engine ready.")

  Start it with:   ./run.sh
  Pick "On this phone (offline)" as the engine.

  Honest expectations for a 1-hour recording on a phone:
    transcription   12-25 minutes  (small.en-q5_1)
    diarization      6-18 minutes

  Keep the phone plugged in, and see the README's "Keeping it alive" section —
  Android will otherwise kill long jobs when the screen goes off.

EOF
