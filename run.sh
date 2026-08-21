#!/data/data/com.termux/files/usr/bin/bash
#
# Start the Transcribe server.
#
#   ./run.sh                 listen on 127.0.0.1:8756 (this phone only)
#   ./run.sh --port 9000     use a different port
#   ./run.sh --lan           also listen on the local network (prints a token)
#   ./run.sh --check         diagnose the setup instead of starting
#   ./run.sh --disk          show what the app is storing
#   ./run.sh --clean         delete files left by interrupted uploads
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)" || {
  echo "Python is missing. Run: pkg install python" >&2; exit 1; }

HOST=127.0.0.1
PORT="${TRANSCRIBE_PORT:-8756}"
EXTRA=()

# Diagnosis mode short-circuits everything else, including the wake lock.
for arg in "$@"; do
  if [ "$arg" = "--check" ] || [ "$arg" = "--disk" ] || [ "$arg" = "--clean" ]; then
    cd "$HERE" && exec "$PY" -m transcribe "$@"
  fi
done

while [ $# -gt 0 ]; do
  case "$1" in
    --lan)  HOST=0.0.0.0; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    -h|--help) sed -n '3,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# Android aggressively freezes background processes. A wake lock is the
# difference between a transcription that finishes while the phone is in your
# pocket and one that silently stalls. termux-wake-lock ships in termux-tools.
WAKELOCK=0
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock && WAKELOCK=1 && echo "  Wake lock acquired (Android won't freeze the job)."
else
  echo "  Note: termux-wake-lock not found. Long jobs may be paused when the screen"
  echo "        goes off. Install it with:  pkg install termux-tools"
fi

cleanup() {
  [ "$WAKELOCK" = "1" ] && command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock || true
}
trap cleanup EXIT INT TERM

cd "$HERE"
exec "$PY" -m transcribe --host "$HOST" --port "$PORT" ${EXTRA+"${EXTRA[@]}"}
