"""Entry point: python -m transcribe"""

import argparse
import sys

from . import config, server


def main(argv=None) -> int:
    cfg = config.load()
    ap = argparse.ArgumentParser(prog="transcribe", description="Speaker-labelled transcription, served locally.")
    ap.add_argument("--host", default=cfg["host"],
                    help="address to bind (default 127.0.0.1 — this device only)")
    ap.add_argument("--port", type=int, default=cfg["port"], help="port to listen on")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    ap.add_argument("--check", action="store_true",
                    help="check the setup and report exactly what needs fixing")
    ap.add_argument("--network", action="store_true",
                    help="with --check, also verify your API keys actually work")
    args = ap.parse_args(argv)

    if args.version:
        print(f"Transcribe {server.VERSION}")
        return 0

    if args.check:
        from . import doctor
        return doctor.run(network=args.network)

    if args.port < 1024:
        print(f"Port {args.port} needs root on Android. Pick something above 1024.", file=sys.stderr)
        return 2

    try:
        server.serve(host=args.host, port=args.port)
    except OSError as e:
        if getattr(e, "errno", None) in (98, 48):      # EADDRINUSE
            print(f"\n  Port {args.port} is already in use.\n"
                  f"  Either the server is already running — open http://127.0.0.1:{args.port}/ —\n"
                  f"  or pick another port with:  ./run.sh --port {args.port + 1}\n", file=sys.stderr)
            return 1
        print(f"\n  Could not start the server: {e}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
