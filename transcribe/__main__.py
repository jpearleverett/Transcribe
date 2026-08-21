"""Entry point: python -m transcribe"""

import argparse
import os
import sys

from . import config, server


def _disk(clean: bool = False) -> int:
    from . import storage
    from .storage import human

    config.ensure_dirs()
    data = storage.report()

    print(f"\n\033[1mTranscribe is storing {human(data['total'])}\033[0m  in {data['home']}\n")
    for part in data["parts"]:
        if not part["size"] and not part["files"]:
            continue
        print(f"  {human(part['size']):>10}  {part['label']:<16} "
              f"{part['files']:>4} file(s)   {part['note']}")

    if data["orphans"]:
        print(f"\n\033[33m  {human(data['orphan_bytes'])} of that is orphaned\033[0m — "
              "files from uploads that were interrupted, which no transcript uses:")
        for o in data["orphans"][:10]:
            print(f"      {human(o['size']):>10}  {o['name'][:58]}"
                  f"  ({o['age'] / 3600:.0f}h old)")
        if len(data["orphans"]) > 10:
            print(f"      … and {len(data['orphans']) - 10} more")
        if clean:
            count, freed = storage.reap()
            print(f"\n  \033[32mDeleted {count} file(s), freeing {human(freed)}.\033[0m")
        else:
            print("\n  Delete them with:  ./run.sh --clean")
    elif clean:
        print("\n  Nothing orphaned to clean up.")

    build = next((p for p in data["parts"] if p["label"] == "Build tree"), None)
    if build and build["size"] > 500 << 20:
        print(f"\n  The build tree is {human(build['size'])}. It is only needed to rebuild")
        print(f"  the offline engine, so it is safe to delete:")
        print(f"      rm -rf {build['path']}")

    # Android reports one figure for the whole Termux install, so account for
    # all of it rather than only the part this app wrote.
    print("\n\033[1mAll of Termux\033[0m (what Android's app-size figure covers)\n")
    scan = storage.scan_termux()
    for area in scan["areas"]:
        print(f"  {human(area['size']):>10}  {area['label']:<26} {area['files']:>6} files")
    print(f"  {human(scan['total']):>10}  \033[1mtotal\033[0m")

    if scan["children"]:
        print("\n  Biggest things in your home directory:")
        for child in scan["children"]:
            if child["size"] < 1 << 20:
                continue
            print(f"      {human(child['size']):>10}  {child['name'][:52]}")

    if scan["big_files"]:
        print("\n  Individual files over 64 MB:")
        for f in scan["big_files"]:
            print(f"      {human(f['size']):>10}  {_shorten(f['path'])}")

    if scan["caches"]:
        print("\n  Caches you can clear safely:")
        for c in scan["caches"]:
            print(f"      {human(c['size']):>10}  {c['label']}")
        print("        apt clean                 # package archives")
        print("        rm -rf ~/.cache/pip       # pip downloads")

    print("\n  Note: ~/storage/* are links to your shared storage (photos, Downloads).")
    print("  They are not counted here and are not part of Termux's own size.")
    print()
    return 0


def _shorten(path: str, width: int = 58) -> str:
    home = os.path.expanduser("~")
    prefix = os.environ.get("PREFIX", "")
    if prefix and path.startswith(prefix):
        path = "$PREFIX" + path[len(prefix):]
    elif path.startswith(home):
        path = "~" + path[len(home):]
    return path if len(path) <= width else "…" + path[-(width - 1):]


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
    ap.add_argument("--disk", action="store_true",
                    help="show what the app is storing, and what can be reclaimed")
    ap.add_argument("--clean", action="store_true",
                    help="delete orphaned files left behind by interrupted uploads")
    args = ap.parse_args(argv)

    if args.version:
        print(f"Transcribe {server.VERSION}")
        return 0

    if args.check:
        from . import doctor
        return doctor.run(network=args.network)

    if args.disk or args.clean:
        return _disk(clean=args.clean)

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
