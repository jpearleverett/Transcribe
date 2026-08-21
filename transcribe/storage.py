"""Accounting for the disk the app uses, and reclaiming what it should not.

One path leaks by construction: an upload is streamed to disk *before* the job
record that points at it exists. If the process dies in between — and on Android
processes do die — the file survives with nothing referencing it, and nothing
ever looks for it again. A partly-uploaded video can sit there for good.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from . import config, jobs as jobs_mod

# An upload in flight is younger than this and must not be touched. Generous:
# a multi-gigabyte file over loopback still takes minutes.
ORPHAN_MIN_AGE = 3600.0


def _tree_size(path: Path) -> tuple:
    total, count = 0, 0
    if not path.exists():
        return 0, 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                continue
    return total, count


def referenced_files() -> set:
    """Every file a job currently points at."""
    keep = set()
    for job in jobs_mod.store().list():
        for value in (job.audio_file, job.output_file):
            if value:
                try:
                    keep.add(str(Path(value).resolve()))
                except OSError:
                    keep.add(value)
    return keep


def recent_unreferenced(min_age: float = ORPHAN_MIN_AGE) -> list:
    """Unreferenced files too young to reap — possibly still being uploaded.

    Reported rather than silently skipped: a user who just saw a huge file in
    --disk and then gets "nothing to clean up" would reasonably conclude the
    tool is broken.
    """
    keep = referenced_files()
    now = time.time()
    found = []
    for path in config.UPLOAD_DIR.glob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if str(path.resolve()) in keep or now - stat.st_mtime >= min_age:
            continue
        found.append({"path": path, "size": stat.st_size, "age": now - stat.st_mtime})
    return sorted(found, key=lambda e: -e["size"])


def orphans(min_age: float = ORPHAN_MIN_AGE) -> list:
    """Files in the upload directory that no job references.

    Age-gated rather than reference-counted against in-flight requests: an
    upload that is still arriving has no job record yet and would otherwise
    look exactly like an orphan.
    """
    keep = referenced_files()
    now = time.time()
    found = []
    for path in config.UPLOAD_DIR.glob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if str(path.resolve()) in keep:
            continue
        if now - stat.st_mtime < min_age:
            continue        # possibly still being written
        found.append({"path": path, "size": stat.st_size, "age": now - stat.st_mtime})
    return sorted(found, key=lambda e: -e["size"])


def reap(min_age: float = ORPHAN_MIN_AGE) -> tuple:
    """Delete orphaned uploads. Returns (count, bytes freed)."""
    freed = count = 0
    for entry in orphans(min_age):
        try:
            entry["path"].unlink()
            freed += entry["size"]
            count += 1
        except OSError:
            continue
    return count, freed


def report() -> dict:
    """Where the app's disk has gone."""
    parts = []
    for label, path, note in (
        ("Uploaded audio", config.UPLOAD_DIR, "originals kept so the player can seek"),
        ("Transcripts", config.JOB_DIR, "job records and results"),
        ("Speech models", config.MODEL_DIR, "offline engine models"),
        ("Build tree", config.HOME / "build", "source and objects from ./install.sh --local"),
        ("Binaries", config.BIN_DIR, "whisper.cpp"),
    ):
        size, count = _tree_size(path)
        parts.append({"label": label, "path": str(path), "size": size,
                      "files": count, "note": note})

    stray = orphans()
    young = recent_unreferenced()
    total = sum(p["size"] for p in parts)
    return {
        "parts": parts,
        "total": total,
        "orphans": [{"name": o["path"].name, "size": o["size"], "age": o["age"]}
                    for o in stray],
        "orphan_bytes": sum(o["size"] for o in stray),
        "recent": [{"name": o["path"].name, "size": o["size"], "age": o["age"]}
                   for o in young],
        "home": str(config.HOME),
    }


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------
# What is Termux itself holding?
# --------------------------------------------------------------------------

# ~/storage/* are symlinks into shared storage. Following them would walk the
# whole phone and attribute the user's photos and videos to Termux, which is
# both wrong and slow.
SKIP_DIR_NAMES = {"storage"}


def _walk_size(path: Path, skip_names=SKIP_DIR_NAMES) -> tuple:
    """(bytes, files) under `path`, never following symlinks out of it."""
    total = files = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue        # never follow: may leave the tree
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in skip_names:
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except (OSError, PermissionError):
            continue
    return total, files


def scan_termux(top: int = 12, min_file_bytes: int = 64 << 20) -> dict:
    """Where Termux's own footprint has gone.

    Answers the question the Android settings screen raises but cannot: the
    figure it shows covers the whole Termux install — its Linux userland, its
    package caches, and everything under the home directory.
    """
    home = Path(os.path.expanduser("~"))
    prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))

    areas = []
    for label, path in (("Home (~)", home), ("Linux packages ($PREFIX)", prefix)):
        if not path.is_dir():
            continue
        size, files = _walk_size(path)
        areas.append({"label": label, "path": str(path), "size": size, "files": files})

    # The biggest immediate children of home, which is where user data lives.
    children = []
    try:
        with os.scandir(home) as it:
            for entry in it:
                try:
                    if entry.is_symlink() or entry.name in SKIP_DIR_NAMES:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        size, files = _walk_size(Path(entry.path))
                        children.append({"name": entry.name + "/", "path": entry.path,
                                         "size": size, "files": files})
                    elif entry.is_file(follow_symlinks=False):
                        children.append({"name": entry.name, "path": entry.path,
                                         "size": entry.stat().st_size, "files": 1})
                except OSError:
                    continue
    except OSError:
        pass
    children.sort(key=lambda e: -e["size"])

    # Individual files big enough to matter on their own.
    big = []
    for root in (home, prefix):
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name not in SKIP_DIR_NAMES:
                                    stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                size = entry.stat(follow_symlinks=False).st_size
                                if size >= min_file_bytes:
                                    big.append({"path": entry.path, "size": size})
                        except OSError:
                            continue
            except (OSError, PermissionError):
                continue
    big.sort(key=lambda e: -e["size"])

    caches = []
    for label, rel in (("apt package archives", "var/cache/apt/archives"),
                       ("pip cache", None)):
        path = (prefix / rel) if rel else (home / ".cache" / "pip")
        if path.is_dir():
            size, files = _walk_size(path)
            if size:
                caches.append({"label": label, "path": str(path),
                               "size": size, "files": files})

    return {"areas": areas, "children": children[:top], "big_files": big[:top],
            "caches": caches, "total": sum(a["size"] for a in areas)}
