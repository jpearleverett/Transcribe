"""Browsing the device's own media, so huge files never need uploading.

A 2-hour 4K video can be tens of gigabytes. Sending that through the browser
would have Chrome read all of it and the server write a second copy to the same
phone — needing twice the space to accomplish nothing, since ffmpeg can read the
original where it already sits and write out only the audio.

Exposing a filesystem over HTTP needs care even on loopback, so this module is
deliberately narrow: browsing is confined to a fixed set of media roots, every
path is resolved and re-checked against them, and only directories and media
files are ever listed.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config

# Extensions worth showing. Video included: the point is extracting from it.
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".3gpp",
              ".mts", ".m2ts", ".ts", ".wmv", ".flv", ".mpg", ".mpeg", ".ogv"}
from .audio import AUDIO_EXTS                                   # noqa: E402

MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

# Where Android puts things, in the order a person is most likely to look.
CANDIDATE_ROOTS = [
    ("~/storage/movies", "Movies"),
    ("~/storage/dcim", "Camera"),
    ("~/storage/downloads", "Downloads"),
    ("~/storage/music", "Music"),
    ("~/storage/shared", "Internal storage"),
    ("/sdcard", "Internal storage"),
    ("~/storage/external-1", "SD card"),
    ("~", "Termux home"),
]


def roots() -> list:
    """The directories browsing is allowed to start from.

    Termux only gets at shared storage after `termux-setup-storage`, so most of
    these simply will not exist until then; we list what is really there.
    """
    extra = config.load().get("media_roots") or []
    found, seen = [], set()
    for raw, label in [(p, p) for p in extra] + CANDIDATE_ROOTS:
        try:
            path = Path(os.path.expanduser(raw)).resolve()
        except OSError:
            continue
        if not path.is_dir() or str(path) in seen:
            continue
        # Skip a root that merely duplicates one already listed.
        if any(_is_within(path, Path(r["path"])) for r in found):
            continue
        seen.add(str(path))
        found.append({"path": str(path), "label": label if label != raw else path.name})
    return found


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def is_allowed(path: Path) -> bool:
    """True when `path` sits inside one of the media roots.

    Checked against the *resolved* path, so a symlink pointing outside the roots
    cannot be used to read arbitrary files.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(_is_within(resolved, Path(r["path"])) for r in roots())


def listing(path: str = "") -> dict:
    """Directories and media files in `path`, or the roots when empty."""
    if not path:
        return {"path": "", "parent": None, "roots": roots(), "entries": []}

    target = Path(os.path.expanduser(path))
    if not is_allowed(target):
        raise PermissionError("That folder is outside the media folders.")
    target = target.resolve()
    if not target.is_dir():
        raise NotADirectoryError("Not a folder.")

    dirs, files = [], []
    try:
        for entry in os.scandir(target):
            try:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    dirs.append({"name": entry.name, "path": entry.path, "dir": True})
                elif entry.is_file(follow_symlinks=False):
                    ext = Path(entry.name).suffix.lower()
                    if ext not in MEDIA_EXTS:
                        continue
                    stat = entry.stat()
                    files.append({
                        "name": entry.name,
                        "path": entry.path,
                        "dir": False,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "video": ext in VIDEO_EXTS,
                    })
            except OSError:
                continue        # unreadable entry; skip rather than fail the listing
    except PermissionError:
        raise PermissionError(
            "Android has not granted Termux access to that folder. "
            "Run  termux-setup-storage  and allow the permission.")

    dirs.sort(key=lambda e: e["name"].lower())
    files.sort(key=lambda e: -e["modified"])

    parent = str(target.parent) if is_allowed(target.parent) and target.parent != target else None
    return {"path": str(target), "parent": parent, "roots": roots(),
            "entries": dirs + files}


def resolve_media(path: str) -> Path:
    """Validate a user-supplied media path, or raise."""
    target = Path(os.path.expanduser(path))
    if not is_allowed(target):
        raise PermissionError("That file is outside the media folders.")
    target = target.resolve()
    if not target.is_file():
        raise FileNotFoundError("That file no longer exists.")
    if target.suffix.lower() not in MEDIA_EXTS:
        raise ValueError(f"{target.suffix or 'That file type'} isn't audio or video.")
    return target


def output_dir() -> Path:
    """Where extracted audio is written — somewhere the user can find it."""
    configured = (config.load().get("extract_dir") or "").strip()
    if configured:
        p = Path(os.path.expanduser(configured))
        p.mkdir(parents=True, exist_ok=True)
        return p
    for candidate in ("~/storage/downloads", "~/storage/shared/Download"):
        p = Path(os.path.expanduser(candidate))
        if p.is_dir():
            return p
    p = config.HOME / "extracted"
    p.mkdir(parents=True, exist_ok=True)
    return p


def free_bytes(path: Path) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0
