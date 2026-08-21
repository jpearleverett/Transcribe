"""Reclaiming disk the app leaked.

An upload is streamed to disk before the job that references it is created. A
process killed in between — routine on Android — leaves the file with nothing
pointing at it, and before this nothing ever looked for it again. A partly
uploaded video can be tens of gigabytes.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-storage-"))

from transcribe import config, jobs as jobs_mod, storage        # noqa: E402

_SAVED = {}
_PATHS = ("HOME", "CONFIG_PATH", "UPLOAD_DIR", "JOB_DIR", "MODEL_DIR", "BIN_DIR")


def setUpModule():
    for attr in _PATHS:
        _SAVED[attr] = getattr(config, attr)
    home = Path(tempfile.mkdtemp(prefix="transcribe-st-"))
    config.HOME = home
    config.CONFIG_PATH = home / "config.json"
    config.UPLOAD_DIR = home / "uploads"
    config.JOB_DIR = home / "jobs"
    config.MODEL_DIR = home / "models"
    config.BIN_DIR = home / "bin"
    config.ensure_dirs()
    config.load(force=True)
    jobs_mod.STORE = jobs_mod.JobStore()


def tearDownModule():
    for attr, value in _SAVED.items():
        setattr(config, attr, value)
    config.load(force=True)
    jobs_mod.STORE = None


def make_upload(name, size, age_hours=0.0):
    path = config.UPLOAD_DIR / name
    path.write_bytes(b"0" * size)
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))
    return path


class OrphanTest(unittest.TestCase):
    def setUp(self):
        for f in config.UPLOAD_DIR.glob("*"):
            f.unlink()
        for job in jobs_mod.store().list():
            jobs_mod.store().delete(job.id)

    def test_a_file_no_job_references_is_an_orphan(self):
        stray = make_upload("1234-abcd-interrupted.mp4", 4096, age_hours=5)
        found = [o["path"] for o in storage.orphans()]
        self.assertIn(stray, found)

    def test_a_file_a_job_references_is_kept(self):
        kept = make_upload("1234-abcd-real.wav", 4096, age_hours=5)
        jobs_mod.store().create(name="real.wav", audio_file=str(kept))
        self.assertEqual(storage.orphans(), [])

    def test_an_extract_output_is_kept(self):
        out = make_upload("extracted.m4a", 4096, age_hours=5)
        jobs_mod.store().create(name="v.mp4", kind="extract", output_file=str(out))
        self.assertEqual(storage.orphans(), [])

    def test_an_upload_still_arriving_is_not_touched(self):
        """A file being written right now has no job yet and must survive."""
        make_upload("in-flight.mp4", 4096, age_hours=0)
        self.assertEqual(storage.orphans(), [],
                         "a fresh file could be an upload in progress")

    def test_reap_frees_the_space_and_reports_it(self):
        make_upload("a.mp4", 10_000, age_hours=5)
        make_upload("b.mp4", 20_000, age_hours=5)
        kept = make_upload("c.wav", 5_000, age_hours=5)
        jobs_mod.store().create(name="c", audio_file=str(kept))

        count, freed = storage.reap()
        self.assertEqual(count, 2)
        self.assertEqual(freed, 30_000)
        self.assertTrue(kept.exists())
        self.assertEqual(storage.orphans(), [])

    def test_deleting_a_job_makes_its_file_reapable(self):
        path = make_upload("gone.wav", 4096, age_hours=5)
        job = jobs_mod.store().create(name="gone", audio_file=str(path))
        self.assertEqual(storage.orphans(), [])
        jobs_mod.store().delete(job.id)
        # delete() removes the file itself, so nothing is left to reap.
        self.assertFalse(path.exists())

    def test_a_job_that_lost_its_file_does_not_crash_the_scan(self):
        jobs_mod.store().create(name="ghost", audio_file="/nonexistent/x.wav")
        make_upload("stray.mp4", 1000, age_hours=5)
        self.assertEqual(len(storage.orphans()), 1)


class RecentSkipTest(unittest.TestCase):
    """A file skipped for being young must be reported, not silently ignored."""

    def setUp(self):
        for f in config.UPLOAD_DIR.glob("*"):
            f.unlink()
        for job in jobs_mod.store().list():
            jobs_mod.store().delete(job.id)

    def test_a_young_orphan_is_listed_as_recent(self):
        make_upload("just-uploaded.mp4", 8000, age_hours=0)
        self.assertEqual(storage.orphans(), [], "too young to reap")
        recent = storage.recent_unreferenced()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["size"], 8000)
        self.assertIn("recent", storage.report())
        self.assertEqual(len(storage.report()["recent"]), 1)

    def test_an_old_orphan_is_not_double_counted(self):
        make_upload("old.mp4", 9000, age_hours=5)
        self.assertEqual(len(storage.orphans()), 1)
        self.assertEqual(storage.recent_unreferenced(), [],
                         "an old file belongs to orphans, not recent")

    def test_a_referenced_young_file_is_neither(self):
        path = make_upload("in-use.wav", 4000, age_hours=0)
        jobs_mod.store().create(name="in-use", audio_file=str(path))
        self.assertEqual(storage.orphans(), [])
        self.assertEqual(storage.recent_unreferenced(), [])


class ReportTest(unittest.TestCase):
    def setUp(self):
        for f in config.UPLOAD_DIR.glob("*"):
            f.unlink()

    def test_report_totals_and_flags_orphans(self):
        make_upload("orphan.mp4", 50_000, age_hours=9)
        data = storage.report()
        self.assertGreaterEqual(data["total"], 50_000)
        self.assertEqual(len(data["orphans"]), 1)
        self.assertEqual(data["orphan_bytes"], 50_000)
        labels = [p["label"] for p in data["parts"]]
        self.assertIn("Uploaded audio", labels)
        self.assertIn("Build tree", labels)

    def test_human_readable_sizes(self):
        self.assertEqual(storage.human(0), "0 B")
        self.assertEqual(storage.human(2048), "2.0 KB")
        self.assertEqual(storage.human(5 * 1024 ** 3), "5.0 GB")
        self.assertIn("GB", storage.human(26 * 1024 ** 3))

    def test_report_survives_a_missing_directory(self):
        import shutil
        shutil.rmtree(config.MODEL_DIR, ignore_errors=True)
        data = storage.report()          # must not raise
        self.assertTrue(data["parts"])
        config.ensure_dirs()


class TermuxScanTest(unittest.TestCase):
    """Accounting for Termux's own footprint, and only its own.

    ~/storage/* are symlinks into shared storage. Following them would walk the
    entire phone and attribute the user's photos and videos to Termux — wrong,
    and slow enough to look hung on a large device.
    """

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="fake-home-"))
        self.shared = Path(tempfile.mkdtemp(prefix="fake-shared-"))
        # A big file that belongs to the user, not to Termux.
        with open(self.shared / "PXL_video.mp4", "wb") as f:
            f.truncate(8 * 1024 ** 3)
        (self.home / "real").mkdir()
        (self.home / "real" / "data.bin").write_bytes(b"0" * 5000)
        self._saved_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)

    def tearDown(self):
        if self._saved_home is not None:
            os.environ["HOME"] = self._saved_home
        import shutil
        shutil.rmtree(self.home, ignore_errors=True)
        shutil.rmtree(self.shared, ignore_errors=True)

    def test_storage_symlinks_are_not_counted(self):
        storage_dir = self.home / "storage"
        storage_dir.mkdir()
        try:
            (storage_dir / "shared").symlink_to(self.shared)
        except OSError:
            self.skipTest("symlinks unavailable")

        scan = storage.scan_termux()
        home_area = next(a for a in scan["areas"] if a["label"].startswith("Home"))
        self.assertLess(home_area["size"], 1 << 20,
                        "the user's 8 GB video must not be attributed to Termux")
        names = [c["name"] for c in scan["children"]]
        self.assertNotIn("storage/", names)

    def test_a_symlink_anywhere_is_not_followed(self):
        """Not just ~/storage — any symlink could leave the tree."""
        try:
            (self.home / "sneaky").symlink_to(self.shared)
        except OSError:
            self.skipTest("symlinks unavailable")
        scan = storage.scan_termux()
        home_area = next(a for a in scan["areas"] if a["label"].startswith("Home"))
        self.assertLess(home_area["size"], 1 << 20)

    def test_real_content_is_counted(self):
        scan = storage.scan_termux()
        home_area = next(a for a in scan["areas"] if a["label"].startswith("Home"))
        self.assertGreaterEqual(home_area["size"], 5000)
        self.assertIn("real/", [c["name"] for c in scan["children"]])

    def test_large_files_are_listed(self):
        big = self.home / "real" / "huge.bin"
        with open(big, "wb") as f:
            f.truncate(200 << 20)
        scan = storage.scan_termux(min_file_bytes=64 << 20)
        self.assertTrue(any(f["path"].endswith("huge.bin") for f in scan["big_files"]))

    def test_unreadable_directories_do_not_break_the_scan(self):
        locked = self.home / "locked"
        locked.mkdir()
        (locked / "x").write_bytes(b"0" * 100)
        os.chmod(locked, 0o000)
        try:
            scan = storage.scan_termux()      # must not raise
            self.assertTrue(scan["areas"])
        finally:
            os.chmod(locked, 0o755)


if __name__ == "__main__":
    unittest.main(verbosity=2)
