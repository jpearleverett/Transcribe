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


if __name__ == "__main__":
    unittest.main(verbosity=2)
