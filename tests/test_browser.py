"""Drives the real app in a real Chromium, emulating an Android phone.

Optional: skipped unless Playwright and a Chromium build are present, so the
default test run stays dependency-free. Run it with:

    pip install playwright && playwright install chromium
    python3 -m unittest tests.test_browser

Everything else in this suite tests the server. This tests what you actually
touch — and it is what caught `.menu { display: flex }` overriding the `hidden`
attribute, which left both menus permanently open on screen.
"""

import math
import os
import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["TRANSCRIBE_TEST"] = "1"
os.environ.setdefault("TRANSCRIBE_HOME", tempfile.mkdtemp(prefix="transcribe-browser-"))

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False


def find_chromium():
    """Prefer an explicit path, then the usual Playwright locations."""
    explicit = os.environ.get("CHROMIUM_PATH")
    if explicit and Path(explicit).exists():
        return explicit
    roots = [Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")),
             Path.home() / ".cache" / "ms-playwright"]
    for root in roots:
        if not root or not root.exists():
            continue
        for candidate in sorted(root.glob("chromium-*/chrome-linux/chrome")):
            return str(candidate)
        for candidate in sorted(root.glob("chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium")):
            return str(candidate)
    return None


CHROMIUM = find_chromium() if HAVE_PLAYWRIGHT else None


@unittest.skipUnless(HAVE_PLAYWRIGHT and CHROMIUM,
                     "needs playwright and a chromium build")
class BrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        from transcribe import config, jobs as jobs_mod, runner, server
        from transcribe.engines import registry            # noqa: F401

        # A real key in the environment would change which engines are ready.
        for name in list(config.ENV_KEYS.values()) + ["RUNPOD_ENDPOINT_ID"]:
            os.environ.pop(name, None)
        config.load(force=True)
        config.ensure_dirs()
        jobs_mod.store().set_runner(runner.run_job)

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

        cls.audio = Path(os.environ["TRANSCRIBE_HOME"]) / "board meeting.wav"
        with wave.open(str(cls.audio), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(
                struct.pack("<h", int(8000 * math.sin(i / 13.0))) for i in range(16000 * 6)))

        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.httpd.shutdown()

    def setUp(self):
        self.ctx = self.browser.new_context(**self.pw.devices["Pixel 7"])
        self.page = self.ctx.new_page()
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console",
                     lambda m: self.errors.append(f"console: {m.text}") if m.type == "error" else None)
        self.page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")

    def tearDown(self):
        self.ctx.close()

    def upload(self):
        self.page.select_option("#engineSelect", "mock")
        self.page.wait_for_timeout(200)
        self.page.set_input_files("#fileInput", str(self.audio))
        self.page.wait_for_selector("#toolbar:not([hidden])", timeout=30000)
        self.page.wait_for_selector(".turn-body", timeout=15000)

    # ---------------- tests ----------------

    def test_boots_clean(self):
        self.assertEqual(self.page.title(), "Transcribe")
        self.assertGreaterEqual(self.page.locator("#engineSelect option").count(), 5)
        self.assertEqual(self.errors, [])

    def test_hidden_elements_are_actually_hidden(self):
        """Regression: a component `display` rule outranks the hidden attribute."""
        for sel in ("#detailView", "#player", "#toolbar", "#exportMenu",
                    "#moreMenu", "#backBtn", "#uploadProgress"):
            self.assertFalse(self.page.locator(sel).is_visible(),
                             f"{sel} should be hidden on the list view")

    def test_upload_is_refused_without_a_key(self):
        # Choosing an engine persists it as the default, so pick one needing a
        # key explicitly rather than relying on whatever a previous test left.
        self.page.select_option("#engineSelect", "deepgram")
        self.page.wait_for_timeout(200)
        self.page.set_input_files("#fileInput", str(self.audio))
        self.page.wait_for_timeout(800)
        self.assertFalse(self.page.locator("#detailView").is_visible(),
                         "an engine with no key must not start a job")

    def test_full_transcript_flow(self):
        self.upload()
        self.assertGreaterEqual(self.page.locator(".turn-body").count(), 2)
        chips = self.page.locator(".chip").all_inner_texts()
        self.assertEqual(len(chips), 2, chips)
        self.assertTrue(any("%" in c for c in chips))
        self.assertIn("Hello everyone", self.page.locator(".turn-body").first.inner_text())
        self.assertGreater(self.page.locator(".w").count(), 5, "word spans drive highlighting")
        self.assertTrue(self.page.locator("#player").is_visible())
        self.assertIn("/audio", self.page.eval_on_selector("#audio", "el => el.src"))
        self.assertEqual(self.errors, [])

    def test_only_one_menu_opens_at_a_time(self):
        self.upload()
        self.page.click("#exportBtn")
        self.page.wait_for_timeout(200)
        self.assertTrue(self.page.locator("#exportMenu").is_visible())
        self.assertFalse(self.page.locator("#moreMenu").is_visible(),
                         "the other menu must not overlap and swallow clicks")
        self.page.click("#moreBtn")
        self.page.wait_for_timeout(200)
        self.assertFalse(self.page.locator("#exportMenu").is_visible())
        self.assertTrue(self.page.locator("#moreMenu").is_visible())

    def test_export_downloads(self):
        self.upload()
        self.page.click("#exportBtn")
        self.page.wait_for_selector("#exportMenu:not([hidden])", timeout=5000)
        with self.page.expect_download(timeout=15000) as dl:
            self.page.click("#exportMenu button[data-fmt='srt']")
        body = Path(dl.value.path()).read_text()
        self.assertIn("-->", body)
        self.assertIn("[Speaker", body)
        self.assertTrue(body.strip().startswith("1"))

    def test_search_filters(self):
        self.upload()
        total = self.page.locator(".turn-body").count()
        self.page.fill("#searchInput", "revenue")
        self.page.wait_for_timeout(400)
        shown = self.page.locator(".turn-body").count()
        self.assertTrue(0 < shown < total, f"{shown} of {total}")

    def test_tap_to_seek_does_not_error(self):
        self.upload()
        self.page.locator(".turn-time").first.click()
        self.page.wait_for_timeout(400)
        self.assertEqual(self.errors, [])

    def test_settings_never_exposes_a_key(self):
        self.page.click("#settingsBtn")
        self.page.wait_for_timeout(500)
        self.assertTrue(self.page.locator("#settingsDialog").evaluate("d => d.open"))
        html = self.page.locator("#keyFields").inner_html()
        self.assertIn("password", html)
        for marker in ("rpa_", "sk-", "hf_"):
            self.assertNotIn(marker, html)

    def test_transcript_survives_a_reload(self):
        """The server owns the state, so a discarded tab loses nothing.

        This is the whole point of the architecture: Android can kill Chrome
        mid-job and reopening the URL must land back on the same transcript.
        """
        self.upload()
        text_before = self.page.locator(".turn-body").first.inner_text()
        self.assertIn("#", self.page.url, "the open transcript should be deep-linkable")

        self.page.reload(wait_until="networkidle")
        self.page.wait_for_selector(".turn-body", timeout=15000)
        self.assertEqual(self.page.locator(".turn-body").first.inner_text(), text_before)
        self.assertTrue(self.page.locator("#toolbar").is_visible())
        self.assertEqual(self.errors, [])

    def test_back_returns_to_a_populated_list(self):
        before = self.page.locator(".job").count()
        self.upload()
        self.page.click("#backBtn")
        self.page.wait_for_selector("#listView:not([hidden])", timeout=5000)
        self.page.wait_for_timeout(300)
        self.assertEqual(self.page.locator(".job").count(), before + 1)
        self.assertFalse(self.page.locator("#detailView").is_visible())


if __name__ == "__main__":
    unittest.main(verbosity=2)
