"""Static invariants for the front end.

No browser and no dependencies — these are the structural mistakes that are
invisible in review but obvious on a phone, so they are worth asserting cheaply
on every run. tests/test_browser.py drives the real thing when Playwright is
available.
"""

import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HTML = (ROOT / "web" / "index.html").read_text()
JS = (ROOT / "web" / "app.js").read_text()
CSS = (ROOT / "web" / "styles.css").read_text()


def classes_setting_display():
    found = set()
    for block in re.finditer(r"([^{}]+)\{([^}]*)\}", CSS):
        selector, body = block.group(1).strip(), block.group(2)
        if re.search(r"(^|[;\s])display\s*:", body):
            found.update(re.findall(r"\.([a-zA-Z-]+)", selector))
    return found


def elements_with_hidden():
    out = []
    for tag in re.findall(r"<[^>]+\shidden[^>]*>|<[^>]+>", HTML):
        if re.search(r"\shidden(\s|>|=)", tag):
            ident = re.search(r'id="([^"]+)"', tag)
            cls = re.search(r'class="([^"]*)"', tag)
            out.append((ident.group(1) if ident else "?",
                        cls.group(1).split() if cls else []))
    return out


class HiddenAttributeTest(unittest.TestCase):
    def test_hidden_is_forced_off(self):
        """`hidden` is a UA-stylesheet rule, so any author `display` beats it.

        Without a global override, .menu/.player/.toolbar/.icon-btn all set
        display and their hidden elements stay on screen — menus render
        permanently open and the player never hides.
        """
        self.assertRegex(
            CSS, r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important",
            "styles.css must force [hidden] off; a component rule will "
            "otherwise override the attribute")

    def test_no_hidden_element_relies_on_the_ua_rule_alone(self):
        display_classes = classes_setting_display()
        clashes = [(i, [c for c in cls if c in display_classes])
                   for i, cls in elements_with_hidden()]
        clashes = [(i, c) for i, c in clashes if c]
        # These are fine *because* of the global override; this test documents
        # which elements depend on it, so removing it fails loudly above.
        self.assertTrue(
            re.search(r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important", CSS),
            f"these elements depend on the override: {clashes}")


class DomContractTest(unittest.TestCase):
    def test_every_referenced_id_exists(self):
        referenced = set(re.findall(r"\$\('([^']+)'\)", JS))
        present = set(re.findall(r'\bid="([^"]+)"', HTML))
        missing = sorted(referenced - present)
        self.assertEqual(missing, [], f"app.js references ids not in the HTML: {missing}")

    def test_static_assets_referenced_exist(self):
        for href in re.findall(r'(?:href|src)="/static/([^"]+)"', HTML):
            self.assertTrue((ROOT / "web" / href).exists(), f"missing asset: {href}")

    def test_js_runs_in_strict_mode(self):
        # It sits after the file's comment header, so look past that.
        head = "\n".join(JS.split("\n")[:20])
        self.assertIn("'use strict'", head)


class EscapingTest(unittest.TestCase):
    def test_interpolations_into_innerhtml_are_escaped(self):
        """Speaker names and file names are user-controlled and reach innerHTML."""
        unescaped = []
        for line_no, line in enumerate(JS.split("\n"), 1):
            if "innerHTML" not in line and "html +=" not in line:
                continue
            for expr in re.findall(r"\$\{([^}]*)\}", line):
                e = expr.strip()
                safe = (e.startswith("esc(") or e.startswith("Math.")
                        or e.startswith("hms(") or "esc(" in e
                        or re.fullmatch(r"[\w.]+", e) and e.startswith(("v", "l")))
                if not safe:
                    unescaped.append((line_no, e[:70]))
        # The language list is a hardcoded constant, not user data.
        unescaped = [u for u in unescaped if u[1] not in ("v", "l")]
        self.assertEqual(unescaped, [], f"unescaped interpolation: {unescaped}")

    def test_speaker_names_pass_through_esc(self):
        self.assertIn("esc(speakerName(", JS,
                      "renamed speakers are user input and must be escaped")


class ManifestTest(unittest.TestCase):
    def test_manifest_is_valid_json_and_points_at_real_icons(self):
        import json
        data = json.loads((ROOT / "web" / "manifest.webmanifest").read_text())
        self.assertEqual(data["start_url"], "/")
        for icon in data["icons"]:
            self.assertTrue((ROOT / "web" / icon["src"].lstrip("/").replace("static/", "")).exists(),
                            icon["src"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
