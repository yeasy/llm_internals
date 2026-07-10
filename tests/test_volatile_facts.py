from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "appendix" / "a5_volatile_facts.md"
CLAUDE_CHAPTER = ROOT / "13_decoder_models" / "13.3_deepseek_gemini.md"
SCRIPT = ROOT / "check_project_rules.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_project_rules", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VolatileFactsTests(unittest.TestCase):
    def setUp(self):
        self.checker = load_checker()
        self.text = LEDGER.read_text(encoding="utf-8")

    def issues(self, text: str, today: date = date(2026, 7, 10)) -> list[str]:
        return self.checker.check_volatile_facts(LEDGER, text, today=today)

    def test_current_ledger_has_exact_thirty_day_ttl_and_resolved_conflict(self):
        self.assertEqual(self.issues(self.text), [])
        self.assertIn("verified_at=2026-07-10", self.text)
        self.assertIn("expires_at=2026-08-09", self.text)
        self.assertIn("ttl_days=30", self.text)
        self.assertIn("conflict_status=resolved-conflict", self.text)
        self.assertIn("GPT-5.6 Sol", self.text)
        self.assertIn("2026-07-09", self.text)
        for endpoint in ("v1/responses", "v1/chat/completions", "v1/batch"):
            self.assertIn(endpoint, self.text)

    def test_future_verification_date_fails_closed(self):
        changed = self.text.replace(
            "verified_at=2026-07-10", "verified_at=2026-07-11", 1
        ).replace("expires_at=2026-08-09", "expires_at=2026-08-10", 1)
        self.assertTrue(any("future" in issue for issue in self.issues(changed)))

    def test_non_exact_ttl_and_expired_ledger_fail_closed(self):
        changed = self.text.replace("expires_at=2026-08-09", "expires_at=2026-08-08", 1)
        self.assertTrue(
            any("exactly 30 days" in issue for issue in self.issues(changed))
        )
        self.assertTrue(
            any(
                "expired" in issue
                for issue in self.issues(self.text, today=date(2026, 8, 10))
            )
        )

    def test_open_or_unknown_conflict_state_fails_closed(self):
        for status in ("open-conflict", "unknown"):
            changed = self.text.replace(
                "conflict_status=resolved-conflict",
                f"conflict_status={status}",
                1,
            )
            with self.subTest(status=status):
                self.assertTrue(
                    any("conflict" in issue for issue in self.issues(changed))
                )

    def test_claude_access_status_distinguishes_ga_from_limited_access(self):
        chapter = CLAUDE_CHAPTER.read_text(encoding="utf-8")
        self.assertIn("Fable 5 已恢复全球访问", chapter)
        self.assertIn(
            "Mythos 5 仍非普遍可用，仅向 Project Glasswing 获批客户有限开放",
            chapter,
        )
        self.assertIn("于 7 月 1 日恢复", chapter)


if __name__ == "__main__":
    unittest.main()
