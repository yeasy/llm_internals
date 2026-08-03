from __future__ import annotations

import importlib.util
import sys
import re
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

    def _stamped(self) -> date:
        """The ledger's own verified_at.

        Previously this suite hardcoded 2026-07-22 both as the default `today`
        and as an asserted literal, so every honest ledger refresh broke three
        tests at once. Reading the date out of the ledger keeps the contract
        (TTL exactness, fail-closed on future/expired) while letting the ledger
        be re-verified without editing the tests.
        """
        found = re.search(r"verified_at=(\d{4})-(\d{2})-(\d{2})", self.text)
        assert found is not None, "ledger must carry verified_at"
        return date(*(int(g) for g in found.groups()))

    def issues(self, text: str, today: date | None = None) -> list[str]:
        return self.checker.check_volatile_facts(
            LEDGER, text, today=today or self._stamped()
        )

    def test_prose_dates_match_the_machine_readable_metadata(self):
        """The ledger states its dates twice; only one of them is enforced.

        `check_volatile_facts` parses the `volatile-facts` HTML comment, so a
        stale date in the human-readable line silently disagrees with the gate.
        That is exactly what happened in 83549c5, which advanced the prose to
        2026-07-30 while leaving the comment at 2026-07-28 — the checker stayed
        green because it never reads the prose. Assert the two agree.
        """
        comment = re.search(
            r"verified_at=(\d{4}-\d{2}-\d{2})\s+expires_at=(\d{4}-\d{2}-\d{2})",
            self.text,
        )
        self.assertIsNotNone(comment, "ledger must carry the volatile-facts comment")
        prose = re.search(
            r"核验日期：(\d{4}-\d{2}-\d{2})；到期日期：(\d{4}-\d{2}-\d{2})", self.text
        )
        self.assertIsNotNone(prose, "ledger must state its dates in prose too")
        self.assertEqual(
            (prose.group(1), prose.group(2)),
            (comment.group(1), comment.group(2)),
            "prose dates drifted from the machine-readable verified_at/expires_at",
        )

    def test_current_ledger_has_exact_thirty_day_ttl_and_resolved_conflict(self):
        self.assertEqual(self.issues(self.text), [])
        self.assertIn("verified_at=2026-07-28", self.text)
        self.assertIn("expires_at=2026-08-27", self.text)
        self.assertIn("`claude-opus-5`", self.text)
        self.assertNotIn("与 Opus 4.8（Opus 档）", self.text)
        self.assertIn("ttl_days=30", self.text)
        self.assertIn("conflict_status=resolved-conflict", self.text)
        self.assertIn("GPT-5.6 Sol", self.text)
        self.assertIn("2026-07-09", self.text)
        for endpoint in ("v1/responses", "v1/chat/completions", "v1/batch"):
            self.assertIn(endpoint, self.text)

    def test_future_verification_date_fails_closed(self):
        changed = self.text.replace(
            "verified_at=2026-07-28", "verified_at=2026-07-29", 1
        ).replace("expires_at=2026-08-27", "expires_at=2026-08-28", 1)
        self.assertTrue(any("future" in issue for issue in self.issues(changed)))

    def test_non_exact_ttl_and_expired_ledger_fail_closed(self):
        changed = self.text.replace("expires_at=2026-08-27", "expires_at=2026-08-26", 1)
        self.assertTrue(
            any("exactly 30 days" in issue for issue in self.issues(changed))
        )
        self.assertTrue(
            any(
                "expired" in issue
                for issue in self.issues(self.text, today=date(2026, 8, 28))
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
