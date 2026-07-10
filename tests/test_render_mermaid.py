from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "render_mermaid.py"


class RenderMermaidTests(unittest.TestCase):
    def book(self, root: Path, diagrams: int = 1) -> Path:
        book = root / "book"
        book.mkdir()
        (book / "marker").write_text("keep", encoding="utf-8")
        (book / "SUMMARY.md").write_text("* [C](c.md)\n", encoding="utf-8")
        (book / "c.md").write_text(
            "\n".join(
                f"```mermaid\ngraph TD\nA{i}-->B{i}\n```" for i in range(diagrams)
            ),
            encoding="utf-8",
        )
        return book

    def fake_tools(
        self, root: Path, partial: bool = False, max_diagrams: int | None = None
    ) -> tuple[Path, Path]:
        binary = root / "bin"
        binary.mkdir()
        chrome = binary / "chrome"
        chrome.write_text("", encoding="utf-8")
        mmdc = binary / "mmdc"
        count = "1 if 'A0-->B0' in text else 0" if partial else "count"
        limit = (
            f"\n                if count > {max_diagrams}: raise SystemExit(1)"
            if max_diagrams is not None
            else ""
        )
        mmdc.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import pathlib, sys
                args=sys.argv[1:]; src=pathlib.Path(args[args.index('-i')+1]); out=pathlib.Path(args[args.index('-o')+1])
                text=src.read_text(); count=text.count('```mermaid'); count={count}{limit}
                for i in range(1, count+1): out.with_name(f'{{out.stem}}-{{i}}.svg').write_text('<svg></svg>')
                """
            ),
            encoding="utf-8",
        )
        mmdc.chmod(0o755)
        return binary, chrome

    def fake_nonzero_after_write_tools(
        self, root: Path, *, fail_first_only: bool
    ) -> tuple[Path, Path]:
        binary = root / "bin"
        binary.mkdir()
        chrome = binary / "chrome"
        chrome.write_text("", encoding="utf-8")
        attempt = root / "mmdc-attempt"
        mmdc = binary / "mmdc"
        mmdc.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import pathlib, sys
                args=sys.argv[1:]; src=pathlib.Path(args[args.index('-i')+1]); out=pathlib.Path(args[args.index('-o')+1])
                count=src.read_text().count('```mermaid')
                for i in range(1, count+1): out.with_name(f'{{out.stem}}-{{i}}.svg').write_text('<svg></svg>')
                attempt=pathlib.Path({str(attempt)!r})
                attempt_count=int(attempt.read_text()) if attempt.exists() else 0
                attempt.write_text(str(attempt_count + 1))
                if {fail_first_only!r} is False or attempt_count == 0:
                    print('renderer reported fatal error', file=sys.stderr)
                    raise SystemExit(1)
                """
            ),
            encoding="utf-8",
        )
        mmdc.chmod(0o755)
        return binary, chrome

    def run_renderer(self, book: Path, output: Path, env: dict[str, str], *flags: str):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--book-dir",
                str(book),
                "--svg-out",
                str(output),
                *flags,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_output_must_not_overlap_source_and_unrelated_files_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self.book(root)
            result = self.run_renderer(book, book / "svg", os.environ.copy())
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((book / "marker").is_file())
            output = root / "output"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            binary, chrome = self.fake_tools(root)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{binary}{os.pathsep}{env['PATH']}",
                    "CHROME_BIN": str(chrome),
                }
            )
            self.assertEqual(self.run_renderer(book, output, env).returncode, 0)
            self.assertTrue((output / "keep.txt").is_file())

    def test_default_is_strict_and_partial_render_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self.book(root, 2)
            env = os.environ.copy()
            env["CHROME_BIN"] = str(root / "missing")
            self.assertNotEqual(
                self.run_renderer(book, root / "out1", env).returncode, 0
            )
            binary, chrome = self.fake_tools(root, partial=True)
            env.update(
                {
                    "PATH": f"{binary}{os.pathsep}{env['PATH']}",
                    "CHROME_BIN": str(chrome),
                }
            )
            result = self.run_renderer(book, root / "out2", env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed for diagrams", result.stderr)

    def test_default_limits_per_browser_render_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self.book(root, 9)
            binary, chrome = self.fake_tools(root, max_diagrams=4)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{binary}{os.pathsep}{env['PATH']}",
                    "CHROME_BIN": str(chrome),
                }
            )
            result = self.run_renderer(book, root / "out", env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("RENDERED 9/9", result.stdout)

    def test_strict_rejects_complete_output_from_nonzero_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self.book(root)
            binary, chrome = self.fake_nonzero_after_write_tools(
                root, fail_first_only=False
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{binary}{os.pathsep}{env['PATH']}",
                    "CHROME_BIN": str(chrome),
                }
            )
            output = root / "out"
            result = self.run_renderer(book, output, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("renderer reported fatal error", result.stderr)
            self.assertIn("RENDERED 0/1", result.stdout)
            self.assertFalse((output / "d-1.svg").exists())

    def test_nonzero_batch_can_recover_on_successful_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self.book(root)
            binary, chrome = self.fake_nonzero_after_write_tools(
                root, fail_first_only=True
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{binary}{os.pathsep}{env['PATH']}",
                    "CHROME_BIN": str(chrome),
                }
            )
            output = root / "out"
            result = self.run_renderer(book, output, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("renderer reported fatal error", result.stderr)
            self.assertIn("RENDERED 1/1", result.stdout)
            self.assertTrue((output / "d-1.svg").is_file())
            self.assertEqual((root / "mmdc-attempt").read_text(), "2")


if __name__ == "__main__":
    unittest.main()
