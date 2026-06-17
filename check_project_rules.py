#!/usr/bin/env python3
"""Lightweight Markdown project checks for book repositories."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {
    ".agent",
    ".git",
    ".mdpress",
    "_book",
    "_site",
    "dist",
    "node_modules",
}
LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+(?:\s+\"[^\"]*\")?)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def strip_fenced_blocks(text: str) -> str:
    output: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_len = 0
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            char = marker[0]
            length = len(marker)
            if not in_fence:
                in_fence = True
                fence_marker = char
                fence_len = length
            elif char == fence_marker and length >= fence_len:
                in_fence = False
            output.append("")
            continue
        output.append("" if in_fence else line)
    return "\n".join(output)


def check_fences(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    stack: list[tuple[str, int, int]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        match = FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group(1)
        char = marker[0]
        length = len(marker)
        if not stack:
            stack.append((char, length, line_no))
            continue
        open_char, open_len, _ = stack[-1]
        if char == open_char and length >= open_len:
            stack.pop()
        else:
            stack.append((char, length, line_no))
    for _, _, line_no in stack:
        issues.append(f"{path.relative_to(ROOT)}:{line_no}: unclosed fenced code block")
    return issues


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def is_local_target(target: str) -> bool:
    parsed = urlparse(target)
    return not parsed.scheme and not parsed.netloc


def normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if " " in target and target.count('"') >= 2:
        target = target.split(" ", 1)[0]
    return unquote(target)


def split_target(target: str) -> tuple[str, str]:
    if "#" not in target:
        return target, ""
    path_part, anchor = target.split("#", 1)
    return path_part, anchor


def slugify_heading(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.strip().lower()
    chars: list[str] = []
    previous_dash = False
    for char in text:
        if char.isalnum() or char in "\u4e00-\u9fff":
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-")


def heading_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in strip_fenced_blocks(text).splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        anchor = slugify_heading(match.group(2))
        if not anchor:
            continue
        count = counts.get(anchor, 0)
        counts[anchor] = count + 1
        anchors.add(anchor if count == 0 else f"{anchor}-{count}")
    return anchors


def check_links(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    body = strip_fenced_blocks(text)
    for match in LINK_RE.finditer(body):
        raw_target = match.group(2).strip()
        target = normalize_target(raw_target)
        if not target or not is_local_target(raw_target):
            continue
        target_name, anchor = split_target(target)
        target_path = path if not target_name else (path.parent / target_name).resolve()
        line_no = body[: match.start()].count("\n") + 1
        try:
            target_path.relative_to(ROOT)
        except ValueError:
            issues.append(f"{rel(path)}:{line_no}: local link escapes book root: {raw_target}")
            continue
        if not target_path.exists():
            issues.append(f"{rel(path)}:{line_no}: missing local link target: {raw_target}")
            continue
        if anchor and target_path.suffix == ".md":
            target_text = target_path.read_text(encoding="utf-8", errors="ignore")
            if anchor not in heading_anchors(target_text):
                issues.append(f"{rel(path)}:{line_no}: missing local link anchor: {raw_target}")
    return issues


def expected_first_heading(path: Path) -> int | None:
    name = path.name.lower()
    if path.parent == ROOT and name in {"readme.md", "summary.md"}:
        return None
    if name == "readme.md":
        return 1
    if name == "summary.md":
        return 2
    return 2


def check_headings(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    headings: list[tuple[int, int]] = []
    body = strip_fenced_blocks(text)
    for line_no, line in enumerate(body.splitlines(), 1):
        match = HEADING_RE.match(line)
        if match:
            headings.append((line_no, len(match.group(1))))
    if not headings:
        return issues
    expected = expected_first_heading(path)
    if expected is not None and headings[0][1] != expected:
        if expected == 2:
            issues.append(f"{rel(path)}:{headings[0][0]}: section files must start with level-2 heading")
        else:
            issues.append(f"{rel(path)}:{headings[0][0]}: README files must start with level-1 heading")
    previous_level = headings[0][1]
    for line_no, level in headings[1:]:
        if level > previous_level + 1:
            issues.append(
                f"{rel(path)}:{line_no}: heading level jumps from H{previous_level} to H{level}"
            )
        previous_level = level
    return issues


def check_summary_links() -> list[str]:
    summary = ROOT / "SUMMARY.md"
    if not summary.exists():
        return []
    return check_links(summary, summary.read_text(encoding="utf-8", errors="ignore"))


def summary_markdown_targets(summary_text: str) -> set[str]:
    targets: set[str] = set()
    for match in LINK_RE.finditer(strip_fenced_blocks(summary_text)):
        raw_target = match.group(2).strip()
        target = normalize_target(raw_target)
        if not target or not is_local_target(raw_target):
            continue
        target_name, _ = split_target(target)
        if target_name.endswith(".md"):
            targets.add(target_name)
    return targets


def check_summary_coverage(summary_text: str, files: list[Path]) -> list[str]:
    issues: list[str] = []
    targets = summary_markdown_targets(summary_text)
    for path in files:
        relative = rel(path)
        if relative == "SUMMARY.md":
            continue
        if relative not in targets:
            issues.append(f"SUMMARY.md: missing Markdown file from summary: {relative}")
    for target in sorted(targets):
        target_path = (ROOT / target).resolve()
        try:
            target_path.relative_to(ROOT)
        except ValueError:
            issues.append(f"SUMMARY.md: local link escapes book root: {target}")
            continue
        if not target_path.exists():
            issues.append(f"SUMMARY.md: missing local link target: {target}")
    return issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight Markdown checks for this book."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show scanned Markdown files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    issues: list[str] = []
    files = iter_markdown_files()
    if args.verbose:
        print("Scanned Markdown files:")
        for path in files:
            print(f"- {path.relative_to(ROOT)}")
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        issues.extend(check_fences(path, text))
        issues.extend(check_headings(path, text))
        issues.extend(check_links(path, text))
    issues.extend(check_summary_links())
    summary = ROOT / "SUMMARY.md"
    if summary.exists():
        issues.extend(
            check_summary_coverage(
                summary.read_text(encoding="utf-8", errors="ignore"), files
            )
        )

    if issues:
        print("\n".join(sorted(set(issues))))
        print(f"\n{len(set(issues))} issue(s) found across {len(files)} Markdown files.")
        return 1
    print(f"All {len(files)} Markdown files passed project checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
