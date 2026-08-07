#!/usr/bin/env python3
"""Lightweight Markdown project checks for book repositories."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
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
# 这些目录下的 Markdown 是工具说明而非书稿章节：不要求出现在 SUMMARY.md 中，
# 但仍参与链接、围栏、标题等全部其余检查。
NON_CHAPTER_DIRS = {"tools", "tests"}
LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+(?:\s+\"[^\"]*\")?)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
VOLATILE_METADATA_RE = re.compile(
    r"<!--\s*volatile-facts:\s*"
    r"verified_at=(\d{4}-\d{2}-\d{2})\s+"
    r"expires_at=(\d{4}-\d{2}-\d{2})\s+"
    r"ttl_days=(\d+)\s+"
    r"conflict_status=([a-z-]+)\s*-->"
)


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
            issues.append(
                f"{rel(path)}:{line_no}: local link escapes book root: {raw_target}"
            )
            continue
        if not target_path.exists():
            issues.append(
                f"{rel(path)}:{line_no}: missing local link target: {raw_target}"
            )
            continue
        if anchor and target_path.suffix == ".md":
            target_text = target_path.read_text(encoding="utf-8", errors="ignore")
            if anchor not in heading_anchors(target_text):
                issues.append(
                    f"{rel(path)}:{line_no}: missing local link anchor: {raw_target}"
                )
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
            issues.append(
                f"{rel(path)}:{headings[0][0]}: section files must start with level-2 heading"
            )
        else:
            issues.append(
                f"{rel(path)}:{headings[0][0]}: README files must start with level-1 heading"
            )
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
        # 工具目录的说明文档不是书的章节，不该出现在目录里；但它仍然参与本文件
        # 其余各项检查（链接、围栏、标题），所以只在这一项上豁免。
        if relative.split("/", 1)[0] in NON_CHAPTER_DIRS:
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


def check_volatile_facts(
    path: Path, text: str, *, today: date | None = None
) -> list[str]:
    """Fail closed when the volatile-facts ledger is stale or ambiguous."""
    issues: list[str] = []
    current_date = today or date.today()
    match = VOLATILE_METADATA_RE.search(text)
    location = rel(path) if path.is_relative_to(ROOT) else str(path)
    if match is None:
        return [f"{location}: missing or malformed volatile-facts metadata"]
    verified_raw, expires_raw, ttl_raw, conflict_status = match.groups()
    try:
        verified = datetime.strptime(verified_raw, "%Y-%m-%d").date()
        expires = datetime.strptime(expires_raw, "%Y-%m-%d").date()
    except ValueError:
        return [f"{location}: invalid volatile-facts date"]
    ttl_days = int(ttl_raw)
    if ttl_days != 30 or expires - verified != timedelta(days=30):
        issues.append(f"{location}: volatile-facts TTL must be exactly 30 days")
    if verified > current_date:
        issues.append(
            f"{location}: volatile-facts verified_at is in the future ({verified_raw})"
        )
    if current_date > expires:
        issues.append(f"{location}: volatile-facts ledger expired on {expires_raw}")
    if conflict_status != "resolved-conflict":
        issues.append(
            f"{location}: volatile-facts conflict state must be resolved-conflict"
        )
    return issues


SECTION_FILE_RE = re.compile(r"^\d+\.\d+")
MIN_INTRO_CHARS = 40

# 关门句：把某项技术说成终局，会让后面的小节显得毫无理由。
# 只在末段命中且同段没有任何限定词时报错——精度优先，宁可漏报。
CLOSING_CLAIM_RE = re.compile(
    r"(标配|事实标准|主流选择|默认选择|已成为|关键技术之一)"
)
HEDGE_RE = re.compile(
    r"(但|然而|不过|除非|前提|边界|代价|局限|仅|只在|尚未|还没|之外|另一方面|随之"
    r"|不能|并非|未必|不再|取决于|强相关|视.{0,4}而定)"
)

# 先用后定义会当场制造无效认知负荷。表里每个术语的定义处由
# tests/test_prereq_terms.py 校验，改表必须改代码，必然过 review。
PREREQ_TERMS = {
    "Pre-Norm": "03_components/3.6_layer_norm.md",
    "TTFT": "11_serving/11.5_best_practices.md",
    "TPOT": "11_serving/11.5_best_practices.md",
    # 书里 11.3 写小写 goodput、11.5 写大写 Goodput，匹配一律大小写不敏感。
    "Goodput": "11_serving/11.5_best_practices.md",
}


def summary_ordered_targets() -> list[str]:
    """SUMMARY.md 里出现的本地 Markdown 目标，保持出现顺序。"""
    summary = ROOT / "SUMMARY.md"
    if not summary.exists():
        return []
    seen: list[str] = []
    body = summary.read_text(encoding="utf-8", errors="ignore")
    for match in LINK_RE.finditer(body):
        raw = match.group(2).strip()
        target = normalize_target(raw)
        if not is_local_target(raw):
            continue
        path_part, _ = split_target(target)
        if path_part.endswith(".md") and path_part not in seen:
            seen.append(path_part)
    return seen


def is_numbered_section(path: Path) -> bool:
    return bool(SECTION_FILE_RE.match(path.name))


def check_section_intro(path: Path, text: str) -> list[str]:
    """编号小节在第一个 H3 之前必须有引入段。

    没有引入段意味着读者从 `## X.Y` 直接掉进 `### X.Y.1`，既拿不到本节的
    自足定位（跳读者），也接不上前一节留下的缺口（顺读者）。
    """
    if not is_numbered_section(path):
        return []
    body = strip_fenced_blocks(text)
    h2 = re.search(r"^##\s+.+$", body, re.MULTILINE)
    h3 = re.search(r"^###\s", body, re.MULTILINE)
    if not h2 or not h3 or h3.start() < h2.end():
        return []
    intro = body[h2.end() : h3.start()].strip()
    # 只算实际散文：跳过引用块标记、注释和空行。
    prose = "\n".join(
        line
        for line in intro.splitlines()
        if line.strip() and not line.lstrip().startswith(("<!--", "|", "```"))
    ).strip()
    if len(prose) >= MIN_INTRO_CHARS:
        return []
    return [
        f"{rel(path)}: 缺少引入段（`## ` 与第一个 `### ` 之间只有 {len(prose)} 字符，"
        f"至少需要 {MIN_INTRO_CHARS}）。第一句给跳读者定位，第二句接上一节留下的缺口。"
    ]


def check_closing_claim(path: Path, text: str) -> list[str]:
    """末段不得用无限定的「标配/事实标准」把门焊死。"""
    if not is_numbered_section(path):
        return []
    body = strip_fenced_blocks(text)
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    last = paragraphs[-1]
    if not CLOSING_CLAIM_RE.search(last) or HEDGE_RE.search(last):
        return []
    claim = CLOSING_CLAIM_RE.search(last).group(1)
    return [
        f"{rel(path)}: 末段用「{claim}」收尾且不带任何限定，"
        f"读者会认为问题已经解决，下一节因此显得没有理由。"
        f"补一条它没覆盖的边界即可。"
    ]


def check_prereq_terms(files: list[Path]) -> list[str]:
    """术语不得在其定义小节之前裸用。

    修法是就地补一句括注（`10_inference_optimization/10.1_bottleneck.md` 里的
    「TPOT（每输出词元时间）」就是标准写法），或在同段给出指向定义处的链接。
    """
    order = summary_ordered_targets()
    position = {target: index for index, target in enumerate(order)}
    issues: list[str] = []
    for path in files:
        if not is_numbered_section(path):
            continue
        here = position.get(rel(path))
        if here is None:
            continue
        body = strip_fenced_blocks(path.read_text(encoding="utf-8", errors="ignore"))
        for term, definition in PREREQ_TERMS.items():
            defined_at = position.get(definition)
            if defined_at is None or here >= defined_at:
                continue
            term_re = re.compile(re.escape(term), re.IGNORECASE)
            # 只看**首次**出现：术语第一次露面时交代清楚即可，
            # 后续复用再逐处括注反而会把正文写成词汇表。
            for paragraph in body.split("\n\n"):
                if not term_re.search(paragraph):
                    continue
                # 就地括注或指向定义处的链接，二者有其一即可。
                glossed = re.search(
                    re.escape(term) + r"\s*[（(]", paragraph, re.IGNORECASE
                )
                linked = Path(definition).name in paragraph
                if not glossed and not linked:
                    issues.append(
                        f"{rel(path)}: 「{term}」在其定义处（{definition}）之前就被裸用，"
                        f"读者此时无法理解。就地补一句括注，或在同段链接到定义处。"
                    )
                break  # 首次出现已判定，无论通过与否都不再往后看
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
        issues.extend(check_section_intro(path, text))
        issues.extend(check_closing_claim(path, text))
        if path == ROOT / "appendix" / "a5_volatile_facts.md":
            issues.extend(check_volatile_facts(path, text))
    issues.extend(check_prereq_terms(files))
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
        print(
            f"\n{len(set(issues))} issue(s) found across {len(files)} Markdown files."
        )
        return 1
    print(f"All {len(files)} Markdown files passed project checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
