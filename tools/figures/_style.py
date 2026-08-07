"""插图脚本共用的中文字体与输出路径设置。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]

# 按 macOS / Windows / Linux 的常见中文字体依次挑选第一个可用的。
_CJK_CANDIDATES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)


def use_cjk_font() -> str | None:
    """让 matplotlib 用一款可用的中文字体，返回选中的字体名（找不到时返回 None）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


def image_path(chapter_dir: str, filename: str) -> Path:
    """返回某章 `_images/` 下的输出路径，并确保目录存在。"""
    target = ROOT / chapter_dir / "_images" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
