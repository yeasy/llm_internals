"""生成 5.5 节插图：一条公开数据管线的步骤顺序与逐步保留量。

正文位置：05_pretraining/5.5_data_pipeline.md
输出：05_pretraining/_images/ch05_data_pipeline.png

数值全部取自 FineWeb 论文（arXiv 2406.17557）对 96 个 Common Crawl 快照的自报口径：
基础过滤后约 36T 词元（GPT-2 分词器计）；按快照分别做 MinHash 近似去重后剩 20T；
再叠加 C4 过滤子集与自定义启发式过滤，得到 15T 的 FineWeb。
右下那一条是论文的反例：改成跨全部快照的全局去重只剩 4T，且留下来的数据更差。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT, INK,
                      MASK_EDGE, MASK_FACE, MUTED, NEW_EDGE, NEW_FACE, arrow, box,
                      finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("05_pretraining", "ch05_data_pipeline.png")

SCALE = 0.30  # 每万亿词元多宽
BAR_H = 0.78
STEPS = [
    ("URL 过滤 + 正文抽取 + 语言识别 + 启发式过滤", 36.0, "data"),
    ("按快照分别做 MinHash 近似去重", 20.0, "new"),
    ("C4 过滤子集 + 自定义启发式过滤", 15.0, "new"),
]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.4, 4.4))

    x0 = 0.0
    y = 3.4
    prev = None
    for name, tok, kind in STEPS:
        w = tok * SCALE
        box(ax, x0, y, w, BAR_H, f"{tok:.0f}T 词元", kind=kind, fontsize=FS_TEXT)
        label(ax, x0 - 0.25, y + BAR_H / 2, name, fontsize=FS_SMALL, color=INK, ha="right")
        if prev is not None:
            label(ax, w + 0.3, y + BAR_H / 2, f"保留 {tok / prev:.0%}", fontsize=FS_SMALL,
                  color=ACCENT, ha="left")
        prev = tok
        y -= 1.25

    # 反例：全局去重
    y_bad = y - 0.55
    w_bad = 4.0 * SCALE
    box(ax, x0, y_bad, w_bad, BAR_H, "4T", kind="neutral", fontsize=FS_TEXT)
    label(ax, x0 - 0.25, y_bad + BAR_H / 2, "（反例）跨全部快照做全局去重",
          fontsize=FS_SMALL, color=MUTED, ha="right")
    label(ax, w_bad + 0.3, y_bad + BAR_H / 2,
          "只保留 11%，而且留下来的数据\n经训练检验比被删掉的更差",
          fontsize=FS_SMALL, color=MASK_EDGE, ha="left")

    label(ax, -8.5, y_bad + BAR_H + 0.44, "同一份 36T，换一种去重口径：",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    label(ax, 36 * SCALE / 2, 3.4 + BAR_H + 0.42,
          "起点：96 个 Common Crawl 快照的原始 WARC 文件",
          fontsize=FS_SMALL, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-8.6, 36 * SCALE + 5.6),
           ylim=(y_bad - 0.55, 3.4 + BAR_H + 0.9))


if __name__ == "__main__":
    main()
