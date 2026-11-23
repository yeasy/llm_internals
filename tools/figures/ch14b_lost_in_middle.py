"""生成 14.7 节插图：多文档问答的准确率随答案文档位置的变化（Lost in the Middle 实测）。

正文位置：14_future_trends/14.7_long_context.md
输出：14_future_trends/_images/ch14b_lost_in_middle.png

数据取自 Liu 等人 Lost in the Middle（arXiv:2307.03172v3）附录 G.2 表 6：
20 篇检索文档（约 4K 词元），含答案的文档放在第 0、4、9、14、19 位。
虚线是 GPT-3.5-Turbo 的闭卷准确率（不给任何文档），取自同文表 1。
四个模型用四种颜色加四种标记区分，每条线末端另有文字标签，颜色不单独承担含义。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE, WEIGHT_EDGE, ACCENT
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14b_lost_in_middle.png")

POSITIONS = [1, 5, 10, 15, 20]  # 论文的 Index 0/4/9/14/19，按第几篇文档计
SERIES = [
    ("GPT-3.5-Turbo", [75.8, 57.2, 53.8, 55.4, 63.2], DATA_EDGE, "o"),
    ("LongChat-13B (16K)", [68.6, 57.4, 55.3, 52.5, 55.0], WEIGHT_EDGE, "s"),
    ("Claude-1.3", [59.9, 55.9, 56.8, 57.2, 60.1], NEW_EDGE, "^"),
    ("MPT-30B-Instruct", [53.7, 51.8, 52.2, 52.7, 56.3], ACCENT, "D"),
]
CLOSED_BOOK_GPT35 = 56.1
LABEL_NUDGE = {"MPT-30B-Instruct": 1.1, "LongChat-13B (16K)": -1.1}  # 两条线末端相距仅 1.3 个点


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for name, acc, color, marker in SERIES:
        ax.plot(POSITIONS, acc, color=color, marker=marker, lw=2.0, ms=7, zorder=3)
        ax.text(POSITIONS[-1] + 0.6, acc[-1] + LABEL_NUDGE.get(name, 0.0), name, color=color,
                fontsize=FS_SMALL, va="center", ha="left")

    ax.hlines(CLOSED_BOOK_GPT35, 0, POSITIONS[-1] + 0.3, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate("GPT-3.5-Turbo 闭卷（不给文档）56.1%\n答案放在第 10 篇时反而低于闭卷",
                xy=(12.5, CLOSED_BOOK_GPT35), xytext=(9.0, 70.5), color=MUTED,
                fontsize=FS_SMALL, ha="left", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))

    ax.set_xticks(POSITIONS)
    ax.set_xlim(0, 27)
    ax.set_ylim(48, 80)
    ax.set_xlabel("含答案的文档在 20 篇中的位置", fontsize=FS_TEXT, color=INK)
    ax.set_ylabel("准确率（%）", fontsize=FS_TEXT, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e4e1", lw=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
