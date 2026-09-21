"""生成 8.5 节的示意图：目标域收益与通用能力损失的双曲线，以及停点。

正文位置：08_alignment/8.5_practice.md
输出：08_alignment/_images/ch08_tradeoff_curves.png

数值与正文表格一致，都是教学示意数据，不来自任何公开实验。左轴是目标域
验证集准确率，右轴是通用能力回归集准确率，柱子是每轮的交换比
（回归集每掉几个百分点，换来目标域涨一个百分点）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("08_alignment", "ch08_tradeoff_curves.png")

EPOCH = [0, 1, 2, 3, 4]
TARGET = [35.0, 41.0, 47.5, 50.5, 51.5]
GENERAL = [68.8, 68.2, 67.0, 64.6, 60.6]
RATIO = [0.10, 0.18, 0.80, 4.00]


def main() -> None:
    use_cjk_font()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.6, 5.6), height_ratios=[2.0, 1.0], sharex=True)

    ax.plot(EPOCH, TARGET, "-o", color=NEW_EDGE, lw=2.2, ms=6, label="目标域验证集")
    ax.set_ylabel("目标域准确率 (%)", fontsize=FS_TEXT, color=NEW_EDGE)
    ax.tick_params(axis="y", labelcolor=NEW_EDGE, labelsize=FS_SMALL)
    ax.set_ylim(30, 56)

    ax2 = ax.twinx()
    ax2.plot(EPOCH, GENERAL, "--s", color=DATA_EDGE, lw=2.2, ms=6, label="通用能力回归集")
    ax2.set_ylabel("通用能力准确率 (%)", fontsize=FS_TEXT, color=DATA_EDGE)
    ax2.tick_params(axis="y", labelcolor=DATA_EDGE, labelsize=FS_SMALL)
    ax2.set_ylim(58, 71)

    ax.axvline(2, color=ACCENT, lw=1.4, ls=":")
    ax.annotate("停点：第 3 轮起交换比翻了几倍", xy=(2, 47.5), xytext=(0.25, 52.5),
                fontsize=FS_SMALL, color=ACCENT,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
    ax.set_title("同一次微调，两条曲线一起看", fontsize=FS_NAME, color=INK, pad=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=FS_SMALL, loc="lower right", frameon=False)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    bx.bar(EPOCH[1:], RATIO, width=0.32, color=ACCENT, alpha=0.8)
    for e, r in zip(EPOCH[1:], RATIO):
        bx.text(e, r + 0.12, f"{r:.2f}", ha="center", fontsize=FS_SMALL, color=ACCENT)
    bx.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    bx.text(0.1, 1.15, "交换比 = 1：通用掉 1 点换目标域涨 1 点", fontsize=FS_SMALL, color=MUTED)
    bx.set_ylabel("交换比（点/点）", fontsize=FS_TEXT)
    bx.set_xlabel("训练轮次", fontsize=FS_TEXT)
    bx.set_xticks(EPOCH)
    bx.set_ylim(0, 4.9)
    bx.tick_params(labelsize=FS_SMALL)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
