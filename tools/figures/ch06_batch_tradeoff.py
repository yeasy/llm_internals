"""生成 6.4 节的图：批量在「步数」与「词元数」之间的取舍曲线。

正文位置：06_training_techniques/6.4_batch_sequence.md
输出：06_training_techniques/_images/ch06_batch_tradeoff.png

曲线是 McCandlish 等人的取舍式 (S/S_min - 1)(E/E_min - 1) = 1，
参数化为 B/B_crit = r 时 S/S_min = 1 + 1/r、E/E_min = 1 + r。
横轴是达到同一损失所需的优化步数（串行时间），纵轴是所需的词元数（算力）。
图上标出 r = 1/4、1、4、16 四个点：越过临界批量后，步数几乎不再下降，
词元数却按 r 线性上涨。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("06_training_techniques", "ch06_batch_tradeoff.png")

MARKS = [
    (0.25, "B = B_crit / 4\n步数 5 倍，词元 1.25 倍", (3.35, 2.15), NEW_EDGE),
    (1.0, "B = B_crit\n步数 2 倍，词元 2 倍", (2.55, 3.45), ACCENT),
    (4.0, "B = 4 B_crit\n步数 1.25 倍，词元 5 倍", (2.05, 5.55), WEIGHT_EDGE),
]


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    rs = [0.06 + 0.01 * k for k in range(3000)]
    xs = [1 + 1 / r for r in rs]
    ys = [1 + r for r in rs]
    ax.plot(xs, ys, color=DATA_EDGE, linewidth=2.0, zorder=2)

    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.2)
    ax.axvline(1.0, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(3.15, 0.74, "词元数的下界 E_min（批量趋近 0）", fontsize=10, color=MUTED)
    ax.text(0.86, 6.2, "步数的下界 S_min（批量趋于无穷）", fontsize=10,
            color=MUTED, rotation=90, va="top")

    for r, text, textpos, color in MARKS:
        x, y = 1 + 1 / r, 1 + r
        ax.plot([x], [y], marker="o", markersize=8, color=color, zorder=4)
        ax.annotate(text, xy=(x, y), xytext=textpos, fontsize=10, color=color,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))

    ax.set_xlabel("达到同一损失所需的优化步数  S / S_min（串行时间）",
                  fontsize=11, color=INK)
    ax.set_ylabel("所需的词元数  E / E_min（算力）", fontsize=11, color=INK)
    ax.set_xlim(0.72, 6.2)
    ax.set_ylim(0.6, 6.4)
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")
    for r, _, _, _ in MARKS:
        print(f"  B/B_crit = {r}: S/S_min = {1 + 1 / r:.4g}, E/E_min = {1 + r:.4g}")


if __name__ == "__main__":
    main()
