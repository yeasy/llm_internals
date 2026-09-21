"""生成图 4-4：ALiBi 的 8 个头各自把注意力压平在多远之外。

正文位置：04_position_encoding/4.4_alibi_others.md
输出：04_position_encoding/_images/ch04_alibi_decay.png

ALiBi 给第 h 个头的分数加上 -r_h*(m-n)。Softmax 之前加一个负偏置，等价于把该键
的权重乘上 exp(-r_h*(m-n))。把这个乘子对距离画出来，就能看到每个头实际上是一扇
多宽的软窗：斜率 1/2 的头十几个词元之外已经乘上 1e-4，最平缓的 1/256 头到两千
词元之外也只剩 1e-4。斜率取自 ALiBi 论文的几何级数 r_h = 2^(-8h/n)。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import ACCENT, DATA_EDGE, INK, MUTED, NEW_EDGE, WEIGHT_EDGE
from _style import image_path, use_cjk_font

OUTPUT = image_path("04_position_encoding", "ch04_alibi_decay.png")

N_HEAD = 8
COLORS = [DATA_EDGE, "#5b9bde", WEIGHT_EDGE, "#f09a6f",
          NEW_EDGE, "#5cb99a", ACCENT, "#8a7fd0"]


def main() -> None:
    use_cjk_font()
    dist = np.arange(1, 16385)
    fig, ax = plt.subplots(figsize=(9, 4.4))

    for h in range(1, N_HEAD + 1):
        r = 2.0 ** (-h)
        ax.loglog(dist, np.exp(-r * dist), color=COLORS[h - 1], lw=1.8,
                  label=f"$r$ = 1/{2 ** h}")

    ax.axhline(1e-2, color=MUTED, ls="--", lw=1.2)
    ax.text(1.3, 1.35e-2, "权重乘子降到 1%", color=MUTED, fontsize=10)

    # 标出每个头降到 1% 的距离：d = ln(100)/r
    offsets = {1: (0.62, 2.0e-4), 4: (0.62, 2.0e-4), 8: (2.4, 1.2e-3)}
    for h in (1, 4, 8):
        r = 2.0 ** (-h)
        d = np.log(100) / r
        fx, fy = offsets[h]
        ax.plot([d], [1e-2], "o", color=COLORS[h - 1], ms=6)
        ax.annotate(f"{d:,.0f}", xy=(d, 1e-2), xytext=(d * fx, fy),
                    color=COLORS[h - 1], fontsize=10,
                    arrowprops=dict(arrowstyle="->", color=COLORS[h - 1], lw=1.1))

    ax.set_xlabel("查询与键的距离 m - n（词元）")
    ax.set_ylabel("权重乘子 exp(-r × 距离)")
    ax.set_title("8 个头的 ALiBi 偏置：每个头是一扇多宽的软窗", fontsize=12, color=INK)
    ax.set_xlim(1, 16384)
    ax.set_ylim(1e-8, 2)
    # 中文字体缺 U+2212，负指数的默认刻度标签会掉字形，这里改用 ASCII 减号自绘。
    ax.set_yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8])
    ax.set_yticklabels(["1", "1e-2", "1e-4", "1e-6", "1e-8"])
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")
    for h in range(1, N_HEAD + 1):
        r = 2.0 ** (-h)
        print(f"  r = 1/{2 ** h:<4d} 降到 1% 的距离 {np.log(100) / r:8,.0f}；"
              f"距离 2,048 处乘子 {np.exp(-r * 2048):.3e}")


if __name__ == "__main__":
    main()
