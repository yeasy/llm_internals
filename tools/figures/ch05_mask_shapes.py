"""生成 5.3 节插图：三种注意力掩码形状。

正文位置：05_pretraining/5.3_encoder_decoder.md
输出：05_pretraining/_images/ch05_mask_shapes.png

同一张 T x T 的可见性方阵，三种范式只差掩码形状：
全可见（编码器）、因果（仅解码器）、前缀（前缀语言模型，前 4 个位置互相可见）。
深色格表示“可读”，浅色格表示被置为 -inf。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import (DATA_EDGE, DATA_STRONG, FS_NAME, FS_SMALL, INK, MASK_FACE,
                      MUTED, NEUTRAL_EDGE)
from _style import image_path, use_cjk_font

OUTPUT = image_path("05_pretraining", "ch05_mask_shapes.png")

T = 8
PREFIX = 4


def masks() -> list[tuple[str, np.ndarray, str]]:
    idx = np.arange(T)
    causal = (idx[:, None] >= idx[None, :]).astype(float)
    full = np.ones((T, T))
    prefix = causal.copy()
    prefix[:PREFIX, :PREFIX] = 1.0
    return [
        ("全可见（编码器）", full, "BERT、T5 的编码器一侧"),
        ("因果（仅解码器）", causal, "GPT 式，每个位置只读自己和左边"),
        (f"前缀（前 {PREFIX} 个位置互相可见）", prefix, "前缀语言模型、UL2"),
    ]


def main() -> None:
    use_cjk_font()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9))
    for ax, (title, m, note) in zip(axes, masks()):
        for i in range(T):
            for j in range(T):
                face = DATA_STRONG if m[i, j] else MASK_FACE
                ax.add_patch(plt.Rectangle((j, T - i - 1), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1.2))
        ax.add_patch(plt.Rectangle((0, 0), T, T, fill=False, edgecolor=DATA_EDGE,
                                   linewidth=1.6))
        if title.startswith("前缀"):
            ax.add_patch(plt.Rectangle((0, T - PREFIX), PREFIX, PREFIX, fill=False,
                                       edgecolor=NEUTRAL_EDGE, linewidth=2.0,
                                       linestyle=(0, (4, 2))))
        ax.set_xlim(-1.1, T + 0.6)
        ax.set_ylim(-1.5, T + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(T / 2, T + 0.95, title, ha="center", va="center", fontsize=FS_NAME,
                color=INK)
        ax.text(T / 2, -0.85, note, ha="center", va="center", fontsize=FS_SMALL,
                color=MUTED)
        ax.text(T / 2, T + 0.28, "列：被读取的位置", ha="center", va="center",
                fontsize=FS_SMALL, color=MUTED)
        ax.text(-0.75, T / 2, "行：当前位置", ha="center", va="center", rotation=90,
                fontsize=FS_SMALL, color=MUTED)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
