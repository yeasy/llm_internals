"""生成 9.3 节的图：同一组 logits 在三种温度下的分布，以及 Top-p 与 Min-p 各截在哪里。

正文位置：09_decoding/9.3_sampling.md
输出：09_decoding/_images/ch09_temperature_truncation.png

logits 固定为 [2, 1, 0, -1, -2]；Top-p 取 0.9，Min-p 取 0.1。数值与正文表 9-6 一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, INK, MASK_EDGE, MASK_FACE,
                      MUTED)
from _style import image_path, use_cjk_font

OUTPUT = image_path("09_decoding", "ch09_temperature_truncation.png")

LOGITS = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
TOP_P, MIN_P = 0.9, 0.1


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    use_cjk_font()
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.9), sharey=True)
    for ax, t in zip(axes, (0.5, 1.0, 2.0)):
        p = softmax(LOGITS / t)
        m = int(np.searchsorted(np.cumsum(p), TOP_P) + 1)        # Top-p 保留的个数
        entropy = float(-(p * np.log2(p)).sum())
        thr = MIN_P * p.max()
        for i, v in enumerate(p):
            keep = i < m
            ax.bar(i + 1, v, width=0.72, color=DATA_FACE if keep else MASK_FACE,
                   edgecolor=DATA_EDGE if keep else MASK_EDGE, linewidth=1.3, zorder=2)
            y_text = max(v, thr) + 0.02 if abs(v - thr) < 0.09 else v + 0.02   # 让数字避开阈值虚线
            ax.text(i + 1, y_text, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                    color=INK if keep else MUTED)
        ax.axhline(thr, color=ACCENT, lw=1.4, linestyle=(0, (4, 2)), zorder=3)
        n_minp = int((p >= thr).sum())
        ax.text(5.5, 0.93, f"Min-p 阈值 0.1 × {p.max():.3f} = {thr:.3f}\n保留 {n_minp} 个", ha="right",
                va="top", fontsize=9.5, color=ACCENT, linespacing=1.4)
        ax.set_title(f"T = {t:g}    熵 {entropy:.3f} bit\nTop-p(0.9) 保留 {m} 个", fontsize=FS_SMALL)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("词元（按 logit 从高到低）", fontsize=FS_SMALL)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0.4, 5.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9.5)
    axes[0].set_ylabel("概率", fontsize=FS_SMALL)
    fig.suptitle("logits = [2, 1, 0, -1, -2]：蓝色柱在 Top-p(0.9) 的核内，灰色柱被截掉；紫色虚线是 Min-p(0.1) 的阈值",
                 fontsize=FS_TEXT - 0.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
