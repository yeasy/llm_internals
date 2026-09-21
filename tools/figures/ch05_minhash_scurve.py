"""生成 5.5 节插图：MinHash-LSH 的分带曲线，以及调 (b, r) 就是调阈值。

正文位置：05_pretraining/5.5_data_pipeline.md
输出：05_pretraining/_images/ch05_minhash_scurve.png

两篇文档的 Jaccard 相似度为 s 时，成为候选对的概率是 1 - (1 - s^r)^b，
其中签名切成 b 个带、每带 r 行。拐点约在 (1/b)^(1/r)。
实线是 FineWeb 公开的配置（112 个哈希切成 14 x 8），虚线是另外两组对照。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED,
                      NEW_EDGE, WEIGHT_EDGE)
from _style import image_path, use_cjk_font

OUTPUT = image_path("05_pretraining", "ch05_minhash_scurve.png")

CONFIGS = [
    (14, 8, DATA_EDGE, "-", "b = 14, r = 8（FineWeb，共 112 个哈希）"),
    (9, 13, NEW_EDGE, "--", "b = 9, r = 13（共 117 个哈希，阈值更严）"),
    (40, 3, WEIGHT_EDGE, ":", "b = 40, r = 3（共 120 个哈希，阈值更松）"),
]
MARKS = (0.70, 0.75, 0.80, 0.85)


def prob(s, b, r):
    return 1 - (1 - s ** r) ** b


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    s = np.linspace(0, 1, 600)
    for b, r, color, ls, name in CONFIGS:
        ax.plot(s, prob(s, b, r), color=color, lw=2.3, ls=ls, label=name)
        thr = (1 / b) ** (1 / r)
        ax.scatter([thr], [prob(thr, b, r)], s=55, color=color, edgecolor="white",
                   linewidth=1.3, zorder=5)
        ax.annotate(f"拐点 {thr:.2f}", xy=(thr, prob(thr, b, r)),
                    xytext=(10, -16), textcoords="offset points", fontsize=FS_SMALL,
                    color=color, ha="left", va="top")

    for m in MARKS:
        p = prob(m, 14, 8)
        ax.plot([m, m], [0, p], color=MUTED, lw=0.8, ls=":")
        ax.text(m, p + 0.035, f"{p:.0%}", fontsize=FS_SMALL, color=DATA_EDGE,
                ha="center", va="bottom")

    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("两篇文档的 n-gram 相似度 s", fontsize=FS_TEXT, color=INK)
    ax.set_ylabel("被判为候选重复对的概率", fontsize=FS_TEXT, color=INK)
    ax.set_title("同样约 120 个哈希，切法不同，阈值就不同", fontsize=FS_NAME, color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.legend(fontsize=FS_SMALL, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=1, handlelength=2.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=FS_SMALL)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
