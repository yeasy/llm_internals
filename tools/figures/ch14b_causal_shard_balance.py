"""生成 14.7 节插图：因果掩码下序列分片的负载不均，以及 2P 分块配平的效果。

正文位置：14_future_trends/14.7_long_context.md
输出：14_future_trends/_images/ch14b_causal_shard_balance.png

把序列切成 2P = 8 个等长块，格子 (i, j) 表示“第 i 块的查询读第 j 块的键”。
因果掩码下只有 j <= i 的格子要算，对角块只算一半。每格计一个单位，即 (N/2P)^2 对。

左：朴素连续分片，第 k 张卡拿第 2k、2k+1 块，工作量 2、6、10、14，最大是均值的 1.75 倍。
右：Llama 3 论文 3.3.2 节的做法，把序列切成 2×CP 块，第 i 张卡同时拿第 i 块与第
(2×CP-1-i) 块（"each CP rank receives two chunks for better load balancing"），
四张卡各 8 个单位，完全配平。数值由 ch14b_numbers.py 复核。

四张卡用四种颜色区分，每张卡的工作量另以数字标出，颜色不单独承担含义。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, INK, MUTED, NEUTRAL_EDGE,
                      NEUTRAL_FACE, NEW_EDGE, NEW_FACE, WEIGHT_EDGE, WEIGHT_FACE,
                      finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14b_causal_shard_balance.png")

P = 4
NB = 2 * P           # 块数
FS, FS_S = 10.0, 9.0
RANK_COLORS = [(DATA_FACE, DATA_EDGE), (WEIGHT_FACE, WEIGHT_EDGE),
               (NEW_FACE, NEW_EDGE), (NEUTRAL_FACE, NEUTRAL_EDGE)]

NAIVE = [i // 2 for i in range(NB)]                       # 0,0,1,1,2,2,3,3
ZIGZAG = [i if i < P else NB - 1 - i for i in range(NB)]   # 0,1,2,3,3,2,1,0


def panel(ax, owner, x0, title):
    """画一个 8x8 的因果块图，行按所属卡上色；返回每张卡的工作量。"""
    work = [0.0] * P
    for i in range(NB):
        face, edge = RANK_COLORS[owner[i]]
        for j in range(NB):
            if j > i:
                continue
            y = -i
            ax.add_patch(Rectangle((x0 + j, y - 1), 1, 1, facecolor=face,
                                   edgecolor=edge, linewidth=0.9, zorder=2))
            if j == i:      # 对角块只算下三角的一半
                ax.plot([x0 + j, x0 + j + 1], [y - 1, y], color=edge, lw=0.9, zorder=3)
                ax.add_patch(Rectangle((x0 + j, y - 1), 1, 1, facecolor="white",
                                       edgecolor="none", alpha=0.55, zorder=2.5))
                work[owner[i]] += 0.5
            else:
                work[owner[i]] += 1.0
        label(ax, x0 - 0.45, -i - 0.5, f"块 {i}", fontsize=FS_S, color=MUTED, ha="right")
        label(ax, x0 + NB + 0.35, -i - 0.5, f"卡 {owner[i]}", fontsize=FS_S,
              color=RANK_COLORS[owner[i]][1], ha="left")
    label(ax, x0 + NB / 2, 1.35, title, fontsize=FS, bold=True)
    label(ax, x0 + NB / 2, 0.55, "键所在的块 →", fontsize=FS_S, color=MUTED)
    return work


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    w1 = panel(ax, NAIVE, 0.0, "朴素连续分片")
    w2 = panel(ax, ZIGZAG, 13.0, "切成 2P 块后首尾配对")

    for x0, w in ((0.0, w1), (13.0, w2)):
        txt = "  ".join(f"卡 {k} 算 {v:g}" for k, v in enumerate(w))
        label(ax, x0 + NB / 2, -NB - 0.75, txt, fontsize=FS_S, color=INK)
        ratio = max(w) / (sum(w) / P)
        label(ax, x0 + NB / 2, -NB - 1.5,
              f"合计 {sum(w):g} 个单位，最忙的一张是均值的 {ratio:.2f} 倍",
              fontsize=FS_S, color=ACCENT if ratio > 1.01 else NEW_EDGE)

    label(ax, 10.5, -NB / 2, "同一条\n序列\n换一种\n分法",
          fontsize=FS_S, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-1.6, 22.6), ylim=(-NB - 2.1, 1.9))


if __name__ == "__main__":
    main()
