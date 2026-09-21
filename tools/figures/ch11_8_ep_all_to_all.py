"""生成 11.8 节的图：专家并行里一次 dispatch 的收发矩阵与负载不均。

正文位置：11_serving/11.8_multi_gpu_inference.md
输出：11_serving/_images/ch11_8_ep_all_to_all.png

示意设定：4 张卡，每张卡上有 4 个词元，每个词元选 2 个专家，所以每张卡要发出 8 份
词元副本。矩阵第 i 行第 j 列是“卡 i 发给卡 j 的份数”：行和恒为 8，列和是卡 j 上的
专家这一层要处理的份数。对角线留在本卡，不走网络。数值为手工设定，用来说明机制：
all-to-all 里每一对（源，目的）的数据量都不同，而且每一步都在变；一层的耗时由列和最大
的那张卡决定。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_8_ep_all_to_all.png")

SEND = [[3, 1, 3, 1],
        [4, 2, 1, 1],
        [3, 2, 2, 1],
        [4, 1, 2, 1]]
N = len(SEND)
CW, CH = 1.7, 1.15


def main() -> None:
    use_cjk_font()
    row_sums = [sum(r) for r in SEND]
    col_sums = [sum(SEND[i][j] for i in range(N)) for j in range(N)]
    local = sum(SEND[i][i] for i in range(N))
    total = sum(row_sums)
    mean = total / N

    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    label(ax, 0.2, 9.6, "dispatch：每张卡把词元副本发给专家所在的卡", ha="left",
          fontsize=FS_TITLE, bold=True)

    x0, top = 4.4, 7.6
    label(ax, x0 + N * CW / 2, top + 1.05, "目的卡（专家所在的卡）", fontsize=FS_SMALL, color=MUTED)
    for j in range(N):
        label(ax, x0 + (j + 0.5) * CW, top + 0.4, f"卡 {j}", fontsize=FS_TEXT)
    label(ax, x0 + (N + 0.75) * CW, top + 0.4, "行和", fontsize=FS_TEXT, color=MUTED)
    for i in range(N):
        y = top - (i + 1) * CH
        label(ax, x0 - 0.25, y + CH / 2, f"源：卡 {i}", ha="right", fontsize=FS_TEXT)
        for j in range(N):
            kind = "neutral" if i == j else "data"
            box(ax, x0 + j * CW, y, CW, CH, str(SEND[i][j]), kind, fontsize=FS_NAME,
                rounded=False, lw=1.0)
        label(ax, x0 + (N + 0.75) * CW, y + CH / 2, str(row_sums[i]), fontsize=FS_NAME,
              color=MUTED)
    y_sum = top - (N + 1) * CH
    label(ax, x0 - 0.25, y_sum + CH / 2, "列和", ha="right", fontsize=FS_TEXT, color=ACCENT)
    for j in range(N):
        label(ax, x0 + (j + 0.5) * CW, y_sum + CH / 2, str(col_sums[j]), fontsize=FS_NAME,
              color=ACCENT, bold=True)
    label(ax, 0.2, y_sum - 0.75,
          f"灰色对角线留在本卡，共 {local} 份；其余 {total - local} 份要走网络。",
          ha="left", fontsize=FS_SMALL, color=INK)

    # 右侧：各卡要处理的份数
    bx, by, unit = 16.2, y_sum + CH + 0.1, 0.36
    label(ax, bx - 0.4, 9.6, "各卡上的专家要处理的份数（列和）", ha="left",
          fontsize=FS_TITLE, bold=True)
    bar_w, gap = 1.5, 0.9
    for j, v in enumerate(col_sums):
        x = bx + j * (bar_w + gap)
        kind = "new" if v == max(col_sums) else "data"
        box(ax, x, by, bar_w, v * unit, "", kind, rounded=False, lw=1.2)
        ax.text(x + bar_w / 2, by + v * unit + 0.45, str(v), ha="center", va="center",
                fontsize=FS_NAME, color=ACCENT if v == max(col_sums) else INK,
                fontweight="bold" if v == max(col_sums) else "normal", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        label(ax, x + bar_w / 2, by - 0.45, f"卡 {j}", fontsize=FS_TEXT)
    x_end = bx + N * (bar_w + gap) - gap
    ax.plot([bx - 0.3, x_end + 0.3], [by + mean * unit, by + mean * unit], color=MUTED,
            lw=1.2, ls=(0, (4, 3)), zorder=5)
    label(ax, x_end + 0.45, by + mean * unit, f"均衡时\n每卡 {mean:.0f} 份", ha="left",
          fontsize=FS_SMALL, color=MUTED)
    label(ax, bx - 0.4, y_sum - 0.75,
          f"这一层要等卡 0 算完 {max(col_sums)} 份，工作量是均衡时的 {max(col_sums) / mean:.2f} 倍。",
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    arrow(ax, (x0 + (N + 1.4) * CW, top - 2 * CH), (bx - 0.9, top - 2 * CH), color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(-0.2, 28.6), ylim=(y_sum - 1.4, 10.3))


if __name__ == "__main__":
    main()
