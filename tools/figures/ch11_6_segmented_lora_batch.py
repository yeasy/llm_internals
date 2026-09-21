"""生成 11.6 节的示意图：同一批里不同适配器怎样一起算。

正文位置：11_serving/11.6_multi_lora_serving.md
输出：11_serving/_images/ch11_6_segmented_lora_batch.png

一批 8 个 Decode 词元，分属 3 个秩不同的适配器（a: r=16，b: r=8，c: r=64），
另有 1 个词元不用适配器。基座的 X × W 对整批只做一次；低秩增量按段分别算，
中间结果的宽度等于各自的秩，不必填充到最大秩。形状按 Llama 3 8B 的 W_Q 标注。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_OP, FS_SMALL, FS_TEXT, INK, MUTED,
                      arrow, box, dots, finish, grid, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_6_segmented_lora_batch.png")

ROW_H = 0.8
X_LEFT, X_TOP = 1.5, 11.2
# (适配器名, 秩, 起始行, 行数)
SEGMENTS = (("a", 16, 0, 3), ("b", 8, 3, 2), ("c", 64, 5, 2))
V_WIDTH = {8: 1.5, 16: 1.9, 64: 2.6}


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10, 6.6))

    label(ax, 12.5, 15.6, "同一批 8 个词元、3 个适配器：基座只乘一次，增量按段各算各的",
          fontsize=FS_NAME, bold=True)

    # 左侧：输入矩阵 X[8, 4096]，每行一个词元
    values = [[None, None, None] for _ in range(8)]
    right, bottom = grid(ax, X_LEFT, X_TOP, values, cell_w=ROW_H, cell_h=ROW_H)
    for i in range(8):
        dots(ax, X_LEFT + 1.5 * ROW_H, X_TOP - (i + 0.5) * ROW_H, spread=0.2, r=0.05,
             color=MUTED)
    label(ax, (X_LEFT + right) / 2, bottom - 0.45, "X[8, 4096]", fontsize=FS_TEXT)
    names = ["a", "a", "a", "b", "b", "c", "c", "无"]
    for i, n in enumerate(names):
        label(ax, X_LEFT - 0.5, X_TOP - (i + 0.5) * ROW_H, n, fontsize=FS_TEXT,
              color=ACCENT if n != "无" else MUTED, bold=n != "无")
    label(ax, X_LEFT - 0.5, X_TOP + 0.45, "适配器", fontsize=FS_SMALL, color=MUTED)

    # 上方一路：基座权重，整批一次矩阵乘
    base_y = 12.6
    ax.plot([X_LEFT + 1.2, X_LEFT + 1.2], [X_TOP, base_y + 0.8], color=INK, lw=1.5)
    arrow(ax, (X_LEFT + 1.2, base_y + 0.8), (4.3, base_y + 0.8))
    label(ax, 4.7, base_y + 0.8, "×", fontsize=FS_OP)
    box(ax, 5.2, base_y, 9.6, 1.6, "W[4096, 4096]\n基座权重，整批共用，一次矩阵乘",
        kind="weight", fontsize=FS_TEXT)
    arrow(ax, (14.9, base_y + 0.8), (15.5, base_y + 0.8))
    box(ax, 15.6, base_y, 2.7, 1.6, "XW\n[8, 4096]", kind="data", fontsize=FS_SMALL)

    # 下方一路：按适配器分段
    for name, rank, start, n_rows in SEGMENTS:
        top = X_TOP - start * ROW_H
        h = n_rows * ROW_H
        y0, yc = top - h + 0.08, top - h / 2
        bh = h - 0.16
        # 段的括线
        ax.plot([right + 0.15, right + 0.3, right + 0.3, right + 0.15],
                [top - 0.08, top - 0.08, top - h + 0.08, top - h + 0.08],
                color=ACCENT, lw=1.4)
        label(ax, 4.7, yc, "×", fontsize=FS_OP)
        box(ax, 5.2, y0, 3.0, bh, f"$A_{name}$\n[4096, {rank}]", kind="weight",
            fontsize=FS_SMALL)
        arrow(ax, (8.3, yc), (8.8, yc))
        vw = V_WIDTH[rank]
        box(ax, 8.9, y0, vw, bh, f"[{n_rows}, {rank}]", kind="data", fontsize=FS_SMALL)
        label(ax, 12.1, yc, "×", fontsize=FS_OP)
        box(ax, 12.6, y0, 3.0, bh, f"$B_{name}$\n[{rank}, 4096]", kind="weight",
            fontsize=FS_SMALL)
        arrow(ax, (15.7, yc), (16.2, yc))
        box(ax, 16.3, y0, 2.0, bh, f"[{n_rows},\n4096]", kind="data", fontsize=FS_SMALL)
        arrow(ax, (18.4, yc), (19.3, yc))

    # 第 8 行：不用适配器，跳过
    y_skip = X_TOP - 7 * ROW_H
    box(ax, 5.2, y_skip - ROW_H + 0.08, 13.1, ROW_H - 0.16,
        "不用适配器的词元：这一路跳过，增量记为 0", kind="neutral", fontsize=FS_SMALL,
        color=MUTED)
    arrow(ax, (18.4, y_skip - ROW_H / 2), (19.3, y_skip - ROW_H / 2), color=MUTED)

    # 右侧：逐行相加
    box(ax, 19.4, bottom, 5.2, base_y + 1.6 - bottom,
        "逐行相加\n\nY = XW + 增量\n\nY[8, 4096]", kind="data", fontsize=FS_TEXT, lw=1.8)
    arrow(ax, (18.4, base_y + 0.8), (19.3, base_y + 0.8))

    # 底部：内核需要的索引
    idx_top = 3.4
    label(ax, 0.4, idx_top - 0.4, "每个词元的适配器号", fontsize=FS_SMALL, ha="left")
    grid(ax, 5.6, idx_top, [names], kind="plain", cell_w=0.8, cell_h=0.8, fontsize=FS_SMALL)
    label(ax, 12.8, idx_top - 0.4, "各段边界", fontsize=FS_SMALL, ha="left")
    grid(ax, 15.2, idx_top, [[0, 3, 5, 7]], kind="plain", cell_w=0.8, cell_h=0.8,
         fontsize=FS_SMALL)
    label(ax, 19.2, idx_top - 0.4, "各段词元数", fontsize=FS_SMALL, ha="left")
    grid(ax, 22.1, idx_top, [[3, 2, 2]], kind="plain", cell_w=0.8, cell_h=0.8,
         fontsize=FS_SMALL)
    label(ax, 12.5, 1.35,
          "中间结果的宽度就是各自的秩（16、8、64）：分段计算不必把 a、b 填充到 64。\n"
          "同号词元排在一起，段内是一次小矩阵乘；三段由同一次内核调用并行完成。\n"
          "各段边界是 n + 1 个数：前 n 个是各段起点，末位 7 是末段的终点。",
          fontsize=FS_SMALL, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(0, 25), ylim=(0.4, 16.2))


if __name__ == "__main__":
    main()
