"""生成 10.3 节插图：FlashAttention 分块前向的循环结构与变量驻留位置。

正文位置：10_inference_optimization/10.3_flash_attention.md
输出：10_inference_optimization/_images/ch10_flash_tiling.png

按 FlashAttention-2 的循环次序画：外层遍历 Q 的行块 i，内层遍历 K/V 的列块 j。
示意取 4 × 4 个块、因果掩码；右上方整块被掩码的格子直接跳过。
蓝 = 数据，青绿 = 当前正在片上计算的块，灰 = 被因果掩码跳过，紫 = 强调。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, DATA_STRONG, FS_TEXT, MASK_EDGE, MASK_FACE,
                      MUTED, NEW_EDGE, NEW_FACE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("10_inference_optimization", "ch10_flash_tiling.png")
FS = 9.5
N = 4
CUR_I, CUR_J = 2, 1          # 当前外层块、内层块（从 0 起）


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    gx, gtop, c = 4.2, 9.6, 1.5          # 网格左上角与格子边长
    # K/V 列块（显存）
    for j in range(N):
        kind = "new" if j == CUR_J else "data"
        box(ax, gx + j * c + 0.08, gtop + 0.35, c - 0.16, 0.9, f"$K_{j + 1}, V_{j + 1}$", kind,
            fontsize=FS, rounded=False)
    label(ax, gx + N * c / 2, gtop + 1.65, "K/V 的列块，每块 [$B_c$, $d_h$]，驻留显存", fontsize=FS)
    # Q 行块（显存）
    for i in range(N):
        kind = "new" if i == CUR_I else "data"
        box(ax, gx - 1.75, gtop - (i + 1) * c + 0.08, 1.45, c - 0.16, f"$Q_{i + 1}$", kind,
            fontsize=FS, rounded=False)
    label(ax, gx - 1.05, gtop + 0.85, "Q 的行块\n每块 [$B_r$, $d_h$]", fontsize=FS)

    # 分数矩阵的块
    for i in range(N):
        for j in range(N):
            x, y = gx + j * c, gtop - (i + 1) * c
            if j > i:
                face, edge, txt = MASK_FACE, MASK_EDGE, "跳过"
            elif (i, j) == (CUR_I, CUR_J):
                face, edge, txt = NEW_FACE, NEW_EDGE, "$S_{32}$\n片上"
            elif i < CUR_I or (i == CUR_I and j < CUR_J):
                face, edge, txt = DATA_STRONG, DATA_EDGE, "已算"
            else:
                face, edge, txt = DATA_FACE, DATA_EDGE, ""
            ax.add_patch(Rectangle((x, y), c, c, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=2))
            if txt:
                ax.text(x + c / 2, y + c / 2, txt, ha="center", va="center", fontsize=FS - 0.5,
                        color=MUTED if txt == "跳过" else "black", zorder=3, linespacing=1.3)
    ax.add_patch(Rectangle((gx, gtop - N * c), N * c, N * c, fill=False, edgecolor=DATA_EDGE,
                           linewidth=1.8, zorder=4))
    label(ax, gx + N * c / 2, gtop - N * c - 0.55, "分数矩阵 [T, T] 只在逻辑上存在，从不整张写入显存",
          fontsize=FS, color=ACCENT)
    label(ax, gx + N * c / 2, gtop - N * c - 1.25, "外层 i 遍历行块，内层 j 遍历列块", fontsize=FS, color=MUTED)

    # 片上 SRAM
    sx, sy, sw, sh = 12.6, 4.2, 7.6, 6.9
    box(ax, sx, sy, sw, sh, "", "neutral")
    label(ax, sx + sw / 2, sy + sh - 0.5, "片上 SRAM：一次只放一对块", fontsize=FS_TEXT, bold=True)
    rows = [
        ("$Q_3$  [$B_r$, $d_h$]", "data"),
        ("$K_2, V_2$  [$B_c$, $d_h$]", "new"),
        ("$S_{32} = Q_3 K_2^{\\top}/\\sqrt{d_h}$  [$B_r$, $B_c$]", "new"),
        ("逐行状态 m、$\\ell$  [$B_r$]", "plain"),
        ("未归一化输出 $\\tilde{O}_3$  [$B_r$, $d_h$]", "plain"),
    ]
    for k, (text, kind) in enumerate(rows):
        box(ax, sx + 0.4, sy + sh - 1.95 - k * 1.12, sw - 0.8, 0.92, text, kind, fontsize=FS)
    arrow(ax, (gx + N * c + 0.1, gtop - (CUR_I + 0.5) * c), (sx, sy + 3.3), color=NEW_EDGE)
    label(ax, 11.45, 7.35, "读入", fontsize=FS, color=NEW_EDGE)

    # 写回
    box(ax, 12.6, 1.0, 7.6, 2.2,
        "内层循环走完后写回显存：\n$O_3 = \\tilde{O}_3 / \\ell$   [$B_r$, $d_h$]\n"
        "$\\mathrm{LSE}_3 = m + \\ln \\ell$   [$B_r$]", "data", fontsize=FS)
    arrow(ax, (sx + sw / 2, sy), (sx + sw / 2, 3.2))

    finish(fig, ax, OUTPUT, xlim=(2.0, 20.6), ylim=(0.6, 11.8))


if __name__ == "__main__":
    main()
