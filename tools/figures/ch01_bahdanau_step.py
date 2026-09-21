"""生成 1.3 节插图：Bahdanau 注意力里，解码一步的五个动作与它们的先后顺序。

正文位置：01_introduction/1.3_attention_birth.md
输出：01_introduction/_images/ch01_bahdanau_step.png

关键是时序：打分用的是**上一步**的解码状态 s_{t-1}，算出 c_t 之后才更新出 s_t。
柱状的注意力权重由本脚本里的五个分数经 Softmax 算出，运行时会打印。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_NAME, FS_SMALL, FS_TEXT,
                      INK, MUTED, NEW_EDGE, WEIGHT_EDGE, arrow, box, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("01_introduction", "ch01_bahdanau_step.png")

SCORES = [0.4, 2.4, 1.1, 0.2, 0.3]
EXPS = [math.exp(v) for v in SCORES]
Z = sum(EXPS)
ALPHA = [v / Z for v in EXPS]


def step_marker(ax, x, y, n):
    ax.add_patch(Circle((x, y), 0.32, facecolor=ACCENT, edgecolor="white",
                        linewidth=1.4, zorder=8))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=FS_SMALL,
            color="white", fontweight="bold", zorder=9)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    ax.set_xlim(0, 20.4)
    ax.set_ylim(0.2, 12.7)
    ax.set_aspect("equal")
    ax.axis("off")

    label(ax, 10.2, 12.3, "解码第 t 步的五个动作", fontsize=FS_NAME, color=INK,
          bold=True)

    # ---- 编码器注释：双向 RNN 的前后向拼接，整句只算一次 ----
    hx, hw, hh, gap = 1.0, 1.9, 1.0, 0.35
    label(ax, 1.0, 11.4, "编码器（双向 RNN，整句只算一次，此后固定不动）",
          fontsize=FS_SMALL, color=MUTED, ha="left")
    for j in range(5):
        box(ax, hx + j * (hw + gap), 9.9, hw, hh, f"$h_{j+1}$", kind="data",
            fontsize=FS_TEXT)
    label(ax, 12.4, 10.4,
          "每个 $h_j=[\\vec{h}_j;\\overleftarrow{h}_j]$，形状 $[1,\\,2n]$\n"
          "同时含它左边和右边的上下文",
          fontsize=FS_SMALL, color=MUTED, ha="left", linespacing=1.45)

    # ---- 第 1 步：用 s_{t-1} 与每个 h_j 打分 ----
    box(ax, 1.0, 7.0, 2.6, 1.1, "$s_{t-1}$", kind="data", fontsize=FS_TEXT)
    label(ax, 2.3, 6.68, "上一步的解码状态", fontsize=FS_SMALL, color=MUTED)
    box(ax, 4.6, 6.9, 5.9, 1.4,
        "对齐网络\n$e_{tj}=v_a^{\\top}\\tanh(W_a s_{t-1}+U_a h_j)$",
        kind="weight", fontsize=FS_SMALL)
    arrow(ax, (3.6, 7.55), (4.6, 7.55), color=DATA_EDGE, lw=1.8)
    for j in range(5):
        x = hx + j * (hw + gap) + hw / 2
        arrow(ax, (x, 9.9), (7.55, 8.3), color=DATA_EDGE, lw=1.0, rad=-0.10)
    step_marker(ax, 3.95, 8.55, 1)
    label(ax, 7.55, 6.5, "$U_a h_j$ 与 $t$ 无关，可对全部 $j$ 预先算一次",
          fontsize=FS_SMALL, color=WEIGHT_EDGE)

    # ---- 第 2 步：Softmax 得到本步的权重 ----
    arrow(ax, (10.5, 7.55), (12.2, 7.55), color=INK, lw=1.6)
    step_marker(ax, 11.35, 8.25, 2)
    label(ax, 11.35, 7.1, "Softmax", fontsize=FS_SMALL, color=INK)

    bx, bw, bmax, base = 12.9, 0.66, 2.5, 6.4
    for j, a in enumerate(ALPHA):
        x = bx + j * (bw + 0.30)
        ax.bar(x, a * bmax, width=bw, bottom=base, color=DATA_FACE,
               edgecolor=DATA_EDGE, linewidth=1.2, zorder=3)
        ax.text(x, base + a * bmax + 0.14, f"{a:.2f}", ha="center", va="bottom",
                fontsize=FS_SMALL, color=DATA_EDGE, zorder=4)
        ax.text(x, base - 0.34, f"{j+1}", ha="center", va="center",
                fontsize=FS_SMALL, color=MUTED)
    ax.plot([bx - 0.55, bx + 4 * (bw + 0.30) + 0.55], [base, base], color=MUTED,
            lw=1.0)
    label(ax, bx + 1.9, 9.05, "本步的注意力权重 $\\alpha_{tj}$，和为 1",
          fontsize=FS_SMALL, color=DATA_EDGE)

    # ---- 第 3 步：按权重取走 h_j，得到 c_t ----
    arrow(ax, (14.4, base - 0.7), (13.6, 4.75), color=DATA_EDGE, lw=1.6, rad=0.18)
    step_marker(ax, 14.9, 5.2, 3)
    box(ax, 10.6, 3.7, 3.9, 1.05, "$c_t=\\sum_j \\alpha_{tj} h_j$", kind="new",
        fontsize=FS_TEXT)
    label(ax, 12.55, 3.36, "本步专属的上下文向量，形状 $[1,\\,2n]$",
          fontsize=FS_SMALL, color=NEW_EDGE)

    # ---- 第 4 步：c_t 参与更新解码状态 ----
    arrow(ax, (10.6, 4.22), (8.2, 4.22), color=NEW_EDGE, lw=1.8)
    step_marker(ax, 9.4, 4.85, 4)
    box(ax, 3.5, 3.7, 4.7, 1.05, "$s_t=f(s_{t-1},\\,y_{t-1},\\,c_t)$", kind="data",
        fontsize=FS_TEXT)
    arrow(ax, (2.3, 7.0), (2.3, 4.22), color=DATA_EDGE, lw=1.3)
    arrow(ax, (2.3, 4.22), (3.5, 4.22), color=DATA_EDGE, lw=1.3)

    # ---- 第 5 步：预测输出词 ----
    arrow(ax, (5.85, 3.7), (5.85, 2.4), color=DATA_EDGE, lw=1.8)
    step_marker(ax, 6.7, 3.05, 5)
    box(ax, 3.5, 1.3, 4.7, 1.1, "$p(y_t\\mid s_t,\\,y_{t-1},\\,c_t)$", kind="data",
        fontsize=FS_TEXT)
    arrow(ax, (8.2, 1.85), (9.7, 1.85), color=DATA_EDGE, lw=1.6)
    box(ax, 9.7, 1.3, 1.9, 1.1, "$y_t$", kind="new", fontsize=FS_TEXT)

    label(ax, 12.4, 1.85,
          "下一步重来一遍：分数换成 $e_{t+1,j}=a(s_t, h_j)$，\n"
          "$h_j$ 不变，$c_{t+1}$ 变",
          fontsize=FS_SMALL, color=MUTED, ha="left", linespacing=1.45)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")
    print("分数:", SCORES)
    print("指数:", [round(v, 3) for v in EXPS], "行和:", round(Z, 3))
    print("权重:", [round(a, 3) for a in ALPHA])


if __name__ == "__main__":
    main()
