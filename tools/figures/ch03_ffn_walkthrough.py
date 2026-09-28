"""生成图 3-4：两矩阵 MLP 与 SwiGLU 对同一个 x 各算了什么。

正文位置：03_components/3.4_feedforward.md
输出：03_components/_images/ch03_ffn_walkthrough.png

上半幅是两矩阵 MLP 的三步：x 与 W_1 的每一列做点积（比对），ReLU 把负分置 0（筛选），
再用剩下的分数给 W_2 的各行加权、逐列相加（写回）。下半幅是 SwiGLU：门控一路与
另一路逐元素相乘，得到新的中间向量，写回一步与上半幅相同。数值与正文表 3-6 一致。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, MUTED, arrow, finish,
                      grid, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_ffn_walkthrough.png")

X = [1, 2]
W1 = [[1, 0, 1, -1], [0, 1, -1, 1]]
W2 = [[1, 0], [0, 1], [2, 2], [-1, 1]]
W_UP = [[1, 0, 1, -1], [0, 1, 1, 0]]


def matvec(x, w):
    return [sum(x[i] * w[i][j] for i in range(len(x))) for j in range(len(w[0]))]


def silu(z):
    return z / (1 + math.exp(-z))


H = matvec(X, W1)                          # [1, 2, -1, 1]
A = [max(0, v) for v in H]                 # [1, 2, 0, 1]
PARTS = [[a * v for v in row] for a, row in zip(A, W2)]
Y = [sum(p[c] for p in PARTS) for c in (0, 1)]

G = [silu(v) for v in H]                   # W_gate = W_1
U = matvec(X, W_UP)
HS = [g * u for g, u in zip(G, U)]
YS = [sum(h * row[c] for h, row in zip(HS, W2)) for c in (0, 1)]

C = 1.0  # 单元格边长


def num(v):
    """整数写成整数；负号用 ASCII 减号，避开中文字体缺字。"""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3f}"


def op(ax, x, y, text):
    label(ax, x, y, text, fontsize=FS_TITLE)


def step(ax, x, y, text):
    label(ax, x, y, text, fontsize=FS_TEXT, color=ACCENT, ha="left", bold=True)


def top_panel(ax):
    label(ax, 11.6, 25.4, "两矩阵 MLP：d = 2，d_ff = 4，激活用 ReLU", fontsize=FS_TITLE, bold=True)

    # ① 比对：x × W1 = h
    top = 22.6
    step(ax, 0.0, top + 1.55, "① 比对：x 与 $W_1$ 的每一列做点积")
    label(ax, 1.0, top + 0.55, "x  [1, 2]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 0.0, top - 0.5, [X], cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num)
    op(ax, 2.6, top - 1.0, "×")
    label(ax, 5.2, top + 0.55, "$W_1$  [2, 4]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 3.2, top, W1, kind="weight", cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num)
    for j in range(4):
        label(ax, 3.7 + j, top - 2.45, f"列{j + 1}", fontsize=FS_SMALL - 1, color=MUTED)
    op(ax, 7.8, top - 1.0, "=")
    label(ax, 10.4, top + 0.55, "分数 h  [1, 4]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 8.4, top - 0.5, [H], cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num)
    label(ax, 10.4, top - 2.45, "每格 = x 与对应列的点积", fontsize=FS_SMALL - 1, color=MUTED)

    # ② 筛选：ReLU
    step(ax, 13.4, top + 1.55, "② 筛选：ReLU 把负分置 0")
    arrow(ax, (12.6, top - 1.0), (15.2, top - 1.0))
    label(ax, 13.9, top - 0.55, "ReLU", fontsize=FS_SMALL, color=MUTED)
    label(ax, 17.4, top + 0.55, "ReLU 后  [1, 4]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 15.4, top - 0.5, [A], kind="new", cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num,
         row_kinds=None)
    grid(ax, 17.4, top - 0.5, [[A[2]]], kind="neutral", cell_w=C, cell_h=C, fontsize=FS_TEXT,
         fmt=num)
    label(ax, 17.9, top - 2.45, "单元 3 被关掉", fontsize=FS_SMALL - 1, color=MUTED)

    # ③ 写回：a_j × W2 第 j 行，再逐列相加
    top = 17.0
    step(ax, 0.0, top + 1.9, "③ 写回：用这 4 个数给 $W_2$ 的各行加权，再逐列相加")
    label(ax, 3.5, top + 0.75, "② 的结果\n竖着排", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 3.0, top, [[a] for a in A], kind="new", cell_w=C, cell_h=C, fontsize=FS_TEXT,
         fmt=num, row_kinds={2: "neutral"})
    op(ax, 4.9, top - 2.0, "×")
    label(ax, 6.8, top + 0.75, "$W_2$  [4, 2]\n每行一个写回向量", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 5.8, top, W2, kind="weight", cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num)
    for j in range(4):
        label(ax, 8.1, top - 0.5 - j, f"行{j + 1}", fontsize=FS_SMALL - 1, color=MUTED, ha="left")
    op(ax, 10.1, top - 2.0, "=")
    label(ax, 12.0, top + 0.75, "逐行相乘", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 11.0, top, PARTS, cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num,
         row_kinds={2: "neutral"})
    arrow(ax, (13.4, top - 2.0), (16.0, top - 2.0))
    label(ax, 14.7, top - 1.55, "逐列相加", fontsize=FS_SMALL, color=MUTED)
    label(ax, 17.2, top - 0.9, "输出 y  [1, 2]", fontsize=FS_SMALL, color=MUTED)
    grid(ax, 16.2, top - 1.5, [Y], kind="new", cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num)
    label(ax, 17.2, top - 3.2, "形状回到 x 的 [1, 2]", fontsize=FS_SMALL - 1, color=MUTED)


def bottom_panel(ax):
    label(ax, 11.6, 10.4, "SwiGLU：同一个 x，改的是第 ② 步", fontsize=FS_TITLE, bold=True)
    cw = 1.9   # 小数要宽一点的格子
    gx = 10.2  # 右侧向量的左边界

    rows = [
        (8.2, "门控一路", "$x W_{gate}$，$W_{gate} = W_1$", H, "SiLU", G, "决定开多少"),
        (5.4, "另一路", "$x W_{up}$", U, None, U, "决定写多少、往哪写"),
    ]
    for y, name, how, pre, act, post, note in rows:
        label(ax, 0.0, y - 0.5, name, fontsize=FS_TEXT, bold=True, ha="left")
        label(ax, 0.0, y - 1.25, how, fontsize=FS_SMALL - 1, color=MUTED, ha="left")
        if act:
            grid(ax, 4.8, y, [pre], cell_w=C, cell_h=C, fontsize=FS_TEXT, fmt=num)
            arrow(ax, (8.9, y - 0.5), (10.0, y - 0.5))
            label(ax, 9.45, y + 0.05, act, fontsize=FS_SMALL, color=MUTED)
        grid(ax, gx, y, [post], kind="new" if act else "data", cell_w=cw, cell_h=C,
             fontsize=FS_SMALL, fmt=num)
        label(ax, gx + 4 * cw + 0.3, y - 0.5, note, fontsize=FS_SMALL, color=MUTED, ha="left")

    op(ax, gx + 2 * cw, 6.85, "⊙")
    label(ax, gx + 2 * cw + 0.6, 6.85, "逐元素相乘", fontsize=FS_SMALL - 1, color=MUTED, ha="left")
    y = 3.4
    arrow(ax, (gx + 2 * cw, 4.3), (gx + 2 * cw, y + 0.1))
    label(ax, 0.0, y - 0.5, "中间向量", fontsize=FS_TEXT, bold=True, ha="left")
    label(ax, 0.0, y - 1.25, "代替 ReLU 后的 [1, 2, 0, 1]", fontsize=FS_SMALL - 1, color=MUTED,
          ha="left")
    grid(ax, gx, y, [HS], kind="new", cell_w=cw, cell_h=C, fontsize=FS_SMALL, fmt=num)
    # 与两矩阵版本的两处不同
    for j, text in ((2, "单元 3：门没关死，\n漏过一个负值"), (3, "单元 4：门开着，\n另一路为负，反向写回")):
        cx = gx + (j + 0.5) * cw
        ax.add_patch(plt.Rectangle((gx + j * cw, y - C), cw, C, fill=False, edgecolor=ACCENT,
                                   linewidth=2.4, zorder=6))
        label(ax, cx + (-0.6 if j == 2 else 2.6), y - 2.1, text, fontsize=FS_SMALL - 1, color=ACCENT)
    arrow(ax, (gx + 4 * cw + 0.1, y - 0.5), (gx + 4 * cw + 1.1, y - 0.5))
    label(ax, gx + 4 * cw + 1.3, y - 0.2, "按第 ③ 步写回", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, gx + 4 * cw + 1.3, y - 0.95, "（$W_{down} = W_2$）", fontsize=FS_SMALL - 1, color=MUTED, ha="left")
    label(ax, 11.6, -0.4, f"y = [{num(YS[0])}, {num(YS[1])}]", fontsize=FS_NAME, bold=True)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 10.8))
    top_panel(ax)
    ax.plot([0.0, 23.4], [11.6, 11.6], color="#c9c8c2", lw=1.0)
    bottom_panel(ax)
    finish(fig, ax, OUTPUT, xlim=(-0.3, 24.6), ylim=(-1.2, 26.2))


if __name__ == "__main__":
    main()
