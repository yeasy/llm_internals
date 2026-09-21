"""生成 1.4 节插图：同一段长度为 n 的序列，三种层把首尾连起来各要走几跳。

正文位置：01_introduction/1.4_transformer_idea.md
输出：01_introduction/_images/ch01_path_and_steps.png

三幅并排，序列长度固定为 8，卷积核宽 k = 3，青绿标出的是信息实际走过的通路：
- 循环层：位置 1 的信息要经过中间每一个位置才到位置 8，路径长 n - 1，且这 n 步
  必须依次执行。
- 卷积层：每层把感受野扩大 k - 1，堆到覆盖全序列要 ceil((n-1)/(k-1)) 层；同层
  各位置可并行。
- 自注意力：位置 8 一次读到全部位置，路径长 1，顺序步数 1，代价是 n x n 张分数表。
图中的层数与跳数由脚本按 n、k 算出，运行时打印。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _diagram import FS_NAME, FS_SMALL, INK, MUTED, NEW_EDGE, arrow, box, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("01_introduction", "ch01_path_and_steps.png")

N = 8
K = 3
CONV_LAYERS = math.ceil((N - 1) / (K - 1))

CELL_W, CELL_H, GAP = 0.92, 0.62, 0.22
X0 = 0.45
ROW_STEP = 1.25

TITLE_Y, SUB_Y, NOTE_Y = 8.1, 7.3, -0.95


def cell_x(i: int) -> float:
    return X0 + i * (CELL_W + GAP)


def row(ax, y, highlight=()):
    for i in range(N):
        box(ax, cell_x(i), y, CELL_W, CELL_H, f"{i+1}",
            kind="new" if i in highlight else "data", fontsize=FS_SMALL)


def frame(ax):
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-1.6, 8.6)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_rnn(ax):
    frame(ax)
    y = 3.4
    row(ax, y, highlight=range(N))
    for i in range(N - 1):
        arrow(ax, (cell_x(i) + CELL_W, y + CELL_H / 2),
              (cell_x(i + 1), y + CELL_H / 2), color=NEW_EDGE, lw=1.7)
    label(ax, 4.8, TITLE_Y, "循环层", fontsize=FS_NAME, color=INK, bold=True)
    label(ax, 4.8, SUB_Y, f"位置 1 到位置 {N}：{N - 1} 跳", fontsize=FS_SMALL,
          color=NEW_EDGE)
    label(ax, 4.8, NOTE_Y, f"顺序步数 {N} 步，不可并行\n路径长 O(n)",
          fontsize=FS_SMALL, color=MUTED, linespacing=1.5)


def panel_cnn(ax):
    frame(ax)
    ys = [0.15 + lvl * ROW_STEP for lvl in range(CONV_LAYERS + 1)]
    # 自顶向下算感受野：顶层只看位置 8，每往下一层向左扩 k - 1 个位置。
    cones = []
    for lvl in range(CONV_LAYERS, -1, -1):
        spread = (CONV_LAYERS - lvl) * (K - 1)
        cones.insert(0, set(range(max(0, N - 1 - spread), N)))
    for lvl, y in enumerate(ys):
        row(ax, y, highlight=cones[lvl])
    for lvl in range(CONV_LAYERS):
        for dst in cones[lvl + 1]:
            for src in range(dst - (K - 1), dst + 1):
                if src in cones[lvl]:
                    arrow(ax, (cell_x(src) + CELL_W / 2, ys[lvl] + CELL_H),
                          (cell_x(dst) + CELL_W / 2, ys[lvl + 1]),
                          color=NEW_EDGE, lw=0.8)
    label(ax, 4.8, TITLE_Y, f"卷积层（k = {K}）", fontsize=FS_NAME, color=INK,
          bold=True)
    label(ax, 4.8, SUB_Y, f"感受野每层扩 {K - 1} 个位置，覆盖全序列要 {CONV_LAYERS} 层",
          fontsize=FS_SMALL, color=NEW_EDGE)
    label(ax, 4.8, NOTE_Y, "同层各位置并行，顺序步数 O(1)\n路径长 O(n/k)",
          fontsize=FS_SMALL, color=MUTED, linespacing=1.5)


def panel_attn(ax):
    frame(ax)
    y_in, y_out = 1.8, 4.6
    row(ax, y_in, highlight=range(N))
    box(ax, cell_x(N - 1), y_out, CELL_W, CELL_H, f"{N}", kind="new",
        fontsize=FS_SMALL)
    for i in range(N):
        arrow(ax, (cell_x(i) + CELL_W / 2, y_in + CELL_H),
              (cell_x(N - 1) + CELL_W / 2, y_out), color=NEW_EDGE, lw=0.9,
              rad=-0.16 if i < N - 1 else 0.0)
    label(ax, 4.8, TITLE_Y, "自注意力层", fontsize=FS_NAME, color=INK, bold=True)
    label(ax, 4.8, SUB_Y, f"位置 {N} 一次读到全部位置：1 跳", fontsize=FS_SMALL,
          color=NEW_EDGE)
    label(ax, 4.8, NOTE_Y,
          f"全部位置并行，顺序步数 O(1)\n路径长 O(1)，代价是 {N}x{N} 张分数表",
          fontsize=FS_SMALL, color=MUTED, linespacing=1.5)


def main() -> None:
    use_cjk_font()
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 4.2))
    panel_rnn(axes[0])
    panel_cnn(axes[1])
    panel_attn(axes[2])
    fig.tight_layout(w_pad=0.4)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已写入 {OUTPUT}")
    print(f"n = {N}, k = {K}")
    print(f"循环层路径长 = n - 1 = {N - 1}，顺序步数 = {N}")
    print(f"卷积层层数 = ceil((n-1)/(k-1)) = {CONV_LAYERS}，感受野 = 1+L(k-1) = "
          f"{1 + CONV_LAYERS * (K - 1)}，顺序步数 = 1")
    print(f"自注意力路径长 = 1，顺序步数 = 1，分数表 {N}x{N} = {N * N} 格")


if __name__ == "__main__":
    main()
