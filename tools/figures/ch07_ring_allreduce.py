"""生成 7.1 节的图：4 张卡上的 Ring AllReduce，先 Reduce-Scatter 三步，再 All-Gather 三步。

正文位置：07_distributed_training/7.1_data_parallel.md
输出：07_distributed_training/_images/ch07_ring_allreduce.png

每个小方阵是一个时刻的全局状态：4 行是 4 张卡，4 列是梯度被切成的 4 块。
格子里的数字表示这一块已经累加了几张卡的贡献，写“和”的格子已经是 4 张卡的总和。
紫色粗框标出下一步要发给下家（卡 i 发给卡 i+1，卡 3 发给卡 0）的那一块，
青绿色标出上一步刚收到并更新的那一块。

Reduce-Scatter 第 s 步（s = 0, 1, 2）：卡 i 把第 (i - s) mod 4 块发给下家，下家把它加到自己的同一块上。
All-Gather 第 s 步：卡 i 把第 (i + 1 - s) mod 4 块发给下家，下家直接覆盖。
每一步每张卡只发 1/4 的数据，共 6 步，合计发送 2 x 3/4 = 1.5 份梯度。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, DATA_STRONG, FS_SMALL, FS_TEXT, FS_TITLE,
                      INK, MUTED, NEW_EDGE, NEW_FACE, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("07_distributed_training", "ch07_ring_allreduce.png")

K = 4
CELL = 0.95
PANEL_W = K * CELL
GAP = 2.35
X0 = 2.2


def simulate():
    """返回两个阶段的状态序列；每个状态是 (count, recv, send)。"""
    count = [[1] * K for _ in range(K)]
    states = []

    def snapshot(recv, send):
        states.append(([row[:] for row in count], set(recv), set(send)))

    # Reduce-Scatter
    snapshot([], [(i, i % K) for i in range(K)])
    for s in range(K - 1):
        moves = [(i, (i - s) % K) for i in range(K)]
        new = [row[:] for row in count]
        recv = []
        for i, j in moves:
            dst = (i + 1) % K
            new[dst][j] = count[dst][j] + count[i][j]
            recv.append((dst, j))
        count[:] = new
        nxt = [(i, (i - s - 1) % K) for i in range(K)] if s < K - 2 else \
              [(i, (i + 1) % K) for i in range(K)]
        snapshot(recv, nxt)
    rs = states[:]
    states.clear()
    # All-Gather
    for s in range(K - 1):
        moves = [(i, (i + 1 - s) % K) for i in range(K)]
        new = [row[:] for row in count]
        recv = []
        for i, j in moves:
            dst = (i + 1) % K
            new[dst][j] = count[i][j]
            recv.append((dst, j))
        count[:] = new
        nxt = [(i, (i - s) % K) for i in range(K)] if s < K - 2 else []
        snapshot(recv, nxt)
    return rs, states


def panel(ax, x, top, title, state, show_rank):
    count, recv, send = state
    label(ax, x + PANEL_W / 2, top + 0.95, title, fontsize=FS_TEXT, bold=True)
    for j in range(K):
        label(ax, x + (j + 0.5) * CELL, top + 0.33, str(j), fontsize=FS_SMALL, color=MUTED)
    if show_rank:
        label(ax, x - 0.2, top + 0.33, "块号", ha="right", fontsize=FS_SMALL, color=MUTED)
    for i in range(K):
        y = top - (i + 1) * CELL
        if show_rank:
            label(ax, x - 0.2, y + CELL / 2, f"卡 {i}", ha="right", fontsize=FS_TEXT)
        for j in range(K):
            c = count[i][j]
            if (i, j) in recv:
                face, edge = NEW_FACE, NEW_EDGE
            elif c == K:
                face, edge = DATA_STRONG, DATA_EDGE
            else:
                face, edge = DATA_FACE, DATA_EDGE
            ax.add_patch(Rectangle((x + j * CELL, y), CELL, CELL, facecolor=face,
                                   edgecolor=edge, linewidth=0.9, zorder=2))
            label(ax, x + (j + 0.5) * CELL, y + CELL / 2, "和" if c == K else str(c),
                  fontsize=FS_TEXT, bold=(c == K))
    for i, j in send:
        ax.add_patch(Rectangle((x + j * CELL + 0.06, top - (i + 1) * CELL + 0.06),
                               CELL - 0.12, CELL - 0.12, fill=False, edgecolor=ACCENT,
                               linewidth=2.6, zorder=5))


def row(ax, top, heading, titles, states):
    label(ax, 0.0, top + 2.0, heading, ha="left", fontsize=FS_TITLE, bold=True)
    for n, (title, state) in enumerate(zip(titles, states)):
        panel(ax, X0 + n * (PANEL_W + GAP), top, title, state, show_rank=(n == 0))
        if n:
            xa = X0 + n * (PANEL_W + GAP) - GAP + 0.35
            ax.annotate("", xy=(xa + GAP - 0.7, top - 2 * CELL), xytext=(xa, top - 2 * CELL),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))


def main() -> None:
    use_cjk_font()
    rs, ag = simulate()
    fig, ax = plt.subplots(figsize=(9.8, 7.4))
    row(ax, 12.0, "阶段一 Reduce-Scatter：每步把一块发给下家，下家累加",
        ["开始", "第 1 步后", "第 2 步后", "第 3 步后"], rs)
    row(ax, 4.6, "阶段二 All-Gather：每步把一块已求和的结果发给下家，下家覆盖",
        ["第 3 步后（同上）", "第 4 步后", "第 5 步后", "第 6 步后"], [rs[-1]] + ag)
    lx, ly = 0.0, -0.75
    ax.add_patch(Rectangle((lx, ly - 0.3), 0.6, 0.6, fill=False, edgecolor=ACCENT, linewidth=2.6))
    label(ax, lx + 0.8, ly, "下一步要发给下家的块", ha="left", fontsize=FS_SMALL)
    ax.add_patch(Rectangle((lx + 6.3, ly - 0.3), 0.6, 0.6, facecolor=NEW_FACE, edgecolor=NEW_EDGE))
    label(ax, lx + 7.1, ly, "上一步刚收到的块", ha="left", fontsize=FS_SMALL)
    ax.add_patch(Rectangle((lx + 11.8, ly - 0.3), 0.6, 0.6, facecolor=DATA_STRONG, edgecolor=DATA_EDGE))
    label(ax, lx + 12.6, ly, "数字 = 已累加几张卡；“和” = 4 张卡的总和", ha="left", fontsize=FS_SMALL)
    label(ax, 0.0, -1.85,
          "每步每张卡发送 1/4 份梯度，6 步合计 2 × 3/4 = 1.5 份；K 张卡时为 2(K − 1)/K 份。".replace("−", "-"),
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    finish(fig, ax, OUTPUT, xlim=(-0.2, X0 + 4 * PANEL_W + 3 * GAP + 0.3), ylim=(-2.4, 14.6))


if __name__ == "__main__":
    main()
