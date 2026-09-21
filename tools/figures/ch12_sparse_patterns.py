"""生成 12.3 节插图：四种注意力模式在同一张 [n, n] 矩阵上的形状。

正文位置：12_encoder_models/12.3_longformer_bigbird.md
输出：12_encoder_models/_images/ch12_sparse_patterns.png

取 n = 36（真实配置里 n 是 4,096，格子画不下），行是 Query 位置、列是 Key 位置，
深蓝格表示这一对要算分数。窗口取 w = 7（每侧 3），全局词元取位置 1 和 2，
BigBird 取块大小 3：每个查询块看左中右 3 个窗口块、1 个全局块、1 个随机块。
每幅下方给出被算到的格子数与占满阵的比例，由脚本自己数出。
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_STRONG, FS_SMALL, FS_TEXT, INK, MASK_EDGE,
                      MASK_FACE, MUTED, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("12_encoder_models", "ch12_sparse_patterns.png")

N = 36
CELL = 0.215
HALF = 3          # 窗口每侧 3 个位置，总宽 w = 7
GLOBAL = (1, 2)   # 全局词元所在的位置
BLOCK = 3         # BigBird 的块大小
RANDOM_BLOCKS = 1
SEED = 12


def full(i, j):
    return True


def window(i, j):
    return abs(i - j) <= HALF


def window_global(i, j):
    return window(i, j) or i in GLOBAL or j in GLOBAL


def build_bigbird():
    """按块选中：3 个窗口块 + 第 0 个全局块 + 若干随机块。"""
    rng = random.Random(SEED)
    n_blocks = N // BLOCK
    chosen = {}
    for bi in range(n_blocks):
        picked = {0}
        picked |= {b for b in (bi - 1, bi, bi + 1) if 0 <= b < n_blocks}
        pool = [b for b in range(n_blocks) if b not in picked]
        picked |= set(rng.sample(pool, min(RANDOM_BLOCKS, len(pool))))
        chosen[bi] = picked
    return lambda i, j: (j // BLOCK) in chosen[i // BLOCK]


PANELS = [
    ("（a）全注意力", full, "BERT / RoBERTa"),
    ("（b）滑动窗口", window, "每侧 3，总宽 w = 7"),
    ("（c）窗口 + 全局词元", window_global, "位置 1、2 走十字"),
    ("（d）BigBird 块稀疏", build_bigbird(), "块 3：窗口 3 块 + 全局 1 块 + 随机 1 块"),
]


def panel(ax, x0, y0, title, visible, note):
    hits = 0
    for i in range(N):
        for j in range(N):
            ok = visible(i, j)
            hits += ok
            face, edge = (DATA_STRONG, DATA_EDGE) if ok else (MASK_FACE, MASK_EDGE)
            ax.add_patch(Rectangle((x0 + j * CELL, y0 - (i + 1) * CELL), CELL, CELL,
                                   facecolor=face, edgecolor=edge, linewidth=0.35,
                                   zorder=2))
    side = N * CELL
    ax.add_patch(Rectangle((x0, y0 - side), side, side, fill=False, edgecolor=INK,
                           linewidth=1.2, zorder=3))
    label(ax, x0 + side / 2, y0 + 0.52, title, fontsize=FS_TEXT, bold=True)
    label(ax, x0 + side / 2, y0 - side - 0.78, note, fontsize=FS_SMALL, color=MUTED)
    label(ax, x0 + side / 2, y0 - side - 1.28,
          f"要算 {hits} 格，占满阵 {hits / N / N:.0%}", fontsize=FS_SMALL, color=ACCENT)
    return hits


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    side = N * CELL
    gx, gy = side + 1.6, side + 3.1
    counts = []
    for k, (title, fn, note) in enumerate(PANELS):
        x0 = (k % 2) * gx
        y0 = -(k // 2) * gy
        counts.append(panel(ax, x0, y0, title, fn, note))
    label(ax, gx / 2 + side / 2, 1.30, "每幅都是一张 [n, n] 矩阵：行是 Query 位置，列是 Key 位置",
          fontsize=FS_SMALL, color=MUTED)
    finish(fig, ax, OUTPUT,
           xlim=(-0.5, gx + side + 0.5), ylim=(-gy - side - 1.70, 1.70))
    print("各幅的格子数：", counts, " 满阵 =", N * N)


if __name__ == "__main__":
    main()
