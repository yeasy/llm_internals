"""生成 11.5 节插图：前缀树的匹配、拆分节点与从叶子淘汰。

正文位置：11_serving/11.5_prefix_reuse.md
输出：11_serving/_images/ch11_5_radix_tree_ops.png

三个时刻：① 请求 A（14 个词元）结束，整条序列是一条边；② 请求 B 与 A 共享前 10 个
词元，匹配停在边的中间，节点拆成两段，B 的后缀挂成新叶子，B 经过的路径引用计数加 1；
③ 又来过一个不相干的请求 C 之后显存不足，只在引用计数为 0 的叶子里挑最久未用的淘汰。
词元与 ch11_5_block_hash_chain.py 用同一组，便于对照两种做法的命中长度（8 与 10）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, FS_SMALL, FS_TITLE, MASK_EDGE, MASK_FACE, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_5_radix_tree_ops.png")

ROOT_Y, L1_Y, L2_Y = 10.2, 6.9, 3.3
NODE_H = 1.5


def root(ax, cx):
    box(ax, cx - 0.7, ROOT_Y, 1.4, 0.9, "根", kind="plain", fontsize=FS_SMALL)
    return cx, ROOT_Y


def node(ax, cx, y, w, tokens, meta, kind="data", evicted=False):
    """画一个节点：上一行是这条边上的词元，下一行是引用计数与最近访问时刻。"""
    if evicted:
        ax.add_patch(Rectangle((cx - w / 2, y), w, NODE_H, facecolor=MASK_FACE,
                               edgecolor=MASK_EDGE, linewidth=1.5, linestyle="--", zorder=2))
        label(ax, cx, y + NODE_H / 2, f"{tokens}\n{meta}", fontsize=FS_SMALL, color=MUTED)
    else:
        box(ax, cx - w / 2, y, w, NODE_H, f"{tokens}\n{meta}", kind=kind, fontsize=FS_SMALL)
    return cx, y


def edge(ax, parent, child, *, color=MUTED):
    arrow(ax, (parent[0], parent[1]), (child[0], child[1] + NODE_H), color=color)


def panel_title(ax, x, text):
    label(ax, x, 13.2, text, ha="left", va="top", fontsize=FS_TITLE, bold=True)


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.6, 7.4))

    # ---------- ① ----------
    x1 = 3.2
    panel_title(ax, 0.0, "① 请求 A 结束")
    r = root(ax, x1)
    a = node(ax, x1, L1_Y, 6.4, "ABCDEFGHIJKLMN（14 个）", "引用 0，t=1")
    edge(ax, r, a)
    label(ax, x1, L1_Y - 0.9, "整条序列是一条边，节点里\n存 14 个 KV 位置的下标",
          fontsize=FS_SMALL, color=MUTED)

    # ---------- ② ----------
    x2 = 11.3
    panel_title(ax, 7.0, "② 请求 B 到达：命中 10 个\n    词元，节点一分为二")
    r = root(ax, x2)
    p = node(ax, x2, L1_Y, 5.4, "ABCDEFGHIJ（10 个）", "引用 1，t=2")
    c1 = node(ax, x2 - 2.0, L2_Y, 3.6, "KLMN（4 个）", "引用 0，t=2")
    c2 = node(ax, x2 + 2.0, L2_Y, 3.6, "wxyz（4 个）", "引用 1，t=2", kind="new")
    edge(ax, r, p)
    edge(ax, p, c1)
    edge(ax, p, c2)
    label(ax, x2 - 2.0, L2_Y - 0.65, "A 的旧后缀", fontsize=FS_SMALL, color=MUTED)
    label(ax, x2 + 2.0, L2_Y - 0.9, "B 的后缀：只 Prefill\n这 4 个词元", fontsize=FS_SMALL,
          color=ACCENT)

    # ---------- ③ ----------
    x3 = 20.2
    panel_title(ax, 16.2, "③ 请求 C 来过之后显存不足：\n    要腾出 4 个词元")
    r = root(ax, x3 + 2.4)
    p = node(ax, x3, L1_Y, 5.4, "ABCDEFGHIJ（10 个）", "引用 1，t=2")
    q = node(ax, x3 + 5.2, L1_Y, 4.2, "PQRSTU（6 个）", "引用 0，t=3")
    c1 = node(ax, x3 - 2.0, L2_Y, 3.6, "KLMN（4 个）", "引用 0，t=2", evicted=True)
    c2 = node(ax, x3 + 2.0, L2_Y, 3.6, "wxyz（4 个）", "引用 1，t=2", kind="new")
    edge(ax, r, p)
    edge(ax, r, q)
    edge(ax, p, c1)
    edge(ax, p, c2)
    label(ax, x3 - 2.0, L2_Y - 0.9, "叶子、无人引用、\n最久未用：淘汰", fontsize=FS_SMALL,
          color=ACCENT)
    label(ax, x3 + 2.0, L2_Y - 0.9, "B 还在运行，\n不可淘汰", fontsize=FS_SMALL, color=MUTED)
    label(ax, x3 + 5.2, L1_Y - 0.9, "可淘汰，但比\nKLMN 新，保留", fontsize=FS_SMALL, color=MUTED)

    for x in (6.7, 15.7):
        ax.plot([x, x], [1.6, 13.2], color=MASK_EDGE, lw=1.0, linestyle=":")

    finish(fig, ax, OUTPUT, xlim=(-0.3, 27.8), ylim=(1.4, 13.6))


if __name__ == "__main__":
    main()
