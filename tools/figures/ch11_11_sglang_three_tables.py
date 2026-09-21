"""生成 11.11 节插图：SGLang 里前缀树、req_to_token 与 KV 池三张表怎样接在一起。

正文位置：11_serving/11.11_sglang_internals.md
输出：11_serving/_images/ch11_11_sglang_three_tables.png

画的是正文算例中“请求 B 的 Prefill 结束”那一刻：A 已结束，它的 1,700 个槽位留在树里；
B 与 A 共享 1,200 个词元的系统提示，B 在 req_to_token 里的那一行前 1,200 格抄自树节点
的 value，后 180 格是本次新分配的槽位。共享靠的是两处出现同一批整数，KV 本身不复制。
数字由 scratchpad 里的 radix_walkthrough.py 重放得到。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, MASK_EDGE, MUTED, NEW_EDGE,
                      arrow, box, dots, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_11_sglang_three_tables.png")


def tree(ax):
    label(ax, 0.1, 11.3, "① 前缀树：键是词元序列，值是槽位下标", ha="left",
          fontsize=FS_TEXT, bold=True)
    box(ax, 3.95, 9.7, 1.4, 0.8, "根", kind="plain", fontsize=FS_SMALL)
    box(ax, 1.85, 7.15, 5.6, 1.8, "系统提示 1,200 词元\nvalue = 槽位 0..1199\nlock_ref = 1",
        kind="data", fontsize=FS_SMALL)
    box(ax, 0.1, 4.2, 4.4, 1.8, "A 的用户段 300\nvalue = 1200..1499\nlock_ref = 0",
        kind="neutral", fontsize=FS_SMALL)
    box(ax, 0.1, 1.3, 4.4, 1.8, "A 的输出 200\nvalue = 1500..1699\nlock_ref = 0",
        kind="neutral", fontsize=FS_SMALL)
    box(ax, 4.8, 4.2, 4.4, 1.8, "B 的用户段 180\nvalue = 1700..1879\nlock_ref = 1",
        kind="new", fontsize=FS_SMALL)
    arrow(ax, (4.65, 9.7), (4.65, 8.95))
    arrow(ax, (3.4, 7.15), (2.3, 6.0))
    arrow(ax, (5.9, 7.15), (7.0, 6.0))
    arrow(ax, (2.3, 4.2), (2.3, 3.1))
    label(ax, 4.9, 3.1, "灰：无人引用，可淘汰\n蓝：被 B 锁住的共享段\n青绿：B 本次新挂的叶子",
          ha="left", va="top", fontsize=FS_SMALL, color=MUTED)


def req_table(ax):
    label(ax, 10.2, 11.3, "② req_to_token[请求行, 位置]：int32 下标", ha="left",
          fontsize=FS_TEXT, bold=True)
    ax.add_patch(Rectangle((11.6, 9.6), 10.0, 0.85, facecolor="white", edgecolor=MASK_EDGE,
                           linewidth=1.2, linestyle="--", zorder=2))
    label(ax, 16.6, 10.02, "A 的行：请求结束后整行释放，下标已交给树", fontsize=FS_SMALL,
          color=MUTED)
    label(ax, 11.6, 8.85, "B 的行", ha="left", fontsize=FS_SMALL)
    box(ax, 11.6, 7.6, 5.0, 0.9, "0   1   2   …   1199", kind="data", fontsize=FS_SMALL,
        rounded=False)
    box(ax, 16.6, 7.6, 3.4, 0.9, "1700  …  1879", kind="new", fontsize=FS_SMALL,
        rounded=False)
    box(ax, 20.0, 7.6, 1.6, 0.9, "未用", kind="plain", fontsize=FS_SMALL, rounded=False,
        color=MUTED)
    label(ax, 13.9, 7.2, "位置 0..1199：抄自树", fontsize=FS_SMALL, color=DATA_EDGE)
    label(ax, 19.3, 7.2, "位置 1200..1379：新分配", fontsize=FS_SMALL, color=NEW_EDGE)
    arrow(ax, (7.45, 8.05), (11.6, 8.05), color=ACCENT, lw=1.8)
    label(ax, 9.5, 8.4, "match_prefix", fontsize=FS_SMALL, color=ACCENT)


def kv_pool(ax):
    y, h = 3.9, 1.0
    box(ax, 10.2, y, 3.4, h, "0 .. 1199", kind="data", fontsize=FS_SMALL, rounded=False)
    box(ax, 13.6, y, 2.6, h, "1200..1499", kind="neutral", fontsize=FS_SMALL, rounded=False)
    box(ax, 16.2, y, 2.6, h, "1500..1699", kind="neutral", fontsize=FS_SMALL, rounded=False)
    box(ax, 18.8, y, 2.6, h, "1700..1879", kind="new", fontsize=FS_SMALL, rounded=False)
    box(ax, 21.4, y, 0.9, h, "", kind="plain", fontsize=FS_SMALL, rounded=False)
    dots(ax, 21.85, y + h / 2, color=MUTED, spread=0.22, r=0.06)
    label(ax, 11.9, 3.45, "A 写入，B 直接读", fontsize=FS_SMALL, color=DATA_EDGE)
    label(ax, 16.2, 3.45, "A 留下，可淘汰", fontsize=FS_SMALL, color=MUTED)
    label(ax, 20.1, 3.45, "B 写入", fontsize=FS_SMALL, color=NEW_EDGE)
    arrow(ax, (13.4, 6.8), (11.9, 4.9), color=DATA_EDGE, lw=1.6)
    arrow(ax, (18.9, 6.8), (20.1, 4.9), color=NEW_EDGE, lw=1.6)
    label(ax, 16.2, 5.85, "注意力内核按这一行的\n下标到池里取 K、V", fontsize=FS_SMALL,
          color=MUTED)
    label(ax, 10.2, 2.6, "③ KV 池：每层 k_buffer[槽位, n_kv, d_h]，v_buffer 同形", ha="left",
          fontsize=FS_TEXT, bold=True)
    label(ax, 10.2, 2.05,
          "Llama 3 8B：一个槽位在每层的 K 是 [8, 128]，\n32 层 K 加 V 共 65,536 个数，每个 2 字节，合 128 KiB",
          ha="left", va="top", fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 6.3))
    tree(ax)
    req_table(ax)
    kv_pool(ax)
    finish(fig, ax, OUTPUT, xlim=(0, 22.6), ylim=(0.6, 11.8))
    plt.close(fig)


if __name__ == "__main__":
    main()
