"""生成 11.6 节的示意图：适配器权重在显存里的两种放法，以及与主机内存之间的换入换出。

正文位置：11_serving/11.6_multi_lora_serving.md
输出：11_serving/_images/ch11_6_adapter_memory.png

左：静态槽位，每个槽按最大秩预留，秩小的适配器只用到左边一小段，其余是填零的空占。
右：统一分页，KV 缓存与适配器权重从同一个页池里按需取页，互相穿插、不要求连续。
    S-LoRA 是一个秩一页；为便于画出，右图每格代表 8 页。
下：全部适配器常驻主机内存，只有当前批和下一批要用的才在显存里。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, MASK_EDGE, MASK_FACE,
                      DATA_EDGE, DATA_FACE, WEIGHT_EDGE, WEIGHT_FACE, arrow, box, finish, label)
from _style import image_path, use_cjk_font
from matplotlib.patches import Rectangle

OUTPUT = image_path("11_serving", "ch11_6_adapter_memory.png")

CELL = 0.8
SLOTS = (("a", 16), ("b", 8), ("c", 64), ("d", 8))  # (适配器, 秩)；最大秩 64，每格 8 个秩；a、b、c 与图 11.6-1 同秩
POOL = (
    "1 1 a 2 1 3 . 2 3 b 1 .",
    "2 1 3 1 . a 2 3 1 3 . 2",
    "c c 1 2 c c 3 . c c 1 c",
    ". 2 c 1 3 . 2 1 . 3 2 .",
)


def cell(ax, x, y, face, edge, text="", color=INK):
    ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=face, edgecolor=edge, linewidth=0.9,
                           zorder=2))
    if text:
        ax.text(x + CELL / 2, y + CELL / 2, text, ha="center", va="center",
                fontsize=FS_SMALL, color=color, zorder=3)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10, 6.4))

    # 左：静态槽位
    lx, top = 2.2, 12.6
    label(ax, 5.6, 14.3, "静态槽位：每槽按最大秩 64 预留", fontsize=FS_NAME, bold=True)
    label(ax, 5.6, 13.4, "每格 = 8 个秩的权重", fontsize=FS_SMALL, color=MUTED)
    for i, (name, rank) in enumerate(SLOTS):
        y = top - (i + 1) * CELL - i * 0.25
        label(ax, lx - 0.3, y + CELL / 2, f"槽 {i}：{name}", fontsize=FS_SMALL, ha="right")
        for j in range(8):
            used = j < rank // 8
            cell(ax, lx + j * CELL, y, WEIGHT_FACE if used else MASK_FACE,
                 WEIGHT_EDGE if used else MASK_EDGE)
        label(ax, lx + 8 * CELL + 0.25, y + CELL / 2, f"{rank}/64", fontsize=FS_SMALL,
              ha="left", color=ACCENT)
    label(ax, 5.6, 7.75, "4 个槽实际用到 96/256 = 37.5%，\n灰格是填零的空占，不能借给 KV 缓存",
          fontsize=FS_SMALL, color=MUTED)

    # 右：统一分页
    rx = 13.6
    label(ax, 18.9, 14.3, "统一分页：KV 与适配器共用一个页池", fontsize=FS_NAME, bold=True)
    label(ax, 18.9, 13.4, "每格代表 8 页；数字 = 请求号（KV），字母 = 适配器", fontsize=FS_SMALL,
          color=MUTED)
    for i, line in enumerate(POOL):
        y = top - (i + 1) * CELL
        for j, ch in enumerate(line.split()):
            x = rx + j * CELL
            if ch == ".":
                cell(ax, x, y, "white", MASK_EDGE)
            elif ch.isdigit():
                cell(ax, x, y, DATA_FACE, DATA_EDGE, ch)
            else:
                cell(ax, x, y, WEIGHT_FACE, WEIGHT_EDGE, ch)
    label(ax, 18.9, 8.55, "一个秩占一页：a 画 2 格、b 画 1 格、c 画 8 格，\n"
          "占多少取多少，页不必连续；换出后空页立刻可给 KV 用", fontsize=FS_SMALL, color=MUTED)

    # 图例
    ly = 5.9
    for x, face, edge, text in ((3.0, WEIGHT_FACE, WEIGHT_EDGE, "适配器权重"),
                                (8.4, DATA_FACE, DATA_EDGE, "KV 缓存"),
                                (12.8, MASK_FACE, MASK_EDGE, "预留但空占"),
                                (18.2, "white", MASK_EDGE, "空闲页")):
        cell(ax, x, ly, face, edge)
        label(ax, x + CELL + 0.25, ly + CELL / 2, text, fontsize=FS_SMALL, ha="left")

    # 下：主机内存
    ax.plot([0.6, 24.4], [5.1, 5.1], color=MASK_EDGE, lw=1.0, ls=(0, (4, 3)))
    label(ax, 0.6, 5.45, "显存", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, 0.6, 4.7, "主机内存", fontsize=FS_SMALL, color=MUTED, ha="left")
    box(ax, 4.2, 0.8, 16.6, 2.2, "", kind="neutral")
    label(ax, 12.5, 2.55, "全部已注册的适配器常驻主机内存", fontsize=FS_TEXT)
    for k, name in enumerate("abcdefghij"):
        box(ax, 4.9 + k * 1.55, 1.1, 1.2, 0.95, name, kind="weight", fontsize=FS_SMALL)
    arrow(ax, (8.6, 3.1), (8.6, 4.9), color=INK)
    label(ax, 8.35, 4.0, "换入：等待队列里的请求\n要用、显存里还没有的", fontsize=FS_SMALL,
          ha="right")
    arrow(ax, (16.4, 4.9), (16.4, 3.1), color=INK)
    label(ax, 16.65, 4.0, "淘汰：没有在跑的请求引用，\n按最近最少使用选出", fontsize=FS_SMALL,
          ha="left")

    finish(fig, ax, OUTPUT, xlim=(0, 25), ylim=(0.4, 15.0))


if __name__ == "__main__":
    main()
