"""生成 11.4 节的块表与写时复制示意图。

正文位置：11_serving/11.4_kv_memory_management.md
输出：11_serving/_images/ch11_4_block_table_cow.png

上半幅：一个 50 词元的请求（块大小 16）怎样经块表 [7, 2, 11, 5] 落到物理块池里，
以及位置 37、位置 50 的槽位怎样算出。下半幅：两路并行采样共用同一组物理块，
样本 A 写位置 50 时发现物理块 5 的引用计数为 2，于是取空闲块 9、复制、改自己的块表；
样本 B 随后就地写入块 5。数值与正文的算例一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, MUTED, NEW_EDGE,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_4_block_table_cow.png")

BLOCK = 16
TABLE = [7, 2, 11, 5]


def slot(pos: int, table: list[int]) -> tuple[int, int, int, int]:
    logical, offset = divmod(pos, BLOCK)
    return logical, offset, table[logical], table[logical] * BLOCK + offset


def panel_lookup(ax, top: float) -> None:
    label(ax, 0, top, "（一）块表：逻辑块号 → 物理块号", fontsize=FS_TITLE, bold=True, ha="left")

    # 逻辑视图与块表逐行对齐
    label(ax, 5.5, top - 1.9, "请求看到的连续位置", fontsize=FS_SMALL, color=MUTED)
    label(ax, 14.5, top - 1.9, "块表", fontsize=FS_SMALL, color=MUTED)
    texts = ["逻辑块 0：位置 0–15", "逻辑块 1：位置 16–31",
             "逻辑块 2：位置 32–47", "逻辑块 3：位置 48–49，空 14 格"]
    for k, text in enumerate(texts):
        y = top - 4.4 - k * 2.0
        box(ax, 0, y, 11.0, 1.5, text, "data", fontsize=FS_SMALL)
        arrow(ax, (11.0, y + 0.75), (13.2, y + 0.75))
        box(ax, 13.2, y, 2.6, 1.5, str(TABLE[k]), "data", fontsize=FS_NAME, bold=True,
            rounded=False)
    arrow(ax, (16.1, top - 6.65), (18.6, top - 6.65), color=ACCENT, lw=2)
    label(ax, 17.35, top - 5.3, "按块号\n索引", fontsize=FS_SMALL, color=ACCENT)

    # 物理块池：两行六列
    label(ax, 29.4, top - 1.9, "物理块池：每块 16 个词元 × 32 层，Llama 3 8B 下合 2 MiB",
          fontsize=FS_SMALL, color=MUTED)
    owner = {p: k for k, p in enumerate(TABLE)}
    free = {9, 10}
    for p in range(12):
        row, col = divmod(p, 6)
        x, y = 19.0 + col * 3.5, top - 5.6 - row * 3.4
        if p in owner:
            box(ax, x, y, 3.1, 2.8, f"块 {p}\n逻辑块 {owner[p]}", "data", fontsize=FS_SMALL,
                bold=True)
        elif p in free:
            box(ax, x, y, 3.1, 2.8, f"块 {p}\n空闲", "plain", fontsize=FS_SMALL, color=MUTED)
        else:
            box(ax, x, y, 3.1, 2.8, f"块 {p}\n别的请求", "neutral", fontsize=FS_SMALL,
                color=MUTED)

    lines = []
    for verb, pos in (("写位置", 37), ("写位置", 50)):
        logical, offset, phys, s = slot(pos, TABLE)
        lines.append(f"{verb} {pos}：{pos} ÷ 16 = {logical} 余 {offset}  →  块表[{logical}] = {phys}"
                     f"  →  槽位 = {phys} × 16 + {offset} = {s}")
    label(ax, 0, top - 13.0, lines[0], fontsize=FS_TEXT, color=ACCENT, ha="left")
    label(ax, 0, top - 14.3, lines[1], fontsize=FS_TEXT, color=ACCENT, ha="left")


def table_row(ax, x, y, name, values, new_cols=()):
    label(ax, x, y - 0.6, name, fontsize=FS_SMALL, ha="left")
    for j, v in enumerate(values):
        kind = "new" if j in new_cols else "data"
        box(ax, x + 3.6 + j * 1.9, y - 1.2, 1.9, 1.2, str(v), kind, fontsize=FS_TEXT,
            rounded=False)


def phys_row(ax, x, y, items):
    for j, (pid, note, kind) in enumerate(items):
        box(ax, x + j * 3.9, y, 3.65, 3.1, f"块 {pid}\n{note}", kind, fontsize=FS_SMALL)


def panel_cow(ax, top: float) -> None:
    label(ax, 0, top, "（二）写时复制：同一个 Prompt 采样两路，样本 A 先写位置 50",
          fontsize=FS_TITLE, bold=True, ha="left")

    # 左：分叉之后、写入之前
    lx = 0
    label(ax, lx, top - 1.8, "分叉之后、写入之前：两张块表相同", fontsize=FS_SMALL,
          color=MUTED, ha="left")
    table_row(ax, lx, top - 2.6, "样本 A", TABLE)
    table_row(ax, lx, top - 4.2, "样本 B", TABLE)
    phys_row(ax, lx, top - 9.6, [(7, "引用 2", "data"), (2, "引用 2", "data"),
                                 (11, "引用 2", "data"), (5, "引用 2\n空 14 格", "data")])

    arrow(ax, (16.2, top - 5.4), (19.6, top - 5.4), color=ACCENT, lw=2)
    label(ax, 17.9, top - 4.3, "A 要写块 5\n引用 > 1", fontsize=FS_SMALL, color=ACCENT)

    # 右：A 写入之后
    rx = 20.4
    label(ax, rx, top - 1.8, "A 写入之后：只有 A 的最后一项改了", fontsize=FS_SMALL,
          color=MUTED, ha="left")
    table_row(ax, rx, top - 2.6, "样本 A", [7, 2, 11, 9], new_cols=(3,))
    table_row(ax, rx, top - 4.2, "样本 B", TABLE)
    phys_row(ax, rx, top - 9.6, [(7, "引用 2", "data"), (2, "引用 2", "data"),
                                 (11, "引用 2", "data"), (5, "引用 1\nB 就地写", "data"),
                                 (9, "引用 1\n复制自块 5", "new")])
    label(ax, rx + 16.6, top - 10.4, "复制 1 块 = 2 MiB", fontsize=FS_SMALL, color=NEW_EDGE)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 9.3))
    panel_lookup(ax, 27.0)
    ax.plot([0, 40.0], [11.5, 11.5], color=MUTED, lw=0.8, zorder=0)
    panel_cow(ax, 10.2)
    finish(fig, ax, OUTPUT, xlim=(-0.5, 40.6), ylim=(-0.9, 28.0))


if __name__ == "__main__":
    main()
