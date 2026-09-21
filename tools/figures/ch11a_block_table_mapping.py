"""生成 11.2 节“两个请求的块表指向同一个物理块池”示意图。

正文位置：11_serving/11.2_continuous_batching.md
输出：11_serving/_images/ch11a_block_table_mapping.png

设定：块大小 16。请求 A 已有 50 个词元，占 ceil(50 / 16) = 4 块，最后一块写了 50 - 48 = 2 格；
请求 B 已有 40 个词元，占 ceil(40 / 16) = 3 块，最后一块写了 40 - 32 = 8 格。两个请求的前 32 个
词元相同，前两个逻辑块指向同一组物理块（写满、只读、引用计数 2）。写时复制不在这张图里，
见 11.4 节的 ch11_4_block_table_cow.py。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_STRONG, FS_NAME, FS_SMALL, FS_TEXT, MUTED,
                      arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11a_block_table_mapping.png")

BLOCK = 16
N_POOL = 10
TOKENS = {"A": 50, "B": 40}
TABLES = {"A": [1, 4, 7, 2], "B": [1, 4, 8]}

CH = 1.7                 # 池中每个物理块的高度
POOL_X, POOL_W = 17.5, 12.5
TAB_W, NUM_W = 8.8, 2.4  # 块表里“逻辑块”一格与“物理块号”一格的宽度
RH = 1.7                 # 块表每行的高度


def pool_y(idx: int) -> float:
    """物理块 idx 的下边界；块 0 在最上面。"""
    return (N_POOL - 1 - idx) * CH


def filled(req: str, logical: int) -> int:
    return min(BLOCK, TOKENS[req] - logical * BLOCK)


def main() -> None:
    use_cjk_font()
    for req, table in TABLES.items():
        assert len(table) == math.ceil(TOKENS[req] / BLOCK)
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    top = N_POOL * CH

    # ---- 中：物理块池 ----
    label(ax, POOL_X + POOL_W / 2, top + 1.0, "物理块池（启动时切好，每块 16 个词元）",
          fontsize=FS_NAME, bold=True)
    owners: dict[int, list[tuple[str, int]]] = {}
    for req, table in TABLES.items():
        for logical, phys in enumerate(table):
            owners.setdefault(phys, []).append((req, logical))
    for idx in range(N_POOL):
        y = pool_y(idx)
        who = owners.get(idx)
        if not who:
            box(ax, POOL_X, y, POOL_W, CH, "", "plain", rounded=False, lw=1.0)
            label(ax, POOL_X + 0.5, y + CH / 2, f"块 {idx}", fontsize=FS_TEXT, color=MUTED,
                  ha="left")
            label(ax, POOL_X + POOL_W - 0.5, y + CH / 2, "空闲", fontsize=FS_SMALL,
                  color=MUTED, ha="right")
            continue
        shared = len(who) > 1
        box(ax, POOL_X, y, POOL_W, CH, "", "data", rounded=False, lw=1.0)
        if shared:
            ax.add_patch(Rectangle((POOL_X, y), POOL_W, CH, facecolor=DATA_STRONG,
                                   edgecolor=ACCENT, linewidth=2.2, zorder=3))
        n = filled(*who[0])
        label(ax, POOL_X + 0.5, y + CH / 2, f"块 {idx}", fontsize=FS_TEXT, ha="left", bold=True)
        state = f"写满 {n}/16" if n == BLOCK else f"已写 {n}/16"
        label(ax, POOL_X + 2.8, y + CH / 2, state, fontsize=FS_SMALL, ha="left")
        refs = f"引用 {len(who)}，只读" if shared else f"引用 {len(who)}"
        label(ax, POOL_X + POOL_W - 0.5, y + CH / 2, refs, fontsize=FS_SMALL, ha="right",
              color=ACCENT if shared else MUTED, bold=shared)

    # ---- 左右：两张块表 ----
    def table(req: str, x_logic: float, x_num: float, y_top: float, side: str) -> None:
        rows = TABLES[req]
        title_x = min(x_logic, x_num)
        label(ax, title_x, y_top + 1.55, f"请求 {req} 的块表", fontsize=FS_NAME, bold=True,
              ha="left")
        label(ax, title_x, y_top + 0.6, f"已有 {TOKENS[req]} 个词元，占 {len(rows)} 块",
              fontsize=FS_SMALL, color=MUTED, ha="left")
        for logical, phys in enumerate(rows):
            y = y_top - (logical + 1) * RH
            lo, hi = logical * BLOCK, logical * BLOCK + filled(req, logical) - 1
            box(ax, x_logic, y, TAB_W, RH, f"逻辑块 {logical}：位置 {lo}-{hi}", "data",
                rounded=False, lw=1.0, fontsize=FS_SMALL)
            box(ax, x_num, y, NUM_W, RH, str(phys), "data", rounded=False, lw=1.4,
                fontsize=FS_NAME, bold=True)
            y_mid, y_dst = y + RH / 2, pool_y(phys) + CH / 2
            if side == "left":
                arrow(ax, (x_num + NUM_W, y_mid), (POOL_X, y_dst), color=DATA_EDGE, lw=1.4)
            else:
                arrow(ax, (x_num, y_mid), (POOL_X + POOL_W, y_dst), color=DATA_EDGE, lw=1.4)

    ya_top = top - 1.2 * CH
    table("A", 0.0, TAB_W, ya_top, "left")
    xb_num = POOL_X + POOL_W + 6.2
    table("B", xb_num + NUM_W, xb_num, ya_top, "right")

    label(ax, 0.0, 1.9, "块表只存整数块号；\n物理块不必相邻，也不必按序；\n没写满的块（2、8）还会继续写", fontsize=FS_SMALL,
          color=MUTED, ha="left")
    label(ax, xb_num, 3.4, "A、B 的前 32 个词元相同：\n逻辑块 0、1 指向同一组物理块，\n这两块只读，不需要复制",
          fontsize=FS_SMALL, color=ACCENT, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.5, xb_num + NUM_W + TAB_W + 0.5), ylim=(-0.4, top + 1.9))


if __name__ == "__main__":
    main()
