"""生成图 11-2：同样 8 个请求，静态批处理与连续批处理各占用多少步。

正文位置：11_serving/11.2_continuous_batching.md
输出：11_serving/_images/ch11a_batching_gantt.png

横轴是迭代步，每一行是批里的一个位置。请求的第一格是 Prefill（青绿），其余是 Decode（蓝），
灰格是空转。各请求占用的步数：A2 B5 C3 D8 E4 F3 G5 H6，合计 36 格。
行只是为了画图方便：真实引擎限制的是每轮词元数和空闲 KV 块数，不是固定的行数。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, KINDS, MASK_EDGE, MASK_FACE, MUTED,
                      box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11a_batching_gantt.png")

REQUESTS = [("A", 2), ("B", 5), ("C", 3), ("D", 8), ("E", 4), ("F", 3), ("G", 5), ("H", 6)]
ROWS = 4
CW, CH = 2.3, 1.35
X0 = 4.6


def static_schedule():
    """整批进、整批出：一批的步数由最长的请求决定。"""
    grid, start = {}, 0
    for b in range(0, len(REQUESTS), ROWS):
        batch = REQUESTS[b:b + ROWS]
        span = max(n for _, n in batch)
        for r, (name, n) in enumerate(batch):
            for k in range(n):
                grid[(r, start + k)] = (name, k == 0)
        start += span
    return grid, start


def continuous_schedule():
    """每一步检查：有行空出来、队列里还有请求，就立刻补进去。"""
    queue = list(REQUESTS)
    rows = [None] * ROWS          # (name, 已走步数, 总步数)
    grid, t = {}, 0
    while queue or any(rows):
        for r in range(ROWS):
            if rows[r] is None and queue:
                name, n = queue.pop(0)
                rows[r] = (name, 0, n)
        for r in range(ROWS):
            if rows[r]:
                name, k, n = rows[r]
                grid[(r, t)] = (name, k == 0)
                rows[r] = (name, k + 1, n) if k + 1 < n else None
        t += 1
    return grid, t


def first_step(grid, name):
    return min(t for (r, t), (n, _) in grid.items() if n == name) + 1


def panel(ax, top, title, grid, steps, total_steps):
    label(ax, 0.0, top + 0.95, title, fontsize=FS_NAME, bold=True, ha="left")
    for r in range(ROWS):
        y = top - (r + 1) * CH
        label(ax, X0 - 0.5, y + CH / 2, f"位置 {r + 1}", fontsize=FS_SMALL, color=MUTED,
              ha="right")
        for t in range(total_steps):
            x = X0 + t * CW
            if (r, t) in grid:
                name, is_prefill = grid[(r, t)]
                face, edge = KINDS["new" if is_prefill else "data"]
                ax.add_patch(Rectangle((x, y), CW, CH, facecolor=face, edgecolor=edge, lw=1.0))
                label(ax, x + CW / 2, y + CH / 2, name, fontsize=FS_TEXT)
            elif t < steps:
                ax.add_patch(Rectangle((x, y), CW, CH, facecolor=MASK_FACE,
                                       edgecolor=MASK_EDGE, lw=0.8))
    used = len(grid)
    label(ax, X0 + total_steps * CW + 0.6, top - ROWS * CH / 2,
          f"共 {steps} 步\n{ROWS} × {steps} = {ROWS * steps} 格\n有用 {used} 格\n"
          f"占 {used / (ROWS * steps):.0%}",
          fontsize=FS_SMALL, ha="left")


def main() -> None:
    use_cjk_font()
    s_grid, s_steps = static_schedule()
    c_grid, c_steps = continuous_schedule()
    total = max(s_steps, c_steps)
    print(f"静态：{s_steps} 步，有用 {len(s_grid)}/{ROWS * s_steps} = "
          f"{len(s_grid) / (ROWS * s_steps):.3f}；E 在第 {first_step(s_grid, 'E')} 步开始")
    print(f"连续：{c_steps} 步，有用 {len(c_grid)}/{ROWS * c_steps} = "
          f"{len(c_grid) / (ROWS * c_steps):.3f}；E 在第 {first_step(c_grid, 'E')} 步开始")

    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    top1, top2 = 15.2, 6.6
    panel(ax, top1, "静态批处理：整批进、整批出，一批的步数由最长的请求决定", s_grid, s_steps,
          total)
    panel(ax, top2, "连续批处理：每一步都检查，有请求结束就让排队的请求补进来", c_grid, c_steps,
          total)

    # 第一批与第二批的分界
    xb = X0 + 8 * CW
    ax.plot([xb, xb], [top1 - ROWS * CH - 0.25, top1 + 0.25], color=ACCENT, lw=1.6, ls="--")
    label(ax, xb + 0.3, top1 - ROWS * CH - 0.75, "D 结束之后，第二批才能进", fontsize=FS_SMALL,
          color=ACCENT, ha="left")

    # 时间轴
    ybase = top2 - ROWS * CH
    for t in range(total):
        label(ax, X0 + t * CW + CW / 2, ybase - 0.55, str(t + 1), fontsize=FS_SMALL, color=MUTED)
    label(ax, X0 - 0.5, ybase - 0.55, "迭代步", fontsize=FS_SMALL, color=MUTED, ha="right")

    # 图例
    for i, (kind, text) in enumerate((("new", "请求的第一步：Prefill"),
                                      ("data", "其后每一步：Decode"),
                                      (None, "空转"))):
        lx = X0 + i * 11.0
        if kind:
            box(ax, lx, ybase - 2.35, 1.0, 0.75, "", kind, rounded=False, lw=1.2)
        else:
            ax.add_patch(Rectangle((lx, ybase - 2.35), 1.0, 0.75, facecolor=MASK_FACE,
                                   edgecolor=MASK_EDGE, lw=1.0))
        label(ax, lx + 1.35, ybase - 1.97, text, fontsize=FS_SMALL, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.4, X0 + total * CW + 7.2), ylim=(ybase - 2.9, top1 + 1.8))


if __name__ == "__main__":
    main()
