"""生成 11.8 节的图：流水线并行做 Decode 时，各级在每个时间片里算的是谁。

正文位置：11_serving/11.8_multi_gpu_inference.md
输出：11_serving/_images/ch11_8_pp_timeline.png

4 级流水线（p = 4），横轴是时间片，一格是“某一级为某个微批算一次本级的层”。
上半幅：流水线里只有一个批 A。A 的第 k+1 步要等第 k 步走完第 4 级并采样出词元才能开始，
所以任一时刻只有一级在算，其余三级空转，利用率 1/4。
下半幅：同时放进 4 个微批 A、B、C、D。前 3 个时间片是填充期，此后每一级每个时间片都有活；
每个微批自己的一步仍要走完 4 级，单个请求的每词元时延不变；微批与上半幅的批一样大时，总吞吐是上半幅的 4 倍。
格子里的“A2”表示微批 A 的第 2 个 Decode 步。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_8_pp_timeline.png")

P, SLOTS = 4, 12
CW, CH = 1.85, 1.15
X0 = 4.3  # 第一个时间片的左边界


def schedule(n_micro: int) -> dict[tuple[int, int], str]:
    """返回 {(级, 时间片): 标签}。微批 j 在时间片 j 进入第 1 级，走完第 P 级后立刻开始下一步。"""
    cells = {}
    for j in range(n_micro):
        name = "ABCD"[j]
        step, t = 1, j
        while t < SLOTS:
            for s in range(P):
                if t + s < SLOTS:
                    cells[(s, t + s)] = f"{name}{step}"
            t += max(P, n_micro)
            step += 1
    return cells


def panel(ax, top, title, n_micro, note):
    label(ax, 0.2, top, title, ha="left", fontsize=FS_TITLE, bold=True)
    grid_top = top - 1.5
    cells = schedule(n_micro)
    for t in range(SLOTS):
        label(ax, X0 + (t + 0.5) * CW, grid_top + 0.4, str(t + 1), fontsize=FS_SMALL, color=MUTED)
    label(ax, X0 - 0.2, grid_top + 0.4, "时间片", ha="right", fontsize=FS_SMALL, color=MUTED)
    busy = 0
    for s in range(P):
        y = grid_top - (s + 1) * CH
        label(ax, X0 - 0.2, y + CH / 2, f"第 {s + 1} 级（卡 {s}）", ha="right", fontsize=FS_TEXT)
        for t in range(SLOTS):
            text = cells.get((s, t), "")
            if text:
                busy += 1
                kind = "new" if text.startswith("A") else "data"
                box(ax, X0 + t * CW, y, CW, CH, text, kind, fontsize=FS_TEXT, rounded=False, lw=1.0)
            else:
                box(ax, X0 + t * CW, y, CW, CH, "", "neutral", rounded=False, lw=0.6)
    label(ax, X0, grid_top - P * CH - 0.6, note.format(busy=busy, total=P * SLOTS,
                                                       pct=100 * busy / (P * SLOTS)),
          ha="left", fontsize=FS_SMALL, color=INK)
    return grid_top - P * CH - 0.6


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 7.6))
    b1 = panel(ax, 15.0, "流水线里只有一个批：任一时刻只有一级在算", 1,
               "灰格是空转。{total} 格里有 {busy} 格在算，利用率 {pct:.0f}%；"
               "A 每 4 个时间片产出一个词元。")
    ax.plot([0.2, X0 + SLOTS * CW], [b1 - 0.8, b1 - 0.8], color="#c9c8c2", lw=1.0)
    b2 = panel(ax, b1 - 1.8, "同时放进 4 个微批：填充期过后各级都不空转", 4,
               "{total} 格里有 {busy} 格在算（前 3 个时间片是填充期）。"
               "微批与上图的批一样大时，总吞吐是上图的 4 倍。")
    label(ax, X0, b2 - 0.85,
          "青绿色标出微批 A 的轨迹：A2 必须等 A1 走完第 4 级、采样出词元之后才能进入第 1 级。",
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    finish(fig, ax, OUTPUT, xlim=(-0.2, X0 + SLOTS * CW + 0.4), ylim=(b2 - 1.5, 15.8))


if __name__ == "__main__":
    main()
