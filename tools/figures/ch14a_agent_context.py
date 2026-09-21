"""生成 14.5 节的图：Agent 循环里上下文怎样逐轮变长，前缀缓存省掉了哪一块。

正文位置：14_future_trends/14.5_agent_tool_use.md
输出：14_future_trends/_images/ch14a_agent_context.png

设定：工具定义 10,000 词元（50 个工具，每个约 200 词元），此后每轮新增 2,048
词元（模型的调用片段加工具回灌的结果）。左图画前 6 轮的上下文构成；右图把 30 轮
的预填充总量画成两根柱子。数值由 recompute.py 算出：
  无缓存 sum_{k=1..30}(10000 + 2048k) = 300,000 + 952,320 = 1,252,320 词元；
  全命中 2048 * 30 = 61,440 词元；比值 20.4。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14a_agent_context.png")

ROUNDS = 6
PER_ROUND = 2048
TOOLS = 10000
UNIT = 0.00042          # 词元 -> 绘图单位


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 6.2))

    label(ax, 0.0, 12.4, "Agent 循环：每一轮都要把前面所有内容重新读一遍",
          ha="left", fontsize=FS_TITLE, bold=True)

    rh, pitch = 0.95, 1.55
    top = 10.6
    for r in range(1, ROUNDS + 1):
        y = top - (r - 1) * pitch - rh
        label(ax, -0.3, y + rh / 2, f"第 {r} 轮", ha="right", fontsize=FS_TEXT)
        x = 0.0
        w_tools = TOOLS * UNIT
        box(ax, x, y, w_tools, rh, "工具定义 1 万词元" if r == 1 else "", "weight",
            fontsize=FS_SMALL, rounded=False)
        x += w_tools
        cached = (r - 1) * PER_ROUND * UNIT
        if cached > 0:
            box(ax, x, y, cached, rh,
                f"已有 {(r - 1) * PER_ROUND:,}" if cached > 2.8 else "",
                "data", fontsize=FS_SMALL, rounded=False)
            x += cached
        box(ax, x, y, PER_ROUND * UNIT, rh, "新增", "new", fontsize=FS_SMALL,
            rounded=False)
        label(ax, x + PER_ROUND * UNIT + 0.35, y + rh / 2,
              f"{TOOLS + r * PER_ROUND:,} 词元", ha="left", fontsize=FS_SMALL, color=MUTED)

    ybot = top - ROUNDS * pitch
    label(ax, 0.0, ybot - 0.45,
          "橙色与蓝色合起来是本轮的前缀：与上一轮的前缀逐字节相同，\n"
          "命中前缀缓存就不必重算。青绿色才是本轮真正新增的部分。",
          ha="left", fontsize=FS_TEXT, color=ACCENT)
    label(ax, 0.0, ybot - 2.0,
          "引擎并不在工具等待期间挂起请求：上一轮结束，\n"
          "下一轮带着更长的前缀重新进入。",
          ha="left", fontsize=FS_TEXT, color=INK)

    # 右侧柱状对比
    bx, by = 14.6, ybot - 2.4
    label(ax, bx, 11.4, "30 轮下来，预填充一共要算多少词元", ha="left",
          fontsize=FS_TITLE, bold=True)
    no_cache = sum(TOOLS + k * PER_ROUND for k in range(1, 31))
    full_hit = PER_ROUND * 30
    scale = 11.2 / no_cache
    for i, (name, val, kind) in enumerate(
            [("无前缀缓存", no_cache, "data"), ("前缀全命中", full_hit, "new")]):
        x = bx + i * 3.6
        box(ax, x, by, 2.4, val * scale, "", kind, rounded=False, lw=1.3)
        ax.text(x + 1.2, by + val * scale + 0.42, f"{val:,}", ha="center", va="center",
                fontsize=FS_NAME, color=ACCENT, fontweight="bold", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        label(ax, x + 1.2, by - 0.45, name, fontsize=FS_TEXT)
    label(ax, bx + 6.9, by + 5.6, f"相差 {no_cache / full_hit:.1f} 倍", ha="left",
          fontsize=FS_TEXT, color=ACCENT, bold=True)

    finish(fig, ax, OUTPUT, xlim=(-1.9, 23.6), ylim=(ybot - 3.0, 12.9))


if __name__ == "__main__":
    main()
