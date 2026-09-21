"""生成 11.2 节“分块 Prefill”示意图：整段 Prefill 会挡住 Decode，分块后每一步都有 Decode 搭载。

正文位置：11_serving/11.2_continuous_batching.md
输出：11_serving/_images/ch11a_chunked_prefill_timeline.png

时间轴是示意，未按比例。图上的毫秒数来自正文算例（Llama 3 8B、H100、32K Prompt、
同批 32 个 Decode 请求；按峰值算力与带宽上界估算）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, MUTED, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11a_chunked_prefill_timeline.png")

DW, PW, H = 1.25, 2.0, 2.2


def decode_cells(ax, x, y, n):
    for i in range(n):
        box(ax, x + i * DW, y, DW, H, "D", "data", rounded=False, lw=1.0, fontsize=FS_SMALL)
    return x + n * DW


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 6.2))

    # ---- 上：整段 Prefill ----
    ya = 10.2
    label(ax, 0.0, ya + H + 1.0, "整段 Prefill：长 Prompt 一次算完，这一步里别的请求只能等",
          fontsize=FS_NAME, bold=True, ha="left")
    x = decode_cells(ax, 0.0, ya, 3)
    x_p0 = x
    # 整段 Prefill 与 32 个 Decode 词元同在一步里：Decode 词元只在这一步结束时才各出一个
    box(ax, x, ya + 0.8, 26.0, H - 0.8, "32K 词元的 Prefill：一次前向，不少于 0.75 s", "new",
        rounded=False, lw=1.0, fontsize=FS_SMALL)
    box(ax, x, ya, 26.0, 0.8, "同批的 32 个 Decode 词元，与 Prefill 一起算，这一步结束才出", "data",
        rounded=False, lw=1.0, fontsize=FS_SMALL - 1)
    x = decode_cells(ax, x + 26.0, ya, 4)
    ax.annotate("", xy=(x_p0, ya - 0.55), xytext=(x_p0 + 26.0, ya - 0.55),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=1.5, shrinkA=0, shrinkB=0))
    label(ax, x_p0 + 13.0, ya - 1.25, "正在 Decode 的 32 个请求，相邻两个词元的间隔从 6 ms 拉长到 0.75 s 以上",
          fontsize=FS_SMALL, color=ACCENT)
    label(ax, 0.0, ya - 0.75, "平时一步\n约 6 ms", fontsize=FS_SMALL, color=MUTED, ha="left",
          va="top")

    # ---- 下：分块 Prefill ----
    yb = 1.6
    label(ax, 0.0, yb + H + 1.0, "分块 Prefill：每步只放一个 2,048 词元的块，Decode 词元搭在同一步里",
          fontsize=FS_NAME, bold=True, ha="left")
    x = decode_cells(ax, 0.0, yb, 3)
    x_c0 = x
    for k in range(16):
        box(ax, x, yb + 0.8, PW, H - 0.8, f"P{k + 1}", "new", rounded=False, lw=1.0,
            fontsize=FS_SMALL)
        box(ax, x, yb, PW, 0.8, "D", "data", rounded=False, lw=1.0, fontsize=FS_SMALL - 1)
        x += PW
    x_c1 = x
    decode_cells(ax, x, yb, 4)
    ax.annotate("", xy=(x_c0, yb - 0.55), xytext=(x_c1, yb - 0.55),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=1.5, shrinkA=0, shrinkB=0))
    label(ax, (x_c0 + x_c1) / 2, yb - 1.25,
          "16 步，每步约 30–63 ms；每一步 Decode 请求都照常出一个词元", fontsize=FS_SMALL,
          color=ACCENT)

    # 图例
    lx = 25.0
    box(ax, lx, 7.3, 1.0, 0.75, "", "new", rounded=False, lw=1.2)
    label(ax, lx + 1.35, 7.67, "Prefill 的词元", fontsize=FS_SMALL, ha="left")
    box(ax, lx + 8.0, 7.3, 1.0, 0.75, "", "data", rounded=False, lw=1.2)
    label(ax, lx + 9.35, 7.67, "Decode 的词元", fontsize=FS_SMALL, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.5, 41.6), ylim=(-0.4, ya + H + 1.9))


if __name__ == "__main__":
    main()
