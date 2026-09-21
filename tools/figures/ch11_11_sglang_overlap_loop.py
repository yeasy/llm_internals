"""生成 11.11 节插图：SGLang 调度主循环的两种排法。

正文位置：11_serving/11.11_sglang_internals.md
输出：11_serving/_images/ch11_11_sglang_overlap_loop.png

(a) event_loop_normal：调度、前向、处理结果首尾相接，CPU 干活时 GPU 空等。
(b) event_loop_overlap：第 n+1 批的前向在第 n 批的结果处理之前发射，采样出的词元 ID
    留在 GPU 上直接交给下一批；带语法的批把采样推迟到上一批结果处理（accept_token）之后。
时间为示意，不按比例；先后关系取自 SGLang v0.5.20 的 managers/scheduler.py。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, MASK_EDGE, MASK_FACE, MUTED, arrow, box,
                      finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_11_sglang_overlap_loop.png")

LANE_H = 1.25


def lanes(ax, y_cpu, y_gpu):
    label(ax, 2.7, y_cpu + LANE_H / 2, "CPU", ha="right", fontsize=FS_SMALL, color=MUTED)
    label(ax, 2.7, y_gpu + LANE_H / 2, "GPU", ha="right", fontsize=FS_SMALL, color=MUTED)


def cpu(ax, x, w, y, text):
    box(ax, x, y, w, LANE_H, text, kind="neutral", fontsize=FS_SMALL, rounded=False,
        linespacing=1.15)


def gpu(ax, x, y, text):
    """一步前向（5 个单位）加采样（1 个单位）。"""
    box(ax, x, y, 5.0, LANE_H, text, kind="data", fontsize=FS_SMALL, rounded=False)
    box(ax, x + 5.0, y, 1.0, LANE_H, "采\n样", kind="new", fontsize=FS_SMALL, rounded=False,
        linespacing=1.0)


def idle(ax, x, w, y):
    ax.add_patch(Rectangle((x, y), w, LANE_H, facecolor=MASK_FACE, edgecolor=MASK_EDGE,
                           linewidth=1.0, hatch="///", zorder=2))
    label(ax, x + w / 2, y + LANE_H / 2, "空等", fontsize=FS_SMALL, color=MUTED)


def panel_normal(ax):
    y_cpu, y_gpu = 10.9, 8.95
    label(ax, 0.2, 12.9, "(a) event_loop_normal：调度、前向、处理结果首尾相接", ha="left",
          fontsize=FS_TEXT, bold=True)
    lanes(ax, y_cpu, y_gpu)
    cpu(ax, 3.0, 2.4, y_cpu, "调度\nn")
    gpu(ax, 5.4, y_gpu, "前向 n")
    cpu(ax, 11.4, 2.0, y_cpu, "处理\n结果 n")
    cpu(ax, 13.4, 2.4, y_cpu, "调度\nn+1")
    idle(ax, 11.4, 4.4, y_gpu)
    gpu(ax, 15.8, y_gpu, "前向 n+1")
    cpu(ax, 21.8, 2.0, y_cpu, "处理\n结果 n+1")


def panel_overlap(ax):
    y_cpu, y_gpu = 5.0, 3.05
    label(ax, 0.2, 7.2, "(b) event_loop_overlap：下一批先发射，上一批的结果后处理", ha="left",
          fontsize=FS_TEXT, bold=True)
    lanes(ax, y_cpu, y_gpu)
    for k, name in enumerate(("n", "n+1", "n+2")):
        x = 3.0 + 6.0 * k
        gpu(ax, x, y_gpu, f"前向 {name}")
    for k, (done, nxt) in enumerate((("n-1", "n+1"), ("n", "n+2"), ("n+1", "n+3"))):
        x = 3.2 + 6.0 * k
        cpu(ax, x, 2.0, y_cpu, f"处理\n结果 {done}")
        cpu(ax, x + 2.0, 2.4, y_cpu, f"调度\n{nxt}")
    # 词元 ID 在 GPU 上直接交给下一批
    arrow(ax, (8.5, y_gpu), (10.4, y_gpu), color=ACCENT, lw=1.6, rad=0.9)
    label(ax, 10.9, y_gpu - 0.75, "采样出的词元 ID 留在 GPU 上，经 FutureMap 交给前向 n+1",
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    # 带语法的批：采样等上一批的 accept_token
    arrow(ax, (10.2, y_cpu), (14.45, y_gpu + LANE_H), color=MUTED, lw=1.5)
    label(ax, 3.0, y_gpu - 1.75,
          "灰箭头：带语法的批，采样 n+1 要等“处理结果 n”里的 accept_token；前向 n+1 不必等。",
          ha="left", fontsize=FS_SMALL, color=MUTED)
    label(ax, 3.0, y_gpu - 2.5,
          "代价：请求在第 n 步生成 EOS，CPU 处理结果 n 时才知道，此时前向 n+1 已经把它算了进去。",
          ha="left", fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    panel_normal(ax)
    panel_overlap(ax)
    finish(fig, ax, OUTPUT, xlim=(0, 24.4), ylim=(0.1, 13.5))
    plt.close(fig)


if __name__ == "__main__":
    main()
