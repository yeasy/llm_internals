"""生成 11.7 节的图：掩码计算与 GPU 前向怎样重叠，何时落到关键路径上。

正文位置：11_serving/11.7_constrained_decoding.md
输出：11_serving/_images/ch11_7_mask_overlap.png

三种排法各画两轮 Decode，时间按比例：前向统一取 Llama 3 8B 在 H100 上的单轮下界
16 GB / 3.35 TB/s = 4.8 ms；掩码取每请求 50 微秒、单线程，B = 32 时 1.6 ms，
B = 160 时 8.0 ms。数值与正文 11.7.4 的算例一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, INK, KINDS, MASK_EDGE, MASK_FACE,
                      MUTED, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_7_mask_overlap.png")

T_FWD = 16e9 / 3.35e12 * 1e3          # 4.78 ms
T_MASK_PER_REQ = 0.05                 # ms
SCALE = 1.45                          # 1 ms 画多宽
X0 = 4.4
LANE_H = 0.9


def bar(ax, t0, dur, y, text, kind, *, hatch=None):
    face, edge = KINDS[kind] if kind in KINDS else (MASK_FACE, MASK_EDGE)
    ax.add_patch(Rectangle((X0 + t0 * SCALE, y), dur * SCALE, LANE_H, facecolor=face,
                           edgecolor=edge, linewidth=1.3, hatch=hatch, zorder=3))
    if not text:
        return
    if dur * SCALE < 3.0 and not hatch:      # 条太短，字写在右侧
        label(ax, X0 + (t0 + dur) * SCALE + 0.2, y + LANE_H / 2, text, fontsize=FS_SMALL,
              ha="left")
    else:
        ax.text(X0 + (t0 + dur / 2) * SCALE, y + LANE_H / 2, text, ha="center", va="center",
                fontsize=FS_SMALL, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=face, edgecolor="none")
                if hatch else None)


def sync(ax, t, y_top, y_bot):
    x = X0 + t * SCALE
    ax.plot([x, x], [y_bot - 0.15, y_top + LANE_H + 0.15], color=ACCENT, lw=1.6,
            linestyle=(0, (4, 2)), zorder=5)


def scenario(ax, top, title, t_mask, *, overlap):
    label(ax, 0.2, top + 0.75, title, fontsize=FS_TEXT, bold=True, ha="left")
    y_gpu, y_cpu = top - 1.0, top - 2.15
    label(ax, X0 - 0.3, y_gpu + LANE_H / 2, "GPU：前向", fontsize=FS_SMALL, ha="right", color=MUTED)
    label(ax, X0 - 0.3, y_cpu + LANE_H / 2, "CPU：算掩码", fontsize=FS_SMALL, ha="right",
          color=MUTED)
    t = 0.0
    for k in range(2):
        if overlap:
            step = max(T_FWD, t_mask)
            bar(ax, t, T_FWD, y_gpu, f"前向 {T_FWD:.1f} ms", "data")
            bar(ax, t, t_mask, y_cpu, f"掩码 {t_mask:.1f} ms", "new")
            if t_mask > T_FWD:
                bar(ax, t + T_FWD, t_mask - T_FWD, y_gpu, "空等", "idle", hatch="///")
        else:
            step = T_FWD + t_mask
            bar(ax, t, t_mask, y_cpu, f"掩码 {t_mask:.1f} ms", "new")
            bar(ax, t, t_mask, y_gpu, "空等", "idle", hatch="///")
            bar(ax, t + t_mask, T_FWD, y_gpu, f"前向 {T_FWD:.1f} ms", "data")
        t += step
        sync(ax, t, y_gpu, y_cpu)
    label(ax, X0 + t * SCALE + 0.35, y_gpu + LANE_H / 2 - 0.55,
          f"每轮 {step:.1f} ms", fontsize=FS_TEXT, color=INK, ha="left")
    return step


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    label(ax, 0.2, 13.6, "掩码只依赖已生成的词元，可以与本轮前向同时算；紫色虚线是采样前的同步点",
          fontsize=FS_TITLE, bold=True, ha="left")
    m32, m160 = 32 * T_MASK_PER_REQ, 160 * T_MASK_PER_REQ
    scenario(ax, 11.6, "(a) 串行：先算掩码再前向，B = 32", m32, overlap=False)
    scenario(ax, 7.6, "(b) 重叠：掩码短于前向，B = 32，约束不增加延迟", m32, overlap=True)
    scenario(ax, 3.6, "(c) 重叠：掩码长于前向，B = 160，GPU 每轮空等 3.2 ms", m160, overlap=True)
    # 时间轴
    y = 0.55
    ax.plot([X0, X0 + 16 * SCALE], [y, y], color=MUTED, lw=1.0)
    for ms in range(0, 17, 2):
        x = X0 + ms * SCALE
        ax.plot([x, x], [y, y - 0.15], color=MUTED, lw=1.0)
        label(ax, x, y - 0.5, f"{ms}", fontsize=FS_SMALL, color=MUTED)
    label(ax, X0 + 16 * SCALE + 0.8, y - 0.5, "ms", fontsize=FS_SMALL, color=MUTED, ha="left")
    finish(fig, ax, OUTPUT, xlim=(-0.2, 31.5), ylim=(-0.6, 14.4))


if __name__ == "__main__":
    main()
