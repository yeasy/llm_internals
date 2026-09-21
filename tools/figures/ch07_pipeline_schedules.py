"""生成 7.4 节的图：4 级流水线、8 个微批量时，GPipe 与 1F1B 两种调度的时间线。

正文位置：07_distributed_training/7.4_pipeline_hybrid.md
输出：07_distributed_training/_images/ch07_pipeline_schedules.png

横轴是时间。一个前向格（F，蓝）宽 1 个单位，一个反向格（B，橙）宽 2 个单位，格内数字是微批量编号；
灰色是气泡。两种调度的总时长都是 (m + p - 1)(t_f + t_b) = 11 x 3 = 33 个单位，气泡相同；
区别在每一级同时持有几个微批量的激活：GPipe 是 m = 8 份，1F1B 第 i 级是 p - i + 1 份。

调度由依赖关系模拟得出，不是手摆的：F(级 i, 微批 j) 要等 F(级 i-1, j) 完成，
B(级 i, j) 要等 B(级 i+1, j) 完成；每一级按自己的操作顺序串行执行。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, FS_TITLE, INK, MASK_EDGE,
                      MASK_FACE, MUTED, WEIGHT_EDGE, WEIGHT_FACE, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("07_distributed_training", "ch07_pipeline_schedules.png")

P, M = 4, 8
TF, TB = 1, 2
UW, RH = 0.62, 1.0  # 一个时间单位的宽度、一行的高度
X0 = 3.4


def order(kind: str, stage: int):
    if kind == "gpipe":
        return [("F", j) for j in range(M)] + [("B", j) for j in range(M)]
    warm = min(P - 1 - stage, M)
    ops = [("F", j) for j in range(warm)]
    f, b = warm, 0
    while f < M:
        ops += [("F", f), ("B", b)]
        f, b = f + 1, b + 1
    ops += [("B", j) for j in range(b, M)]
    return ops


def simulate(kind: str):
    """返回 {(级, 操作, 微批): (开始, 结束)}。"""
    queues = [order(kind, i) for i in range(P)]
    done: dict[tuple[int, str, int], tuple[float, float]] = {}
    free = [0.0] * P
    pos = [0] * P
    progressed = True
    while progressed:
        progressed = False
        for i in range(P):
            while pos[i] < len(queues[i]):
                op, j = queues[i][pos[i]]
                dep = (i - 1, "F", j) if op == "F" else (i + 1, "B", j)
                if 0 <= dep[0] < P and dep not in done:
                    break
                ready = done[dep][1] if 0 <= dep[0] < P else 0.0
                start = max(free[i], ready)
                end = start + (TF if op == "F" else TB)
                done[(i, op, j)] = (start, end)
                free[i] = end
                pos[i] += 1
                progressed = True
    return done


def in_flight(done, stage):
    """第 stage 级同时持有激活的微批量数的峰值。"""
    events = []
    for (i, op, j), (s, e) in done.items():
        if i == stage:
            events.append((s, 1) if op == "F" else (e, -1))
    peak = cur = 0
    for _, d in sorted(events, key=lambda t: (t[0], t[1])):
        cur += d
        peak = max(peak, cur)
    return peak


def panel(ax, top, title, kind):
    done = simulate(kind)
    total = max(e for _, e in done.values())
    label(ax, 0.0, top + 0.9, title, ha="left", fontsize=FS_TITLE, bold=True)
    for i in range(P):
        y = top - (i + 1) * RH
        label(ax, X0 - 0.25, y + RH / 2, f"第 {i + 1} 级", ha="right", fontsize=FS_TEXT)
        ax.add_patch(Rectangle((X0, y), total * UW, RH, facecolor=MASK_FACE, edgecolor=MASK_EDGE,
                               linewidth=0.6, zorder=1))
        for (s_i, op, j), (s, e) in done.items():
            if s_i != i:
                continue
            face, edge = (DATA_FACE, DATA_EDGE) if op == "F" else (WEIGHT_FACE, WEIGHT_EDGE)
            ax.add_patch(Rectangle((X0 + s * UW, y), (e - s) * UW, RH, facecolor=face,
                                   edgecolor=edge, linewidth=0.9, zorder=2))
            label(ax, X0 + (s + e) / 2 * UW, y + RH / 2, str(j + 1), fontsize=FS_SMALL)
        label(ax, X0 + total * UW + 0.25, y + RH / 2, f"{in_flight(done, i)} 份", ha="left",
              fontsize=FS_TEXT, color=ACCENT)
    label(ax, X0 + total * UW + 0.25, top + 0.35, "激活峰值", ha="left", fontsize=FS_SMALL,
          color=ACCENT)
    busy = M * (TF + TB)
    label(ax, X0, top - P * RH - 0.45,
          f"总时长 {total:.0f} 个单位，每级忙 {busy} 个单位，气泡 {total - busy:.0f} 个单位："
          f"占理想时间 {(total - busy) / busy:.1%}，占总时长 {(total - busy) / total:.1%}。",
          ha="left", fontsize=FS_SMALL)
    return top - P * RH - 0.45, total


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    b1, total = panel(ax, 12.0, "GPipe：先做完全部前向，再做全部反向", "gpipe")
    b2, _ = panel(ax, b1 - 2.7, "1F1B：预热后一次前向、一次反向交替", "1f1b")
    ly = b2 - 1.1
    for k, (face, edge, text) in enumerate(((DATA_FACE, DATA_EDGE, "前向 F（宽 1）"),
                                            (WEIGHT_FACE, WEIGHT_EDGE, "反向 B（宽 2）"),
                                            (MASK_FACE, MASK_EDGE, "气泡（空转）"))):
        x = X0 + k * 5.2
        ax.add_patch(Rectangle((x, ly - 0.28), 0.9, 0.56, facecolor=face, edgecolor=edge))
        label(ax, x + 1.1, ly, text, ha="left", fontsize=FS_SMALL)
    label(ax, X0 + 15.6, ly, "格内数字是微批量编号", ha="left", fontsize=FS_SMALL, color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(-0.2, X0 + total * UW + 2.6), ylim=(ly - 0.8, 13.5))


if __name__ == "__main__":
    main()
