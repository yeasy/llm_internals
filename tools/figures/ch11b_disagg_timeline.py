"""图：分离式 Prefill-Decode 下一个请求的时间线（11.9.2）。

五条泳道从上到下：用户、全局调度器、Prefill 实例、网络、Decode 实例。要表达三件事：
1. 首词元由 Prefill 侧选出，不等 KV 传输；
2. KV 逐层发出，传输与后面各层的计算重叠；
3. 用户可见的额外等待只剩最后几层 KV 的“尾巴”，落在第二个词元上。

时间轴是示意，不按比例；量级见正文表 11-@11.9-2@。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE, arrow, box,
                      dots, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11b_disagg_timeline.png")

LANES = [("用户", 9.6), ("全局调度器", 7.7), ("Prefill 实例", 5.8), ("网络", 3.9), ("Decode 实例", 2.0)]
H = 1.0  # 方框高度


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10, 4.4))

    # 泳道底线与名称
    for name, y in LANES:
        ax.plot([4.5, 31.5], [y - 0.25, y - 0.25], color="#d9d8d3", lw=1.0, zorder=0)
        label(ax, 4.2, y + H / 2, name, fontsize=FS_NAME, ha="right", bold=True)
    y_user, y_sched, y_pre, y_net, y_dec = (y for _, y in LANES)

    # 1. 请求到达，调度器选一对实例
    box(ax, 4.8, y_user, 2.8, H, "发出请求", "plain", fontsize=FS_SMALL)
    box(ax, 5.4, y_sched, 4.6, H, "① 选一对实例", "neutral", fontsize=FS_SMALL)
    arrow(ax, (6.2, y_user), (6.2, y_sched + H), color=MUTED)
    arrow(ax, (8.4, y_sched), (8.4, y_pre + H), color=MUTED)
    arrow(ax, (6.2, y_sched), (6.2, y_dec + H), color=MUTED, lw=1.0)

    # 2. Prefill 逐层计算，最后选出首词元
    w = 2.8
    layer_x = [7.0, 10.0, 15.0]
    for x, n in zip(layer_x, ["第 1 层", "第 2 层", "第 L 层"]):
        box(ax, x, y_pre, w, H, n, "data", fontsize=FS_SMALL)
    dots(ax, 13.9, y_pre + H / 2, color=INK, spread=0.34, r=0.09)
    label(ax, 12.4, y_pre + H + 0.38, "② Prefill：逐层算出 KV", fontsize=FS_SMALL, color=MUTED)
    pre_end = 21.2
    box(ax, 18.0, y_pre, pre_end - 18.0, H, "选出首词元", "new", fontsize=FS_SMALL)

    # 3. KV 逐层发出：每层算完即发，与后面各层的计算重叠
    net = [(9.9, 3.3, "第 1 层 KV"), (13.3, 3.3, "第 2 层 KV"), (18.0, 4.4, "第 L 层 KV")]
    for (x1, sw, n), x0 in zip(net, layer_x):
        box(ax, x1, y_net, sw, H, n, "data", fontsize=FS_SMALL)
        arrow(ax, (x0 + w - 0.1, y_pre), (x1 + 0.3, y_net + H), color=MUTED, lw=1.0)
    dots(ax, 17.3, y_net + H / 2, color=INK, spread=0.34, r=0.09)
    label(ax, 14.5, y_net - 0.62, "③ 传输与计算重叠", fontsize=FS_SMALL, color=MUTED)
    tail_end = 22.4

    # 尾巴：Prefill 已结束而最后几层的 KV 还在路上
    for x in (pre_end, tail_end):
        ax.plot([x, x], [y_dec + H + 0.05, y_user - 0.2], color=ACCENT, lw=1.1,
                ls=(0, (4, 3)), zorder=0)

    # 4. Decode 侧：预分配块、接收 KV、进入运行队列
    box(ax, 4.8, y_dec, 4.6, H, "④ 预分配 KV 块", "neutral", fontsize=FS_SMALL)
    box(ax, 9.6, y_dec, tail_end - 9.6, H, "接收 KV，同时照常为别的请求 Decode", "plain",
        fontsize=FS_SMALL)
    dec_x = [22.6, 25.6, 28.6]
    for x, n in zip(dec_x, ["⑤ 第 2 个", "第 3 个", "第 4 个"]):
        box(ax, x, y_dec, 2.9, H, n, "new", fontsize=FS_SMALL)

    # 返回用户
    arrow(ax, (19.6, y_pre + H), (19.6, y_user), color=NEW_EDGE)
    box(ax, 18.2, y_user, 2.8, H, "首词元", "new", fontsize=FS_SMALL)
    for x, n in zip(dec_x, ["第 2 个", "第 3 个", "第 4 个"]):
        arrow(ax, (x + 1.45, y_dec + H), (x + 1.45, y_user), color=NEW_EDGE, lw=1.0)
        box(ax, x + 0.2, y_user, 2.5, H, n, "new", fontsize=FS_SMALL)

    # 标注：TTFT 与“尾巴”
    ax.annotate("", xy=(21.0, y_user + H + 0.55), xytext=(4.8, y_user + H + 0.55),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.2))
    label(ax, 12.9, y_user + H + 1.0, "TTFT：排队 + Prefill，不含 KV 传输", fontsize=FS_TEXT)
    label(ax, 31.5, y_dec - 0.85, "两条紫色虚线之间：没被计算遮住的传输尾巴，只推迟第 2 个词元",
          fontsize=FS_SMALL, color=ACCENT, ha="right")

    finish(fig, ax, OUTPUT, xlim=(0.2, 31.8), ylim=(0.6, 12.4))


if __name__ == "__main__":
    main()
