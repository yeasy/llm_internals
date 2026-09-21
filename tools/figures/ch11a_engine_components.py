"""生成图 11-1：推理引擎的组件，以及一个请求走过的路径。

正文位置：11_serving/11.1_engines_overview.md
输出：11_serving/_images/ch11a_engine_components.png

上排是每个请求只经过一次的入口和出口；中排是引擎的迭代循环，每一轮从调度器开始、
到输出处理结束；KV 缓存管理器挂在调度器一侧，因为“还剩多少空闲块”是准入的依据。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, MUTED, NEW_EDGE, arrow, box, finish,
                      label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11a_engine_components.png")

W, H = 8.6, 3.8
XS = (0.0, 11.6, 23.2, 34.8)          # 四列的左边界
Y_TOP, Y_LOOP, Y_MEM = 16.4, 8.6, 0.4
Y_ADMIT, Y_BACK = 14.6, 7.0


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 7.6))

    # ---- 上排：每个请求只经过一次 ----
    label(ax, -0.2, Y_TOP + H + 1.0, "入口与出口：每个请求只走一遍（CPU）", fontsize=FS_NAME,
          bold=True, ha="left")
    box(ax, XS[0], Y_TOP, W, H, "API 服务层\n接收 HTTP 请求\n校验参数", "neutral")
    box(ax, XS[1], Y_TOP, W, H, "输入处理\n套聊天模板\n分词成词元 ID", "neutral")
    box(ax, XS[2], Y_TOP, W, H, "等待队列\n按到达顺序\n或优先级排队", "neutral")
    box(ax, XS[3], Y_TOP, W, H, "流式返回\nAPI 服务层用 SSE\n逐段推给客户端", "neutral")
    for k in (0, 1):
        arrow(ax, (XS[k] + W, Y_TOP + H / 2), (XS[k + 1], Y_TOP + H / 2))

    # 等待队列 → 调度器（折线）
    xq, xs = XS[2] + W / 2, XS[0] + W / 2
    arrow(ax, (xq, Y_TOP), (xq, Y_ADMIT), style="-")
    arrow(ax, (xq, Y_ADMIT), (xs, Y_ADMIT), style="-")
    arrow(ax, (xs, Y_ADMIT), (xs, Y_LOOP + H))
    label(ax, (xq + xs) / 2, Y_ADMIT + 0.6, "准入：本轮词元预算和空闲 KV 块都够，才放进循环",
          fontsize=FS_SMALL, color=MUTED)

    # ---- 中排：迭代循环 ----
    label(ax, 10.2, Y_LOOP + H + 1.0, "迭代循环：每轮一次前向，运行中的请求各前进一步",
          fontsize=FS_NAME, bold=True, ha="left")
    box(ax, XS[0], Y_LOOP, W, H, "调度器\n选出本轮的请求\n和各自的词元数", "neutral")
    box(ax, XS[1], Y_LOOP, W, H, "执行器（GPU）\n用权重做全部 L 层\n的一次前向", "weight")
    box(ax, XS[2], Y_LOOP, W, H, "采样器\n每个请求的 logits\n→ 一个新词元", "data")
    box(ax, XS[3], Y_LOOP, W, H, "输出处理\n增量反分词\n判断是否该停", "neutral")
    for k in range(3):
        arrow(ax, (XS[k] + W, Y_LOOP + H / 2), (XS[k + 1], Y_LOOP + H / 2))

    # 输出处理 → 流式返回
    xo = XS[3] + W / 2
    arrow(ax, (xo, Y_LOOP + H), (xo, Y_TOP), color=NEW_EDGE, lw=1.9)
    label(ax, xo - 0.4, (Y_LOOP + H + Y_TOP) / 2, "新词元\n的文本", fontsize=FS_SMALL,
          ha="right")

    # 输出处理 → 调度器（下一轮）
    arrow(ax, (xo, Y_LOOP), (xo, Y_BACK), style="-", color=ACCENT, lw=1.6)
    arrow(ax, (xo, Y_BACK), (XS[0] + 6.6, Y_BACK), style="-", color=ACCENT, lw=1.6)
    arrow(ax, (XS[0] + 6.6, Y_BACK), (XS[0] + 6.6, Y_LOOP), color=ACCENT, lw=1.6)
    label(ax, 31.0, Y_BACK - 0.65, "没结束的请求留到下一轮；结束的请求释放 KV 块",
          fontsize=FS_SMALL, color=ACCENT)

    # ---- 下排：显存里的 KV ----
    box(ax, XS[0], Y_MEM, W, H, "KV 缓存管理器\n记空闲块的账\n分配与释放", "neutral")
    arrow(ax, (XS[0] + 2.4, Y_LOOP), (XS[0] + 2.4, Y_MEM + H), style="<|-|>")
    label(ax, XS[0] + 2.75, 5.35, "还剩几块", fontsize=FS_SMALL, color=MUTED, ha="left")

    box(ax, XS[1], Y_MEM, W, H, "KV 块池（显存）\n所有请求的 K、V\n按块存放", "data")
    arrow(ax, (XS[1] + W / 2, Y_LOOP), (XS[1] + W / 2, Y_MEM + H), style="<|-|>")
    label(ax, XS[1] + W / 2 + 0.35, 5.35, "注意力内核按块读写", fontsize=FS_SMALL,
          color=MUTED, ha="left")
    arrow(ax, (XS[0] + W, Y_MEM + H / 2), (XS[1], Y_MEM + H / 2), color=MUTED)
    label(ax, (XS[0] + W + XS[1]) / 2, Y_MEM + H / 2 + 0.55, "块号", fontsize=FS_SMALL,
          color=MUTED)

    # 图例
    for i, (kind, text) in enumerate((("neutral", "CPU 侧的控制逻辑"),
                                      ("weight", "权重，以及用权重做的前向"),
                                      ("data", "随请求变化的数据"))):
        y = Y_MEM + H - 1.0 - i * 1.15
        box(ax, XS[2] + 2.0, y, 0.9, 0.7, "", kind, rounded=False, lw=1.2)
        label(ax, XS[2] + 3.3, y + 0.35, text, fontsize=FS_SMALL, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.6, XS[3] + W + 0.6), ylim=(0.0, Y_TOP + H + 1.8))


if __name__ == "__main__":
    main()
