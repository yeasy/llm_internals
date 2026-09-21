"""生成 11.3 节插图：一个请求在调度器里的状态机。

正文位置：11_serving/11.3_scheduler_loop.md
输出：11_serving/_images/ch11_3_request_states.png

四个状态：等待、运行、被抢占、完成。每个方框下方写明该状态占不占 KV 显存、
占不占每步的词元预算；箭头上写触发条件。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import ACCENT, FS_NAME, FS_SMALL, INK, MUTED, arrow, box, finish, label
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_3_request_states.png")

BW, BH = 5.6, 2.2


def state(ax, x, y, name, en, note, kind):
    box(ax, x, y, BW, BH, f"{name}\n{en}", kind, fontsize=FS_NAME, bold=True)
    label(ax, x + BW / 2, y - 0.35, note, fontsize=FS_SMALL, color=MUTED, va="top")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    xw, xr, xf, y0, y1 = 0.6, 13.0, 25.4, 7.0, 0.2
    state(ax, xw, y0, "等待", "WAITING", "不占 KV，不占预算", "neutral")
    state(ax, xr, y0, "运行", "RUNNING", "占 KV；每步占词元预算", "data")
    state(ax, xf, y0, "完成", "FINISHED", "KV 全部归还", "plain")
    state(ax, xr, y1, "被抢占", "PREEMPTED", "KV 已释放；已生成的词元保留", "new")

    ym = y0 + BH / 2
    # 到达 -> 等待
    arrow(ax, (xw + BW / 2, y0 + BH + 2.0), (xw + BW / 2, y0 + BH + 0.05))
    label(ax, xw + BW / 2 + 0.6, y0 + BH + 2.4, "请求到达：套模板、分词、入队", fontsize=FS_SMALL)
    # 等待 -> 运行
    arrow(ax, (xw + BW + 0.05, ym), (xr - 0.05, ym))
    label(ax, (xw + BW + xr) / 2, ym + 1.75,
          "准入，三项都够：\n词元预算、并发上限、\nKV 空间", fontsize=FS_SMALL)
    # 运行 自环
    arrow(ax, (xr + 1.6, y0 + BH + 0.05), (xr + BW - 1.6, y0 + BH + 0.05), rad=-1.1)
    label(ax, xr + BW / 2, y0 + BH + 2.4,
          "每步分到 n 个词元，已算词元数前移 n", fontsize=FS_SMALL)
    # 运行 -> 完成
    arrow(ax, (xr + BW + 0.05, ym), (xf - 0.05, ym))
    label(ax, (xr + BW + xf) / 2, ym + 1.75,
          "选中 EOS、到长度上限、\n命中停止串，\n或客户端取消", fontsize=FS_SMALL)
    # 运行 -> 被抢占
    arrow(ax, (xr + BW / 2 + 1.2, y0 - 0.95), (xr + BW / 2 + 1.2, y1 + BH + 0.05), color=ACCENT)
    label(ax, xr + BW / 2 + 1.5, (y0 + y1 + BH) / 2 - 0.45,
          "KV 空间不够：\n从运行队尾选中，\n释放全部 KV，已算词元数归零",
          fontsize=FS_SMALL, color=ACCENT, ha="left")
    # 被抢占 -> 等待
    arrow(ax, (xr - 0.05, y1 + BH / 2), (xw + BW / 2, y0 - 0.95), color=ACCENT, rad=-0.25)
    label(ax, 4.6, 0.75,
          "插回等待队列队首；\n再次准入时把 Prompt 与\n已生成词元一起重算",
          fontsize=FS_SMALL, color=ACCENT, ha="center")

    label(ax, xf + BW / 2, y1 + BH / 2 + 0.2,
          "占 KV 的只有“运行”一种状态，\n抢占因此等价于\n“用重算换显存”",
          fontsize=FS_SMALL, color=INK)

    finish(fig, ax, OUTPUT, xlim=(0, 31.6), ylim=(-1.4, 12.4))


if __name__ == "__main__":
    main()
