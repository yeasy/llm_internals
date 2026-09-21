"""生成图 3-10：回答“5。”的三次前向计算。

正文位置：03_components/3.8_gpt_inference_flow.md
输出：03_components/_images/inference_timeline.png

三列对应三次前向：Prefill 一次送进 6 个 Prompt 词元，之后每轮 Decode 只送进上一轮
刚选出的那一个词元；KV 缓存从 6 行长到 7 行再到 8 行；选中 EOS 后停止。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED,
                      NEW_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "inference_timeline.png")

COL_W = 13.4      # 每一列的宽度
Y_HEAD, Y_IN, Y_MODEL, Y_CACHE, Y_SEEN = 15.2, 12.2, 7.6, 4.2, 0.9


def chips(ax, x, y, tokens, kind):
    """从 x 开始画一排词元方块，返回 (右边界, 各方块中心 x)。"""
    centers = []
    for text, w in tokens:
        box(ax, x, y, w, 1.1, text, kind, fontsize=FS_SMALL - 1)
        centers.append(x + w / 2)
        x += w + 0.14
    return x - 0.14, centers


def cache_strip(ax, x, y, old, new):
    """画 KV 缓存：old 个已有的行（蓝）加 new 个本轮追加的行（青绿）。"""
    cell = 0.78
    for i in range(old + new):
        box(ax, x + i * cell, y, cell, 0.9, str(i + 1), "new" if i >= old else "data",
            fontsize=FS_SMALL - 1, rounded=False, lw=1.0)
    return x + (old + new) * cell


def column(ax, k, *, order, title, subtitle, tokens, token_kind, in_note, cache_old, cache_new,
           cache_note, out_text, out_note, seen):
    x = k * COL_W
    label(ax, x + 5.2, Y_HEAD + 1.15, order, fontsize=FS_TEXT, color=MUTED)
    label(ax, x + 5.2, Y_HEAD, title, fontsize=FS_TITLE, bold=True)
    label(ax, x + 5.2, Y_HEAD - 1.1, subtitle, fontsize=FS_SMALL, color=MUTED)

    # 输入词元
    total = sum(w for _, w in tokens) + 0.14 * (len(tokens) - 1)
    right, _ = chips(ax, x + 5.2 - total / 2, Y_IN, tokens, token_kind)
    label(ax, x + 5.2, Y_IN - 0.55, in_note, fontsize=FS_SMALL, color=MUTED)
    arrow(ax, (x + 5.2, Y_IN - 1.0), (x + 5.2, Y_MODEL + 2.45))

    # 模型
    box(ax, x + 1.2, Y_MODEL, 7.6, 2.4, "模型：全部 L 层\n每次用的是同一套权重", "weight",
        fontsize=FS_TEXT)

    # KV 缓存
    arrow(ax, (x + 4.2, Y_MODEL - 0.05), (x + 4.2, Y_CACHE + 1.0), style="<|-|>")
    start = x + 5.2 - (cache_old + cache_new) * 0.78 / 2
    cache_strip(ax, start, Y_CACHE, cache_old, cache_new)
    label(ax, x + 5.2, Y_CACHE - 0.6, cache_note, fontsize=FS_SMALL, color=MUTED)

    # 输出词元
    arrow(ax, (x + 8.85, Y_MODEL + 1.2), (x + 10.0, Y_MODEL + 1.2))
    box(ax, x + 10.05, Y_MODEL + 0.6, 1.9, 1.2, out_text, "new", fontsize=FS_NAME, bold=True)
    label(ax, x + 11.0, Y_MODEL + 0.1, out_note, fontsize=FS_SMALL, color=MUTED, va="top")

    # 用户看到的文本
    box(ax, x + 1.6, Y_SEEN, 7.2, 1.2, seen, "plain", fontsize=FS_TEXT)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 5.9))

    prompt = [("〈user〉", 2.35), ("2", 0.9), ("+", 0.9), ("3", 0.9), ("=", 0.9),
              ("〈assistant〉", 3.6)]
    column(ax, 0, order="第 1 次前向", title="Prefill", subtitle="一次读完整个 Prompt",
           tokens=prompt, token_kind="data", in_note="送进模型：6 个词元",
           cache_old=0, cache_new=6, cache_note="KV 缓存：写入 6 行",
           out_text="5", out_note="选出 $y_1$", seen="用户看到：5")
    column(ax, 1, order="第 2 次前向", title="Decode 第 1 轮", subtitle="只处理刚选出的词元",
           tokens=[("5", 1.1)], token_kind="new", in_note="送进模型：1 个词元",
           cache_old=6, cache_new=1, cache_note="读前 6 行，追加第 7 行",
           out_text="。", out_note="选出 $y_2$", seen="用户看到：5。")
    column(ax, 2, order="第 3 次前向", title="Decode 第 2 轮", subtitle="只处理刚选出的词元",
           tokens=[("。", 1.1)], token_kind="new", in_note="送进模型：1 个词元",
           cache_old=7, cache_new=1, cache_note="读前 7 行，追加第 8 行",
           out_text="EOS", out_note="结束标记\n→ 停止", seen="最终文本：5。")

    # 上一次的输出成为下一次的输入
    for k in (0, 1):
        x = k * COL_W
        arrow(ax, (x + 11.0, Y_MODEL + 1.85), (x + COL_W + 4.55, Y_IN + 0.55),
              color=NEW_EDGE, lw=1.8, rad=-0.28)
        label(ax, x + 12.75, Y_IN - 1.45, "成为下一次\n的输入", fontsize=FS_SMALL, color=INK)

    label(ax, 2 * COL_W + 5.2, -0.55,
          "没选中 EOS，Decode 就一轮一轮继续",
          fontsize=FS_SMALL, color=ACCENT)

    # 图例：颜色在本节所有图里含义一致
    for i, (kind, text, lx) in enumerate((("data", "已有的词元、缓存里已有的行", -0.2),
                                          ("new", "本轮新选出的词元、新写入的行", 13.2),
                                          ("weight", "模型权重", 27.6))):
        box(ax, lx, -2.35, 0.9, 0.7, "", kind, rounded=False, lw=1.2)
        label(ax, lx + 1.2, -2.0, text, fontsize=FS_SMALL, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.6, 3 * COL_W - 0.6), ylim=(-2.9, 17.4))


if __name__ == "__main__":
    main()
