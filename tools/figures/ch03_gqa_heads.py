"""生成图 3-15：MHA、GQA 与 MQA 中 Query 头与 K/V 头的对应关系。

正文位置：03_components/3.7_full_architecture.md
输出：03_components/_images/ch03_gqa_heads.png

三块并排，各画 8 个 Query 头。上排是 Query 头，下排是它们共用的 K/V 头，
连线表示“哪些 Q 头读同一份 K/V”。方框下方给出每个词元要缓存的 K/V 头数。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, MUTED,
                      NEW_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "ch03_gqa_heads.png")

NQ = 8
QW, QH = 0.82, 0.9
GAPQ = 0.16
PANEL_W = NQ * (QW + GAPQ) - GAPQ
QY, KY = 7.4, 3.6


def panel(ax, x0, title, n_kv, note):
    label(ax, x0 + PANEL_W / 2, 10.9, title, fontsize=FS_NAME, bold=True)
    label(ax, x0 + PANEL_W / 2, 9.9, note, fontsize=FS_SMALL, color=MUTED)

    qx = []
    for i in range(NQ):
        x = x0 + i * (QW + GAPQ)
        box(ax, x, QY, QW, QH, f"{i + 1}", kind="data", fontsize=FS_SMALL, rounded=False)
        qx.append(x + QW / 2)

    per = NQ // n_kv
    for g in range(n_kv):
        left = x0 + g * per * (QW + GAPQ)
        w = per * (QW + GAPQ) - GAPQ
        box(ax, left, KY, w, QH, "K/V", kind="new", fontsize=FS_SMALL, rounded=False)
        cx = left + w / 2
        for i in range(g * per, (g + 1) * per):
            arrow(ax, (qx[i], QY), (cx, KY + QH), color=NEW_EDGE, lw=1.1, rad=0.0)

    label(ax, x0 + PANEL_W / 2, 2.4, f"每层缓存 {n_kv} 组 K/V",
          fontsize=FS_TEXT, color=ACCENT)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11.8, 5.0))
    label(ax, 13.6, 12.4, "8 个 Query 头共用几组 K/V：同一套注意力，三种缓存代价",
          fontsize=FS_TITLE, bold=True)
    label(ax, 0.3, QY + QH / 2, "Query 头", fontsize=FS_SMALL, color=MUTED, ha="left")
    label(ax, 0.3, KY + QH / 2, "K/V 头", fontsize=FS_SMALL, color=MUTED, ha="left")
    panel(ax, 3.4, "MHA", 8, "每个 Q 头各一份")
    panel(ax, 12.0, "GQA（4 组）", 4, "每 2 个 Q 头共用一份")
    panel(ax, 20.6, "MQA", 1, "全部 Q 头共用一份")
    finish(fig, ax, OUTPUT, xlim=(0, 28.4), ylim=(1.8, 12.9))


if __name__ == "__main__":
    main()
