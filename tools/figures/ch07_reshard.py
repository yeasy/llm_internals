"""生成 7.7 节的图：同一个逻辑张量，保存时切成 8 片，加载时重切成 6 片。

正文位置：07_distributed_training/7.7_checkpoint.md
输出：07_distributed_training/_images/ch07_reshard.png

中间一条是逻辑张量：24 行，行号 0 到 23。上面是保存时的 8 个分片（每片 3 行），
下面是加载时的 6 个分片（每片 4 行）。检查点的元数据只记两样：逻辑张量的全局形状，
和每个已存分片覆盖的行区间。加载方按新拓扑算出自己要的行区间，与已存分片的区间求交，
只读有交集的那几段。图中标出新分片 2（第 8 到 11 行）：它与旧分片 2（第 6 到 8 行）交于第 8 行，
与旧分片 3（第 9 到 11 行）交于第 9 到 11 行。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, FS_TITLE, MUTED, NEW_EDGE,
                      NEW_FACE, NEUTRAL_EDGE, NEUTRAL_FACE, WEIGHT_EDGE, WEIGHT_FACE, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("07_distributed_training", "ch07_reshard.png")

ROWS, OLD, NEW = 24, 8, 6
CW, RH = 0.92, 1.1
X0 = 4.6
FOCUS = 2


def shards(ax, y, n, title, face, edge, focus=None, overlap=None, above=True):
    size = ROWS // n
    label(ax, X0 - 0.3, y + RH / 2, title, ha="right", fontsize=FS_TEXT)
    for k in range(n):
        x = X0 + k * size * CW
        hot = focus == k or (overlap and k in overlap)
        ax.add_patch(Rectangle((x + 0.06, y), size * CW - 0.12, RH,
                               facecolor=NEW_FACE if hot else face,
                               edgecolor=NEW_EDGE if hot else edge,
                               linewidth=2.0 if hot else 1.2, zorder=2))
        label(ax, x + size * CW / 2, y + RH / 2, f"卡 {k}", fontsize=FS_SMALL)
        label(ax, x + size * CW / 2, y + RH + 0.4 if above else y - 0.4,
              f"行 {k * size}~{(k + 1) * size - 1}", fontsize=FS_SMALL, color=MUTED)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    label(ax, 0.0, 9.3, "卡数变了：按逻辑行号重新切，而不是按文件搬", ha="left", fontsize=FS_TITLE,
          bold=True)
    y_old, y_mid, y_new = 6.6, 4.3, 2.0
    old_size, new_size = ROWS // OLD, ROWS // NEW
    lo, hi = FOCUS * new_size, (FOCUS + 1) * new_size
    overlap = sorted({r // old_size for r in range(lo, hi)})
    shards(ax, y_old, OLD, "保存时 8 片", WEIGHT_FACE, WEIGHT_EDGE, overlap=overlap)
    shards(ax, y_new, NEW, "加载时 6 片", WEIGHT_FACE, WEIGHT_EDGE, focus=FOCUS, above=False)

    label(ax, X0 - 0.3, y_mid + RH / 2, "逻辑张量 [24, d]", ha="right", fontsize=FS_TEXT)
    for r in range(ROWS):
        hot = lo <= r < hi
        ax.add_patch(Rectangle((X0 + r * CW, y_mid), CW, RH,
                               facecolor=NEW_FACE if hot else DATA_FACE,
                               edgecolor=NEW_EDGE if hot else DATA_EDGE, linewidth=0.9, zorder=2))
        label(ax, X0 + (r + 0.5) * CW, y_mid + RH / 2, str(r), fontsize=FS_SMALL)

    for k in overlap:
        a, b = max(lo, k * old_size), min(hi, (k + 1) * old_size)
        ax.add_patch(Polygon([(X0 + a * CW, y_old), (X0 + b * CW, y_old),
                              (X0 + b * CW, y_mid + RH), (X0 + a * CW, y_mid + RH)],
                             closed=True, facecolor=NEW_FACE, edgecolor="none", alpha=0.55, zorder=1))
    ax.add_patch(Polygon([(X0 + lo * CW, y_mid), (X0 + hi * CW, y_mid),
                          (X0 + hi * CW, y_new + RH), (X0 + lo * CW, y_new + RH)],
                         closed=True, facecolor=NEW_FACE, edgecolor="none", alpha=0.55, zorder=1))

    label(ax, X0, 0.55,
          f"新的卡 {FOCUS} 要第 {lo} 到 {hi - 1} 行：与旧分片 {overlap[0]} 交于第 {lo} 行，"
          f"与旧分片 {overlap[1]} 交于第 {overlap[1] * old_size} 到 {hi - 1} 行，只读这两段。",
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    label(ax, X0, -0.05, "元数据只需记录：逻辑张量的全局形状，以及每个已存分片覆盖的行区间。",
          ha="left", fontsize=FS_SMALL, color=MUTED)
    finish(fig, ax, OUTPUT, xlim=(-0.2, X0 + ROWS * CW + 0.3), ylim=(-0.5, 10.0))


if __name__ == "__main__":
    main()
