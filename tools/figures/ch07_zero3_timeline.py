"""生成 7.2 节的图：ZeRO-3 / FSDP 在一张卡上的一步训练，3 层模型，每层一个分片单元。

正文位置：07_distributed_training/7.2_zero.md
输出：07_distributed_training/_images/ch07_zero3_timeline.png

三条泳道：
- 计算流：F1 F2 F3（前向，各 1 个单位）后接 B3 B2 B1（反向，各 2 个单位），最后是只更新本卡分片的优化器步。
- 通信流：AG = AllGather 取回某层完整参数，RS = Reduce-Scatter 把某层梯度求和并只留本卡那一份。
  下一层的 AG 与当前层的计算同时进行（预取），上一层的 RS 与当前层的反向同时进行。
- 驻留：每层完整（未分片）参数在显存里停留的时间段。任一时刻最多两层，这是峰值显存里“分片之外”的那一项。

时长是示意值（F = 1，B = 2，AG = RS = 0.8）。第 3 层前向后紧接着反向，示意中不释放；
DeepSpeed 的 stage3_max_reuse_distance 就是控制“很快还要用的参数先不释放”的参数。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _diagram import (ACCENT, DATA_EDGE, DATA_FACE, FS_SMALL, FS_TEXT, FS_TITLE, MUTED, NEW_EDGE,
                      NEW_FACE, NEUTRAL_EDGE, NEUTRAL_FACE, WEIGHT_EDGE, WEIGHT_FACE, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("07_distributed_training", "ch07_zero3_timeline.png")

UW, RH = 1.55, 1.0
X0 = 3.3

COMPUTE = [("F1", 0.8, 1.8, "F"), ("F2", 1.8, 2.8, "F"), ("F3", 2.8, 3.8, "F"),
           ("B3", 3.8, 5.8, "B"), ("B2", 5.8, 7.8, "B"), ("B1", 7.8, 9.8, "B"),
           ("更新分片", 10.6, 12.0, "O")]
COMM = [("AG1", 0.0, 0.8, "AG"), ("AG2", 0.8, 1.6, "AG"), ("AG3", 1.8, 2.6, "AG"),
        ("AG2", 3.8, 4.6, "AG"), ("RS3", 5.8, 6.6, "RS"), ("AG1", 6.6, 7.4, "AG"),
        ("RS2", 7.8, 8.6, "RS"), ("RS1", 9.8, 10.6, "RS")]
RESIDENT = {3: [(1.8, 5.8)], 2: [(0.8, 2.8), (3.8, 7.8)], 1: [(0.0, 1.8), (6.6, 9.8)]}
END = 12.0

STYLE = {"F": (DATA_FACE, DATA_EDGE), "B": (DATA_FACE, DATA_EDGE), "O": (NEUTRAL_FACE, NEUTRAL_EDGE),
         "AG": (WEIGHT_FACE, WEIGHT_EDGE), "RS": (NEW_FACE, NEW_EDGE)}


def lane(ax, y, name, items):
    label(ax, X0 - 0.25, y + RH / 2, name, ha="right", fontsize=FS_TEXT)
    ax.add_patch(Rectangle((X0, y), END * UW, RH, facecolor="white", edgecolor="#c9c8c2",
                           linewidth=0.8, zorder=1))
    for text, s, e, kind in items:
        face, edge = STYLE[kind]
        ax.add_patch(Rectangle((X0 + s * UW, y + 0.08), (e - s) * UW, RH - 0.16, facecolor=face,
                               edgecolor=edge, linewidth=1.2, zorder=2))
        label(ax, X0 + (s + e) / 2 * UW, y + RH / 2, text, fontsize=FS_SMALL)


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    label(ax, 0.0, 9.2, "ZeRO-3 的一步：逐层取回参数，用完即放", ha="left", fontsize=FS_TITLE,
          bold=True)
    lane(ax, 7.2, "计算流", COMPUTE)
    lane(ax, 5.9, "通信流", COMM)
    label(ax, X0 + 1.9 * UW, 8.55, "前向", fontsize=FS_SMALL, color=MUTED)
    label(ax, X0 + 6.8 * UW, 8.55, "反向", fontsize=FS_SMALL, color=MUTED)
    ax.plot([X0 + 3.8 * UW] * 2, [2.0, 8.4], color="#c9c8c2", lw=1.0, ls="--", zorder=0)

    label(ax, 0.0, 4.95, "完整参数的驻留时段", ha="left", fontsize=FS_TEXT, bold=True)
    for k, layer in enumerate((1, 2, 3)):
        y = 3.9 - k * 0.85
        label(ax, X0 - 0.25, y + 0.3, f"第 {layer} 层", ha="right", fontsize=FS_TEXT)
        ax.plot([X0, X0 + END * UW], [y + 0.3, y + 0.3], color="#c9c8c2", lw=0.8, zorder=0)
        for s, e in RESIDENT[layer]:
            ax.add_patch(Rectangle((X0 + s * UW, y), (e - s) * UW, 0.6, facecolor=WEIGHT_FACE,
                                   edgecolor=WEIGHT_EDGE, linewidth=1.2, zorder=2))
    label(ax, X0, 1.35, "任一时刻最多两层的完整参数在显存里：正在算的一层，加上预取的下一层。",
          ha="left", fontsize=FS_SMALL, color=ACCENT)

    ly = 0.45
    for k, (kind, text) in enumerate((("F", "计算（F 前向 / B 反向）"), ("AG", "AllGather 取回完整参数"),
                                      ("RS", "Reduce-Scatter 归约梯度并分片"))):
        x = (0.0, 6.9, 13.6)[k]
        face, edge = STYLE[kind]
        ax.add_patch(Rectangle((x, ly - 0.25), 0.8, 0.5, facecolor=face, edgecolor=edge))
        label(ax, x + 1.0, ly, text, ha="left", fontsize=FS_SMALL)
    finish(fig, ax, OUTPUT, xlim=(-0.2, X0 + END * UW + 0.3), ylim=(-0.2, 9.8))


if __name__ == "__main__":
    main()
