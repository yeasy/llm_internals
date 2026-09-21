"""生成 7.3 节的图：一层 Transformer 里张量并行的通信位置，以及叠加序列并行之后的变化。

正文位置：07_distributed_training/7.3_model_tensor_parallel.md
输出：07_distributed_training/_images/ch07_tp_sp_layer.png

上半幅：只用张量并行。注意力与 MLP 两个子层各被一对算子夹住：
  f 前向不通信、反向 AllReduce；g 前向 AllReduce、反向不通信。
  子层内部的激活按 t 切分，LayerNorm、dropout 与残差上的激活仍是每卡一份完整的 [s, b, h]。
下半幅：再加序列并行。g 的位置换成 Reduce-Scatter（反向 All-Gather），f 的位置换成 All-Gather（反向 Reduce-Scatter），
  圆圈内直接写前向的原语 AG / RS，
  LayerNorm 与 dropout 区域的激活沿序列维切成 [s/t, b, h]。
  一次 AllReduce = 一次 Reduce-Scatter + 一次 All-Gather，所以通信字节数不变。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_SMALL, FS_TEXT, FS_TITLE, INK, MUTED, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("07_distributed_training", "ch07_tp_sp_layer.png")

BW, BH = 3.2, 2.5
OPW = 1.3
GAP = 0.4


def row(ax, top, title, sp: bool):
    label(ax, 0.0, top + 2.2, title, ha="left", fontsize=FS_TITLE, bold=True)
    y = top - BH
    x = 0.0
    seq = [("norm", "LayerNorm"), ("f", ""), ("tp", "注意力\nQKV 列切\n$W_O$ 行切"), ("g", ""),
           ("norm", "dropout\n+ 残差"), ("norm", "LayerNorm"), ("f", ""),
           ("tp", "MLP\n$W_1$ 列切\n$W_2$ 行切"), ("g", ""), ("norm", "dropout\n+ 残差")]
    centers = []
    for kind, text in seq:
        if kind in ("f", "g"):
            if sp:
                # Megatron 序列并行论文把这两个算子记作 g 与 g-bar，与上半幅的 g 重名；
                # 图中直接写前向的通信原语，避免混淆。
                txt = "AG" if kind == "f" else "RS"
            else:
                txt = "f" if kind == "f" else "g"
            ax.add_patch(plt.Circle((x + OPW / 2, y + BH / 2), OPW / 2, facecolor="white",
                                    edgecolor=ACCENT, linewidth=2.2, zorder=3))
            label(ax, x + OPW / 2, y + BH / 2, txt, fontsize=FS_SMALL if sp else FS_TEXT,
                  color=ACCENT, bold=True)
            centers.append((kind, x + OPW / 2))
            w = OPW
        else:
            box(ax, x, y, BW, BH, text, "weight" if kind == "tp" else "data", fontsize=FS_SMALL)
            shape = "[s, b, h/t]" if kind == "tp" else ("[s/t, b, h]" if sp else "[s, b, h]")
            label(ax, x + BW / 2, y - 0.45, shape, fontsize=FS_SMALL,
                  color=ACCENT if (sp and kind == "norm") else MUTED)
            w = BW
        if x > 0:
            arrow(ax, (x - GAP, y + BH / 2), (x, y + BH / 2), lw=1.3)
        x += w + GAP
    for kind, cx in centers:
        if sp:
            fw, bw_ = ("AG", "RS") if kind == "f" else ("RS", "AG")
        else:
            fw, bw_ = ("不通信", "AllReduce") if kind == "f" else ("AllReduce", "不通信")
        label(ax, cx, y + BH + 1.0, f"前向 {fw}", fontsize=FS_SMALL, color=INK)
        label(ax, cx, y + BH + 0.4, f"反向 {bw_}", fontsize=FS_SMALL, color=MUTED)
    return x - GAP


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    right = row(ax, 13.0, "只用张量并行：每层前向 2 次、反向 2 次 AllReduce", sp=False)
    row(ax, 6.4, "再加序列并行：AllReduce 拆成 Reduce-Scatter（RS）与 All-Gather（AG）", sp=True)
    label(ax, 0.0, 1.75,
          "橙色块内的权重与激活按 t 切分；蓝色块不含被切的权重。方块下方是该区域每卡保存的激活形状。",
          ha="left", fontsize=FS_SMALL)
    label(ax, 0.0, 1.0,
          "序列并行不增加通信字节数，只把蓝色区域的激活从每卡一整份降到 1/t。",
          ha="left", fontsize=FS_SMALL, color=ACCENT)
    finish(fig, ax, OUTPUT, xlim=(-0.2, right + 0.2), ylim=(0.5, 15.9))


if __name__ == "__main__":
    main()
