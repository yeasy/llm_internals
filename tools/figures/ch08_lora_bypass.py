"""生成 8.4 节的示意图：LoRA 旁路的形状、冻结/可训练分区与显存去向。

正文位置：08_alignment/8.4_peft.md
输出：08_alignment/_images/ch08_lora_bypass.png

上：一个线性层的数据流，按行向量写法标出每个矩阵的形状（Llama 3 8B 的
`W_Q`，d = 4096、r = 16）。灰色是冻结权重，橙色是可训练权重，蓝色是数据。
下：同一层里三份显存各归谁，说明冻结基座省掉的是梯度与优化器状态，激活照留。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MASK_EDGE, MUTED,
                      NEW_EDGE, WEIGHT_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("08_alignment", "ch08_lora_bypass.png")


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    label(ax, 8.6, 12.4, "一个线性层加上 LoRA 旁路（Llama 3 8B 的 W_Q，d = 4096，r = 16）",
          fontsize=FS_NAME, bold=True)

    # 输入
    x_c = box(ax, 0.3, 8.2, 2.5, 1.3, "输入 x\n[1, 4096]", kind="data")

    # 主干
    w_c = box(ax, 4.0, 9.7, 3.4, 1.3, "W（冻结）\n[4096, 4096]", kind="neutral")
    # 旁路
    a_c = box(ax, 3.6, 6.4, 2.3, 1.3, "A（可训练）\n[4096, 16]", kind="weight")
    v_c = box(ax, 6.4, 6.4, 1.6, 1.3, "v\n[1, 16]", kind="data")
    b_c = box(ax, 8.5, 6.4, 2.3, 1.3, "B（可训练）\n[16, 4096]", kind="weight")

    add_c = box(ax, 11.6, 8.2, 1.5, 1.3, "+", kind="new", fontsize=18)
    y_c = box(ax, 14.0, 8.2, 2.5, 1.3, "输出 y\n[1, 4096]", kind="data")

    arrow(ax, (2.8, 8.85), (4.0, w_c[1]), color=DATA_EDGE, rad=-0.18)
    arrow(ax, (2.8, 8.85), (3.6, a_c[1]), color=DATA_EDGE, rad=0.18)
    arrow(ax, (7.4, w_c[1]), (11.6, 9.0), color=MASK_EDGE, rad=-0.12)
    arrow(ax, (5.9, a_c[1]), (6.4, v_c[1]), color=WEIGHT_EDGE)
    arrow(ax, (8.0, v_c[1]), (8.5, b_c[1]), color=DATA_EDGE)
    arrow(ax, (10.8, b_c[1]), (11.6, 8.5), color=WEIGHT_EDGE, rad=0.12)
    arrow(ax, (13.1, add_c[1]), (14.0, y_c[1]), color=NEW_EDGE)

    label(ax, 5.7, 11.4, "主干：一次 [1, 4096] × [4096, 4096]，2 × 4096 × 4096 ≈ 3,355 万次运算",
          fontsize=FS_SMALL, color=MUTED)
    label(ax, 7.2, 5.6, "旁路：先降到 16 维再升回去，两次共 4 × 4096 × 16 ≈ 26 万次运算，占主干 0.78%",
          fontsize=FS_SMALL, color=WEIGHT_EDGE)
    label(ax, 12.35, 10.0, "$y = xW + \\frac{\\alpha}{r}\\,xAB$", fontsize=FS_TEXT, color=ACCENT)

    # 下半：显存分区
    label(ax, 8.6, 3.9, "这一层的显存各归谁", fontsize=FS_NAME, bold=True)
    items = (("W 的权重\n冻结，仍要读", "neutral", 2 * 4096 * 4096 / 1e6),
             ("W 的梯度与\n优化器状态", "plain", 0.0),
             ("A、B 的权重、梯度\n与优化器状态", "weight", 16 * 2 * 4096 * 16 / 1e6),
             ("前向激活\n三种微调一样多", "data", None))
    for i, (txt, kind, mb) in enumerate(items):
        c = box(ax, 0.2 + i * 4.35, 1.5, 3.95, 1.7, txt, kind=kind, fontsize=FS_SMALL - 0.5)
        if mb is None:
            note = "不随方法变"
        elif mb == 0.0:
            note = "LoRA 下完全不存"
        else:
            note = f"{mb:.2f} MB"
        label(ax, c[0], 1.0, note, fontsize=FS_SMALL,
              color=NEW_EDGE if mb == 0.0 else MUTED)

    finish(fig, ax, OUTPUT, xlim=(-0.2, 17.4), ylim=(0.4, 12.9))


if __name__ == "__main__":
    main()
