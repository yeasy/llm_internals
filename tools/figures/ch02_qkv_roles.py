"""生成 2.1 节插图：三路投影各自流向哪里。

正文位置：02_attention/2.1_qkv_intuition.md
输出：02_attention/_images/ch02_qkv_roles.png

图只画两个位置：正在提问的位置 i，和被读取的位置 j。要读出的一条是
K 只进打分、V 只进输出，两条路径在 Softmax 之后才汇合。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_SMALL, FS_TEXT, INK, MUTED, arrow, box, finish,
                      label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("02_attention", "ch02_qkv_roles.png")

BW, BH = 2.9, 0.95


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.4, 4.4))

    xi = box(ax, 0.0, 5.3, BW, BH, "x_i\n位置 i", kind="data", fontsize=FS_SMALL)
    xj = box(ax, 0.0, 1.6, BW, BH, "x_j\n位置 j", kind="data", fontsize=FS_SMALL)

    wq = box(ax, 3.7, 5.3, BW, BH, "W_Q", kind="weight")
    wk = box(ax, 3.7, 3.45, BW, BH, "W_K", kind="weight")
    wv = box(ax, 3.7, 0.4, BW, BH, "W_V", kind="weight")

    q = box(ax, 7.4, 5.3, 1.7, BH, "q_i", kind="data")
    k = box(ax, 7.4, 3.45, 1.7, BH, "k_j", kind="data")
    v = box(ax, 7.4, 0.4, 1.7, BH, "v_j", kind="data")

    score = box(ax, 10.2, 4.35, 3.0, BH, "点积 ÷ √d_h", kind="plain", fontsize=FS_SMALL)
    soft = box(ax, 14.2, 4.35, 2.5, BH, "Softmax\n沿 j 归一", kind="plain", fontsize=FS_SMALL)
    weight = box(ax, 17.6, 4.35, 1.7, BH, "a_ij", kind="data")
    out = box(ax, 17.6, 0.4, 2.7, BH, "输出 o_i", kind="data")

    arrow(ax, (BW, 5.3 + BH / 2), (3.7, 5.3 + BH / 2))
    arrow(ax, (BW, 2.0 + BH / 2), (3.7, 3.45 + BH / 2), rad=0.12)
    arrow(ax, (BW, 1.6 + BH / 2), (3.7, 0.4 + BH / 2), rad=-0.12)
    for y in (5.3, 3.45, 0.4):
        arrow(ax, (3.7 + BW, y + BH / 2), (7.4, y + BH / 2))
    arrow(ax, (9.1, 5.3 + BH / 2), (10.2, 4.9 + BH / 2), rad=-0.1)
    arrow(ax, (9.1, 3.45 + BH / 2), (10.2, 4.6), rad=0.1)
    arrow(ax, (13.2, 4.35 + BH / 2), (14.2, 4.35 + BH / 2))
    arrow(ax, (16.7, 4.35 + BH / 2), (17.6, 4.35 + BH / 2))
    arrow(ax, (18.45, 4.35), (18.45, 0.4 + BH), color=ACCENT)
    arrow(ax, (9.1, 0.4 + BH / 2), (17.6, 0.4 + BH / 2))

    label(ax, 18.6, 2.5, "按权重加起来", fontsize=FS_SMALL, color=ACCENT, ha="left")
    label(ax, 13.0, 2.75, "k_j 到此为止：它不进入输出", fontsize=FS_SMALL, color=MUTED)
    label(ax, 13.3, 1.32, "v_j 一路直通：它不参与打分", fontsize=FS_SMALL, color=MUTED)
    label(ax, 0.0, 7.0,
          "同一个向量乘三个不同的矩阵：Q 决定“找什么”，K 决定“怎样被找到”，V 决定“被读走什么”",
          fontsize=FS_TEXT, bold=True, ha="left")
    label(ax, 0.0, 6.55, "改 W_K 只改权重的分布，改 W_V 只改读出的内容；x 是 [1, d_model]，三个投影矩阵都是 [d_model, d_h]，q、k、v 都是 [1, d_h]",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    _ = (xi, xj, wq, wk, wv, q, k, v, score, soft, weight, out, DATA_EDGE, INK)
    finish(fig, ax, OUTPUT, xlim=(-0.3, 20.6), ylim=(0.0, 7.4))


if __name__ == "__main__":
    main()
