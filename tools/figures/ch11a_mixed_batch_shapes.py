"""生成 11.2 节“一轮迭代里的混合批”示意图：摊平成一维，线性层一起算，注意力各算各的。

正文位置：11_serving/11.2_continuous_batching.md
输出：11_serving/_images/ch11a_mixed_batch_shapes.png

三个请求处在不同阶段：R1 做 Prefill（5 个词元），R2、R3 各做一步 Decode（缓存里已有 7 行、3 行）。
形状按 Llama 3 8B：d_model = 4096，每头 d_h = 128。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, MUTED, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11a_mixed_batch_shapes.png")

CELL = 0.86


def strip(ax, x, y, cached, new):
    for i in range(cached + new):
        box(ax, x + i * CELL, y, CELL, 0.95, "", "new" if i >= cached else "data",
            rounded=False, lw=1.0)
    return x + (cached + new) * CELL


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.2, 9.4))

    # ---- ①：本轮选中的三个请求 ----
    label(ax, 0.0, 25.2, "① 本轮选中的三个请求", fontsize=FS_NAME, bold=True, ha="left")
    rows = [("R1  Prefill：Prompt 的 5 个词元", 0, 5, 21.6),
            ("R2  Decode：缓存里 7 行，新词元 1 个", 7, 1, 18.6),
            ("R3  Decode：缓存里 3 行，新词元 1 个", 3, 1, 15.6)]
    for text, cached, new, y in rows:
        label(ax, 0.0, y + 1.55, text, fontsize=FS_SMALL, ha="left")
        strip(ax, 0.0, y, cached, new)

    # ---- ②：摊平 ----
    xb, top, rh = 18.0, 23.6, 1.05
    label(ax, xb - 1.0, 25.2, "② 把本轮要算的词元摊平成一个矩阵", fontsize=FS_NAME, bold=True,
          ha="left")
    names = ["R1·1", "R1·2", "R1·3", "R1·4", "R1·5", "R2·8", "R3·4"]
    for i, name in enumerate(names):
        box(ax, xb, top - (i + 1) * rh, 6.0, rh, name, "new", rounded=False, lw=1.0,
            fontsize=FS_SMALL)
    arrow(ax, (14.4, 19.9), (xb - 0.3, 19.9))
    label(ax, xb + 7.0, 21.4, "X[7, 4096]", fontsize=FS_TEXT, color=ACCENT, ha="left")
    label(ax, xb + 7.0, 19.4, "没有“批”这一维\n行数 = 本轮词元数 = 5 + 1 + 1\n"
          "“R2·8”指 R2 的第 8 个位置", fontsize=FS_SMALL, color=MUTED, ha="left")

    # ② → ③
    ybot = top - 7 * rh
    arrow(ax, (xb + 3.0, ybot), (xb + 3.0, 14.2), style="-")
    arrow(ax, (xb + 3.0, 14.2), (13.4, 14.2), style="-")
    arrow(ax, (13.4, 14.2), (13.4, 11.3))

    # ---- ③：线性层 ----
    label(ax, 0.0, 12.9, "③ 带权重的层：7 行一次算完", fontsize=FS_NAME, bold=True, ha="left")
    box(ax, 0.0, 6.3, 15.0, 5.0, "Q/K/V 投影、$W_O$、MLP、归一化\n\n"
        "X[7, 4096] × W[4096, · ]\n每行独立计算，不必知道哪行属于谁", "weight",
        fontsize=FS_SMALL)

    # ---- ④：注意力 ----
    xd = 19.0
    label(ax, xd, 12.9, "④ 注意力：每个请求只对着自己的 K、V", fontsize=FS_NAME, bold=True,
          ha="left")
    att = [("R1：Q[5, 128] 对 K[5, 128]，带因果掩码", 9.5),
           ("R2：q[1, 128] 对 K[8, 128]，7 行来自缓存", 6.9),
           ("R3：q[1, 128] 对 K[4, 128]，3 行来自缓存", 4.3)]
    for text, y in att:
        box(ax, xd, y, 15.4, 2.0, text, "data", fontsize=FS_SMALL)
    arrow(ax, (15.3, 8.8), (xd - 0.3, 8.8))
    label(ax, xd + 16.0, 7.9, "K 的行数各不相同\n拼不成 [B, T, T]\n由注意力内核\n按请求分段计算",
          fontsize=FS_SMALL, color=MUTED, ha="left")

    # ④ → ③（下一层）
    arrow(ax, (xd + 7.7, 4.3), (xd + 7.7, 2.9), style="-", color=ACCENT, lw=1.6)
    arrow(ax, (xd + 7.7, 2.9), (7.5, 2.9), style="-", color=ACCENT, lw=1.6)
    arrow(ax, (7.5, 2.9), (7.5, 6.3), color=ACCENT, lw=1.6)
    label(ax, 17.2, 2.25, "各请求的结果拼回 [7, 4096]，进入下一层", fontsize=FS_SMALL, color=ACCENT)

    # ---- 出口 ----
    box(ax, 0.0, -1.9, 23.6, 2.7, "最后一层之后，只取每个请求的最后一行：\n[3, 4096] × "
        "$W_{vocab}$ → logits[3, $n_{vocab}$] → 采样出 3 个新词元", "plain", fontsize=FS_SMALL)

    # 图例
    for i, (kind, text) in enumerate((("data", "缓存里已有的行"), ("new", "本轮要算的词元"))):
        y = 0.0 - i * 1.15
        box(ax, 26.4, y, 0.9, 0.7, "", kind, rounded=False, lw=1.2)
        label(ax, 27.7, y + 0.35, text, fontsize=FS_SMALL, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.5, 42.5), ylim=(-2.4, 26.3))


if __name__ == "__main__":
    main()
