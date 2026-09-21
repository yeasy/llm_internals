"""生成 14.3 节的图：Mamba 块的形状流，以及选择性为什么逼出并行扫描。

正文位置：14_future_trends/14.3_ssm_hybrid.md
输出：14_future_trends/_images/ch14a_mamba_block.png

上半幅按 Mamba 论文 3.4 节的块结构画形状，取值 d_model = 4096、E = 2（论文固定）、
N = d_state = 16（论文 4.5 节的基准取值），故 d_inner = 8192，状态 [8192, 16]。
下半幅画 8 个时间步的关联扫描：结合律算子 (a1,b1)∘(a2,b2) = (a1a2, a2b1+b2)
把串行深度 8 降到树形深度 3（上行）加 3（下行）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE, MUTED,
                      NEW_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14a_mamba_block.png")

BW, BH = 4.3, 2.1
GAP = 1.5


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 8.6))

    # ---------- 上半幅：块的形状流 ----------
    label(ax, 0.0, 15.6, "Mamba 块：形状怎样一步步变化（d_model = 4096，E = 2，N = 16）",
          ha="left", fontsize=FS_TITLE, bold=True)

    top = 13.7
    main_chain = [
        ("输入 x", "[T, 4096]", "data"),
        ("in_proj", "[4096, 16384]", "weight"),
        ("拆成两支", "各 [T, 8192]", "data"),
    ]
    x = 0.0
    for name, shape, kind in main_chain:
        box(ax, x, top - BH, BW, BH, f"{name}\n{shape}", kind, fontsize=FS_SMALL)
        x += BW
        if name != "拆成两支":
            arrow(ax, (x + 0.15, top - BH / 2), (x + GAP - 0.15, top - BH / 2), color=MUTED)
            x += GAP

    # 主支
    y2 = top - BH - 3.3
    label(ax, 0.0, y2 + BH + 0.6, "主支（走 SSM）", ha="left", fontsize=FS_TEXT,
          bold=True, color=MUTED)
    chain = [
        ("因果 conv1d\nd_conv = 4", "[T, 8192]", "weight"),
        ("SiLU", "[T, 8192]", "neutral"),
        ("选择性 SSM\n状态 [8192, 16]", "[T, 8192]", "new"),
    ]
    x = 0.0
    for name, shape, kind in chain:
        box(ax, x, y2, BW, BH, f"{name}\n{shape}", kind, fontsize=FS_SMALL)
        x += BW
        if name.startswith("SiLU"):
            arrow(ax, (x + 0.15, y2 + BH / 2), (x + GAP - 0.15, y2 + BH / 2), color=MUTED)
            x += GAP
        elif name.startswith("因果"):
            arrow(ax, (x + 0.15, y2 + BH / 2), (x + GAP - 0.15, y2 + BH / 2), color=MUTED)
            x += GAP

    # 门控支与输出
    y3 = y2 - 3.3
    label(ax, 0.0, y3 + BH + 0.6, "门控支（逐格相乘）与输出", ha="left", fontsize=FS_TEXT,
          bold=True, color=MUTED)
    box(ax, 0.0, y3, BW, BH, "SiLU(z)\n[T, 8192]", "data", fontsize=FS_SMALL)
    label(ax, BW + GAP / 2, y3 + BH / 2, "⊙", fontsize=FS_TITLE, color=MUTED)
    box(ax, BW + GAP, y3, BW, BH, "SSM 输出\n[T, 8192]", "new", fontsize=FS_SMALL)
    arrow(ax, (2 * BW + GAP + 0.15, y3 + BH / 2), (2 * BW + 2 * GAP - 0.15, y3 + BH / 2),
          color=MUTED)
    box(ax, 2 * (BW + GAP), y3, BW, BH, "out_proj\n[8192, 4096]", "weight",
        fontsize=FS_SMALL)
    label(ax, 3 * BW + 2 * GAP + 0.5, y3 + BH / 2, "→ [T, 4096]", ha="left",
          fontsize=FS_NAME)

    label(ax, 0.0, y3 - 1.25,
          "解码时只保留两块状态：SSM 的 [8192, 16] 与卷积的 3 列 [3, 8192]，"
          "都与已生成多少词元无关。",
          ha="left", fontsize=FS_TEXT, color=ACCENT)

    # ---------- 下半幅：并行扫描 ----------
    ysc = y3 - 2.9
    label(ax, 0.0, ysc, "选择性破坏了卷积形式，只能靠关联扫描把串行变并行",
          ha="left", fontsize=FS_TITLE, bold=True)

    cw, ch2 = 1.65, 0.85
    lv0 = ysc - 1.5 - ch2
    n = 8
    pitch = 2.2
    for i in range(n):
        box(ax, i * pitch, lv0, cw, ch2, f"t{i + 1}", "data", fontsize=FS_SMALL,
            rounded=False)
    # 三层归并
    y = lv0
    step = 1
    lvl = 1
    while step < n:
        y -= 1.6
        for i in range(0, n, step * 2):
            lx = i * pitch
            rx = (i + step) * pitch
            box(ax, lx, y, cw + (rx - lx), ch2, f"t{i + 1}…t{min(i + 2 * step, n)}",
                "new", fontsize=FS_SMALL, rounded=False)
            arrow(ax, (lx + cw / 2, y + 1.6 - 0.05), (lx + cw / 2, y + ch2 + 0.05),
                  color=NEW_EDGE, lw=1.1)
            arrow(ax, (rx + cw / 2, y + 1.6 - 0.05), (rx + cw / 2, y + ch2 + 0.05),
                  color=NEW_EDGE, lw=1.1)
        label(ax, n * pitch + 0.3, y + ch2 / 2, f"第 {lvl} 层归并", ha="left",
              fontsize=FS_SMALL, color=MUTED)
        step *= 2
        lvl += 1

    label(ax, 0.0, y - 1.1,
          "每个方框做一次 $(a_1,b_1) \\circ (a_2,b_2) = (a_1 a_2,\\ a_2 b_1 + b_2)$："
          "8 步的串行链被压成 3 层。\n"
          "n = 100 万时串行深度 100 万，树形深度只有 20 层（上行）加 20 层（下行）。",
          ha="left", fontsize=FS_TEXT, color=ACCENT)

    finish(fig, ax, OUTPUT, xlim=(-0.4, 22.4), ylim=(y - 2.1, 16.2))


if __name__ == "__main__":
    main()
