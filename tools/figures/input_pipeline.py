"""生成图 3-9：一条消息怎样变成初始表示 X⁽⁰⁾。

正文位置：03_components/3.8_gpt_inference_flow.md
输出：03_components/_images/input_pipeline.png

从上到下四步：聊天模板把消息排成一串文字，分词器切成 6 个词元并给出 ID，
按 ID 从词嵌入矩阵 E 里各取一行，再加上该位置的位置向量，得到 [6, 4] 的 X⁽⁰⁾。
图中的数值与正文表 3-3 完全一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, arrow, box,
                      finish, grid, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("03_components", "input_pipeline.png")

TOKENS = ["〈user〉", "2", "+", "3", "=", "〈assistant〉"]
IDS = [0, 11, 12, 13, 14, 1]
EMB = [[0.0, 0.0, 0, 0], [0.9, -0.1, 1, 0], [-0.2, 0.8, 0, 0],
       [0.7, 0.7, 1, 0], [0.6, -0.4, 0, 1], [0.5, -0.5, 0, 1]]
POS = [[0.1 * i, 0.1 * i, 0, 0] for i in range(6)]
X0 = [[round(e + p, 1) for e, p in zip(er, pr)] for er, pr in zip(EMB, POS)]

COL_X = [2.6 + i * 4.0 for i in range(6)]  # 六个词元各占一列
CHIP_W = 3.7


def fmt(v):
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(14.5, 10.8))
    left, right = COL_X[0], COL_X[-1] + CHIP_W

    # ① 消息
    y = 21.0
    label(ax, left - 0.5, y + 0.6, "① 用户消息", ha="right", bold=True, fontsize=FS_NAME)
    box(ax, left, y, right - left, 1.2, "role: user    content: 2 + 3 等于几？", "plain")
    arrow(ax, ((left + right) / 2, y - 0.05), ((left + right) / 2, y - 1.35))
    label(ax, (left + right) / 2 + 0.3, y - 0.7, "聊天模板：加上角色标记，排成一串", ha="left",
          fontsize=FS_SMALL, color=MUTED)

    # ② 序列化后的文字
    y = 18.4
    label(ax, left - 0.5, y + 0.6, "② 模板输出", ha="right", bold=True, fontsize=FS_NAME)
    box(ax, left, y, right - left, 1.2, "〈user〉 2 + 3 = 〈assistant〉", "plain")
    arrow(ax, ((left + right) / 2, y - 0.05), ((left + right) / 2, y - 1.35))
    label(ax, (left + right) / 2 + 0.3, y - 0.7, "分词器：切成词元，每个词元一个整数 ID",
          ha="left", fontsize=FS_SMALL, color=MUTED)

    # ③ 词元与 ID
    y = 14.9
    label(ax, left - 0.5, y + 0.6, "③ 6 个词元", ha="right", bold=True, fontsize=FS_NAME)
    label(ax, left - 0.5, y - 0.75, "词元 ID", ha="right", fontsize=FS_TEXT, color=MUTED)
    label(ax, left - 0.5, y + 1.75, "位置", ha="right", fontsize=FS_TEXT, color=MUTED)
    for i, (tok, tid) in enumerate(zip(TOKENS, IDS)):
        cx = COL_X[i] + CHIP_W / 2
        label(ax, cx, y + 1.75, str(i + 1), fontsize=FS_TEXT, color=MUTED)
        box(ax, COL_X[i], y, CHIP_W, 1.2, tok, "data", fontsize=FS_TEXT)
        label(ax, cx, y - 0.75, str(tid), fontsize=FS_NAME, bold=True)
        arrow(ax, (cx, y - 1.3), (cx, y - 2.5))

    # ④ 查表 + 位置向量
    y = 11.2
    label(ax, left - 0.5, y + 0.5, "④ 按 ID 查表", ha="right", bold=True, fontsize=FS_NAME)
    label(ax, left - 0.5, y - 0.45, "词嵌入 E 的一行", ha="right", fontsize=FS_SMALL, color=MUTED)
    label(ax, left - 0.5, y - 2.2, "加上位置向量 p", ha="right", fontsize=FS_SMALL, color=MUTED)
    for i in range(6):
        cx = COL_X[i] + CHIP_W / 2
        grid(ax, COL_X[i], y + 0.6, [EMB[i]], kind="weight", cell_w=CHIP_W / 4, cell_h=1.0,
             fontsize=FS_SMALL - 1.5, fmt=fmt)
        label(ax, cx, y - 1.05, "+", fontsize=FS_NAME)
        grid(ax, COL_X[i], y - 1.7, [POS[i]], kind="weight", cell_w=CHIP_W / 4, cell_h=1.0,
             fontsize=FS_SMALL, fmt=fmt)
        arrow(ax, (cx, y - 2.85), (cx, y - 4.05))

    # ⑤ X0：把六个结果按行堆叠
    y = 6.3
    label(ax, left - 0.5, y, "⑤ 逐格相加", ha="right", bold=True, fontsize=FS_NAME)
    for i in range(6):
        grid(ax, COL_X[i], y + 0.5, [X0[i]], kind="data", cell_w=CHIP_W / 4, cell_h=1.0,
             fontsize=FS_SMALL, fmt=fmt)
    # 汇聚成矩阵
    mx, mtop = (left + right) / 2 - 2.8, 3.4
    for i in range(6):
        arrow(ax, (COL_X[i] + CHIP_W / 2, y - 0.6), ((left + right) / 2, mtop + 0.15),
              color=MUTED, lw=1.0)
    grid(ax, mx, mtop, X0, kind="data", cell_w=1.4, cell_h=0.85, fontsize=FS_TEXT, fmt=fmt)
    label(ax, mx - 0.4, mtop - 2.55, "6 行：\n每个位置一行", ha="right", fontsize=FS_SMALL,
          color=MUTED)
    label(ax, mx + 2.8, mtop - 5.55, "4 列：d_model = 4", fontsize=FS_SMALL, color=MUTED)
    label(ax, mx + 6.1, mtop - 1.6, "初始表示 $X^{(0)}$，形状 [6, 4]", ha="left", bold=True,
          fontsize=FS_NAME)
    label(ax, mx + 6.1, mtop - 2.9, "按位置逐行堆叠，\n送入第 1 层", ha="left",
          fontsize=FS_TEXT, color=MUTED)
    label(ax, mx + 6.1, mtop - 4.5, "真实模型每行有几百到上万个数", ha="left",
          fontsize=FS_SMALL, color=ACCENT)

    finish(fig, ax, OUTPUT, xlim=(-3.2, right + 0.6), ylim=(-2.8, 22.8))


if __name__ == "__main__":
    main()
