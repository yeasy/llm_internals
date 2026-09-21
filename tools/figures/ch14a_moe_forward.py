"""生成 14.2 节的图：一次 MoE 前向的六步，以及每一步的张量形状。

正文位置：14_future_trends/14.2_moe.md
输出：14_future_trends/_images/ch14a_moe_forward.png

示意设定：批内 6 个词元、4 个路由专家、top-2，外加 1 个共享专家。每个词元选中的
两个专家是手工设定的，用来说明置换：dispatch 把同一专家的词元聚到一起，才能用一次
矩阵乘法算完；combine 再按原位置加权写回。形状一列按 DeepSeek-V3 的口径写：
d_model = 7168，专家中间层 2048（config.json 的 hidden_size 与 moe_intermediate_size）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, FS_TITLE,
                      MUTED, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("14_future_trends", "ch14a_moe_forward.png")

T = 6                                    # 词元数
CHOICE = [(0, 2), (2, 3), (0, 1), (2, 0), (1, 3), (2, 1)]   # 每个词元选中的两个专家
NE = 4
CW, CH = 1.7, 1.05
XMAX = 26.4


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(12.6, 8.4))

    label(ax, 0.0, 19.6, "一次 MoE 前向：六步，以及每一步换了什么形状",
          ha="left", fontsize=FS_TITLE, bold=True)

    # ① 输入
    y1 = 17.0
    label(ax, 0.0, y1 + CH + 0.45, "① 输入", ha="left", fontsize=FS_TEXT, bold=True)
    for t in range(T):
        box(ax, t * CW, y1, CW, CH, f"x{t + 1}", "data", fontsize=FS_NAME, rounded=False)
    label(ax, T * CW + 0.5, y1 + CH / 2, "X[6, 7168]", ha="left", fontsize=FS_NAME)

    # ② 路由打分
    y2 = y1 - 2.6
    label(ax, 0.0, y2 + CH + 0.45, "② 路由打分：X 乘路由矩阵，再逐格取 sigmoid",
          ha="left", fontsize=FS_TEXT, bold=True)
    box(ax, 0.0, y2, 5.0, CH, "W_g[7168, 256]", "weight", fontsize=FS_NAME)
    label(ax, 5.4, y2 + CH / 2, "→ 亲和度 s[6, 256]", ha="left", fontsize=FS_NAME)

    # ③ top-k
    y3 = y2 - 2.6
    label(ax, 0.0, y3 + CH + 0.45,
          "③ 取 top-8（本图缩到 top-2），再只在被选中的那几个之间归一化",
          ha="left", fontsize=FS_TEXT, bold=True)
    for t in range(T):
        a, b = CHOICE[t]
        box(ax, t * CW, y3, CW, CH, f"E{a},E{b}", "new", fontsize=FS_SMALL, rounded=False)
    label(ax, T * CW + 0.5, y3 + CH / 2, "选择表[6, 2]：这一步不可导", ha="left",
          fontsize=FS_NAME, color=ACCENT)

    # ④ dispatch
    y4 = y3 - 3.4
    label(ax, 0.0, y4 + CH + 1.15,
          "④ dispatch：按专家号重排，把同一专家的词元排到连续的行", ha="left",
          fontsize=FS_TEXT, bold=True)
    x = 0.0
    centers = []
    for e in range(NE):
        rows = [t for t in range(T) if e in CHOICE[t]]
        label(ax, x + len(rows) * CW / 2, y4 + CH + 0.4, f"专家 {e}", fontsize=FS_SMALL,
              color=MUTED)
        for j, t in enumerate(rows):
            box(ax, x + j * CW, y4, CW, CH, f"x{t + 1}", "data", fontsize=FS_NAME,
                rounded=False)
        centers.append((x + len(rows) * CW / 2, len(rows)))
        x += len(rows) * CW + 0.8
    label(ax, x + 0.4, y4 + CH / 2, "共 12 行 = 6 词元 × 2", ha="left", fontsize=FS_NAME)

    # ⑤ 分组 GEMM
    y5 = y4 - 3.4
    label(ax, 0.0, y5 + CH + 1.85,
          "⑤ 分组 GEMM：每组一次矩阵乘法，各组行数不同、权重不同", ha="left",
          fontsize=FS_TEXT, bold=True)
    for e, (cx, nrow) in enumerate(centers):
        box(ax, cx - 2.0, y5, 4.0, CH, f"W_e{e}[7168, 2048]", "weight", fontsize=FS_SMALL)
        arrow(ax, (cx, y5 + CH + 1.15), (cx, y5 + CH + 0.15), color=DATA_EDGE)
        label(ax, cx, y5 - 0.5, f"{nrow} 行", fontsize=FS_SMALL, color=MUTED)

    # ⑥ combine
    y6 = y5 - 3.1
    label(ax, 0.0, y6 + CH + 0.5, "⑥ combine：按门控权重加权，写回原位置，再加上共享专家",
          ha="left", fontsize=FS_TEXT, bold=True)
    for t in range(T):
        box(ax, t * CW, y6, CW, CH, f"y{t + 1}", "data", fontsize=FS_NAME, rounded=False)
    label(ax, T * CW + 0.5, y6 + CH / 2, "Y[6, 7168]，与 X 同形状", ha="left",
          fontsize=FS_NAME)
    box(ax, 19.6, y6, 5.6, CH, "共享专家：对每一行都算", "weight", fontsize=FS_SMALL)

    label(ax, 0.0, y6 - 1.35,
          "④ 与 ⑥ 是一对互逆的置换。专家并行下它们各是一次 all-to-all，"
          "通信量与内核见 11.8 节。",
          ha="left", fontsize=FS_TEXT, color=ACCENT)
    label(ax, 0.0, y6 - 2.15,
          "橙色是训练好就固定的权重，蓝色是随输入变化的数据，青绿色是本层新算出的路由决策。",
          ha="left", fontsize=FS_SMALL, color=MUTED)

    finish(fig, ax, OUTPUT, xlim=(-0.4, XMAX), ylim=(y6 - 2.8, 20.3))


if __name__ == "__main__":
    main()
