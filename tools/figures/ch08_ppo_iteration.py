"""生成 8.2 节的示意图：一轮 PPO 迭代里四个模型谁读谁写。

正文位置：08_alignment/8.2_rlhf.md
输出：08_alignment/_images/ch08_ppo_iteration.png

左侧是采样阶段（推理引擎持有策略权重的副本），中间是三个只读模型各自交出
什么，右侧是逐词元奖励、GAE 与裁剪目标，回到对策略和价值的更新。
橙色是被更新的权重，灰色是冻结权重，蓝色是数据，青绿是本轮新算出来的量。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, FS_TEXT, INK, MUTED, NEW_EDGE,
                      WEIGHT_EDGE, arrow, box, finish, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("08_alignment", "ch08_ppo_iteration.png")

BW, BH = 4.3, 1.5


def main() -> None:
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 6.6))

    # ---- 第 1 步：采样
    x0 = 0.4
    label(ax, x0 + BW / 2, 15.3, "1 采样（rollout）", fontsize=FS_NAME, bold=True, color=ACCENT)
    p_c = box(ax, x0, 13.2, BW, BH, "策略 $\\pi_\\theta$\n被更新", kind="weight", fontsize=FS_TEXT)
    o_c = box(ax, x0, 10.9, BW, BH, "一批提示词 x\n各采样一个回答 o", kind="data", fontsize=FS_TEXT)
    arrow(ax, (p_c[0], 13.2), (o_c[0], 10.9 + BH), color=WEIGHT_EDGE)
    label(ax, x0 + BW / 2, 10.5, "权重先同步给推理引擎", fontsize=FS_SMALL, color=MUTED)

    # ---- 第 2 步：三个只读模型
    x1 = x0 + BW + 1.5
    label(ax, x1 + BW / 2, 15.3, "2 三个只读模型各算一遍", fontsize=FS_NAME, bold=True, color=ACCENT)
    ref_c = box(ax, x1, 13.2, BW, BH, "参考 $\\pi_{ref}$（冻结）\n每位 log-prob", kind="neutral")
    rm_c = box(ax, x1, 11.2, BW, BH, "奖励 $r_\\varphi$（冻结）\n只给末位一个标量", kind="neutral")
    v_c = box(ax, x1, 9.2, BW, BH, "价值 $V_\\psi$\n每位一个估值，被更新", kind="weight")
    for c in (ref_c, rm_c, v_c):
        arrow(ax, (x0 + BW, 11.65), (x1, c[1]), color=DATA_EDGE, rad=-0.12)

    # ---- 第 3 步：拼奖励、算优势
    x2 = x1 + BW + 1.5
    label(ax, x2 + BW / 2, 15.3, "3 拼奖励、算优势", fontsize=FS_NAME, bold=True, color=ACCENT)
    r_c = box(ax, x2, 13.2, BW, BH,
              "逐词元奖励 $r_t$\nKL 惩罚 + 末位 RM 分", kind="new")
    a_c = box(ax, x2, 11.2, BW, BH, "GAE\n$\\hat{A}_t$、回报 $R_t$", kind="new")
    l_c = box(ax, x2, 9.2, BW, BH, "裁剪目标 + 价值损失", kind="new")
    arrow(ax, (x1 + BW, ref_c[1]), (x2, r_c[1]), color=NEW_EDGE, rad=-0.1)
    arrow(ax, (x1 + BW, rm_c[1]), (x2, r_c[1]), color=NEW_EDGE, rad=-0.1)
    arrow(ax, (r_c[0], 13.2), (a_c[0], 11.2 + BH), color=NEW_EDGE)
    arrow(ax, (x1 + BW, v_c[1]), (x2, a_c[1]), color=NEW_EDGE, rad=0.12)
    arrow(ax, (a_c[0], 11.2), (l_c[0], 9.2 + BH), color=NEW_EDGE)

    # ---- 第 4 步：多个小批更新，权重回流
    ax.plot([l_c[0], l_c[0]], [9.2, 8.8], color=WEIGHT_EDGE, lw=1.8, zorder=1)
    ax.plot([l_c[0], -0.15], [8.8, 8.8], color=WEIGHT_EDGE, lw=1.8, zorder=1)
    ax.plot([-0.15, -0.15], [8.8, 13.95], color=WEIGHT_EDGE, lw=1.8, zorder=1)
    arrow(ax, (-0.15, 13.95), (x0, 13.95), color=WEIGHT_EDGE, lw=1.8)
    label(ax, 8.2, 8.05, "4 同一批 rollout 重复走几个小批：算比率 → 裁剪 → 反传 → 更新策略与价值",
          fontsize=FS_TEXT, bold=True, color=ACCENT)
    label(ax, 8.2, 7.35,
          "一轮结束后权重再同步回推理引擎；参考与奖励模型自始至终不更新",
          fontsize=FS_SMALL, color=MUTED)

    # 图例
    for i, (txt, col) in enumerate((("橙框：被更新的权重", WEIGHT_EDGE),
                                    ("灰框：冻结权重", MUTED),
                                    ("蓝框：数据", DATA_EDGE),
                                    ("青绿框：本轮新算出的量", NEW_EDGE))):
        label(ax, 0.4 + i * 4.2, 6.6, txt, fontsize=FS_SMALL, color=col, ha="left")

    finish(fig, ax, OUTPUT, xlim=(-0.6, 16.7), ylim=(6.1, 16.0))


if __name__ == "__main__":
    main()
