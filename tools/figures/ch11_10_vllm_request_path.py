"""生成 11.10 节插图：一个请求在 vLLM 三类进程之间走过的路径。

正文位置：11_serving/11.10_vllm_internals.md
输出：11_serving/_images/ch11_10_vllm_request_path.png

组件名取自 vLLM v0.29.0 的源码（vllm/v1/engine、vllm/v1/core、vllm/v1/executor、
vllm/v1/worker）。图上的编号与正文 11.10.1 的八个步骤一一对应。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from _diagram import (ACCENT, DATA_EDGE, FS_NAME, FS_SMALL, MUTED, NEUTRAL_EDGE,
                      NEW_EDGE, arrow, box, label)
from _style import image_path, use_cjk_font

OUTPUT = image_path("11_serving", "ch11_10_vllm_request_path.png")


def band(ax, y, h, title, note):
    """画一条代表进程的底框，左上角写进程名。"""
    ax.add_patch(FancyBboxPatch((0.2, y), 25.6, h, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="#fafaf8", edgecolor=NEUTRAL_EDGE,
                                linewidth=1.2, linestyle=(0, (5, 3)), zorder=0))
    label(ax, 0.55, y + h - 0.45, title, ha="left", fontsize=FS_NAME, bold=True)
    label(ax, 25.45, y + h - 0.45, note, ha="right", fontsize=FS_SMALL, color=MUTED)


def step(ax, x, y, n, color=ACCENT):
    """手绘带圈数字，避免依赖字体里的 ①②③。"""
    ax.add_patch(Circle((x, y), 0.34, facecolor="white", edgecolor=color, linewidth=1.6, zorder=8))
    label(ax, x, y - 0.02, str(n), fontsize=FS_SMALL, color=color, bold=True, zorder=9)


def main():
    use_cjk_font()
    fig = plt.figure(figsize=(10.0, 8.35))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    # ---------------- API 服务进程 ----------------
    band(ax, 16.2, 5.2, "API 服务进程", "CPU：HTTP、分词、反分词")
    box(ax, 0.8, 16.8, 5.0, 3.0, "OpenAI 兼容接口\n/v1/chat/completions", "neutral", fontsize=FS_SMALL)
    box(ax, 7.0, 18.5, 4.8, 1.7, "Renderer + InputProcessor\n套模板、分词、校验", "neutral", fontsize=FS_SMALL)
    box(ax, 7.0, 16.5, 4.8, 1.7, "OutputProcessor\n增量反分词、停止串", "neutral", fontsize=FS_SMALL)
    box(ax, 13.0, 16.8, 4.9, 3.0, "AsyncLLM\n每个请求\n一条输出队列", "neutral", fontsize=FS_SMALL)
    box(ax, 19.2, 16.8, 6.0, 3.0, "EngineCoreClient\n（AsyncMPClient）", "neutral", fontsize=FS_SMALL)
    arrow(ax, (5.8, 19.0), (7.0, 19.35))
    arrow(ax, (11.8, 19.35), (13.0, 19.0))
    arrow(ax, (17.9, 19.0), (19.2, 19.0))
    arrow(ax, (19.2, 17.5), (17.9, 17.5), color=NEW_EDGE)
    arrow(ax, (13.0, 17.5), (11.8, 17.35), color=NEW_EDGE)
    arrow(ax, (7.0, 17.35), (5.8, 17.5), color=NEW_EDGE)
    step(ax, 6.4, 19.75, 1)
    step(ax, 6.4, 16.85, 8, NEW_EDGE)

    # ---------------- ZMQ ----------------
    arrow(ax, (20.6, 16.8), (20.6, 14.6), color=DATA_EDGE, lw=1.8)
    arrow(ax, (23.0, 14.6), (23.0, 16.8), color=NEW_EDGE, lw=1.8)
    label(ax, 20.0, 15.4, "ZMQ + msgpack：EngineCoreRequest（词元 ID、采样参数）", ha="right",
          fontsize=FS_SMALL, color=DATA_EDGE)
    label(ax, 23.55, 15.4, "新词元\nID", ha="left", fontsize=FS_SMALL, color=NEW_EDGE)
    step(ax, 20.6, 15.75, 2)
    step(ax, 23.0, 15.75, 7, NEW_EDGE)

    # ---------------- 引擎核心进程 ----------------
    band(ax, 7.2, 7.4, "引擎核心进程 EngineCoreProc", "")
    arrow(ax, (20.6, 14.6), (20.6, 13.25), color=DATA_EDGE, lw=1.8, style="-")
    arrow(ax, (20.6, 13.25), (3.1, 13.25), color=DATA_EDGE, lw=1.4, style="-")
    arrow(ax, (3.1, 13.25), (3.1, 12.4), color=DATA_EDGE, lw=1.4)
    label(ax, 15.2, 13.65, "输入线程 → input_queue → add_request", fontsize=FS_SMALL, color=DATA_EDGE)
    box(ax, 0.8, 7.9, 4.6, 4.5, "Scheduler\nwaiting 队列\nrunning 列表\n词元预算", "data", fontsize=FS_SMALL)
    box(ax, 6.8, 10.5, 5.2, 1.9, "KVCacheManager\nallocate_slots", "data", fontsize=FS_SMALL)
    box(ax, 6.8, 7.9, 5.2, 1.9, "BlockPool\n空闲队列 + 哈希表", "data", fontsize=FS_SMALL)
    box(ax, 13.2, 7.9, 5.0, 4.5, "SchedulerOutput\n每个请求本步\n算几个词元、\n新增哪些块号", "data", fontsize=FS_SMALL)
    box(ax, 19.4, 7.9, 5.8, 4.5, "Executor\nuni / mp / ray", "neutral", fontsize=FS_SMALL)
    arrow(ax, (5.4, 11.45), (6.8, 11.45))
    arrow(ax, (9.4, 10.5), (9.4, 9.8))
    arrow(ax, (12.0, 10.15), (13.2, 10.15))
    arrow(ax, (18.2, 10.15), (19.4, 10.15))
    arrow(ax, (23.0, 12.4), (23.0, 14.6), color=NEW_EDGE, lw=1.8, style="-")
    step(ax, 6.1, 12.05, 3)

    # ---------------- 共享内存广播 ----------------
    arrow(ax, (20.6, 7.9), (20.6, 5.9), color=DATA_EDGE, lw=1.8)
    arrow(ax, (23.0, 4.5), (23.0, 7.9), color=NEW_EDGE, lw=1.8)
    label(ax, 20.0, 6.55, "共享内存 MessageQueue：广播给全部 worker", ha="right",
          fontsize=FS_SMALL, color=DATA_EDGE)
    label(ax, 23.55, 6.55, "一个 rank\n回传", ha="left", fontsize=FS_SMALL, color=NEW_EDGE)
    step(ax, 20.6, 6.9, 4)
    step(ax, 23.0, 6.9, 6, NEW_EDGE)

    # ---------------- worker 进程 ----------------
    band(ax, 0.2, 5.7, "GPU worker 进程 × (TP × PP)", "")
    label(ax, 22.4, 5.45, "每张卡一个；单卡时 worker 建在引擎核心进程内", ha="right",
          fontsize=FS_SMALL, color=MUTED)
    box(ax, 0.8, 0.8, 5.0, 3.7, "GPUModelRunner\n请求状态、块表\nprepare_inputs", "data", fontsize=FS_SMALL)
    box(ax, 7.0, 2.85, 5.0, 1.65, "模型权重", "weight", fontsize=FS_SMALL)
    box(ax, 7.0, 0.8, 5.0, 1.65, "KV 缓存张量（按块）", "data", fontsize=FS_SMALL)
    box(ax, 13.2, 0.8, 6.0, 3.7, "前向\n小批：回放 CUDA Graph\n大批：逐个发射内核", "neutral", fontsize=FS_SMALL)
    box(ax, 20.4, 0.8, 4.8, 3.7, "Sampler\n每个请求\n一个新词元", "new", fontsize=FS_SMALL)
    arrow(ax, (5.8, 2.65), (7.0, 2.65))
    arrow(ax, (12.0, 3.65), (13.2, 3.2))
    arrow(ax, (12.0, 1.65), (13.2, 2.1), style="<|-|>")
    arrow(ax, (19.2, 2.65), (20.4, 2.65))
    step(ax, 12.6, 4.35, 5)

    ax.set_xlim(0, 26)
    ax.set_ylim(0, 21.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
