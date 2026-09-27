"""图1-0：技术职责与基础设施组织；箭头表示依赖而非请求顺序。"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _style import image_path, use_cjk_font

OUTPUT = image_path("01_introduction", "llm_technology_map.png")


def main():
    use_cjk_font()
    fig, ax = plt.subplots(figsize=(11, 13))

    def box(x, y, w, h, title, body, color="#edf3fa"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, edgecolor="#8093a5", linewidth=1.2))
        ax.text(x + w / 2, y + .30, title, ha="center", va="center",
                fontsize=16, weight="bold", color="#19354a")
        ax.text(x + w / 2, y + h / 2 + .19, body, ha="center", va="center",
                fontsize=13, linespacing=1.55, color="#273d4b")

    def arrow(x, y, xx, yy):
        ax.annotate("", xy=(xx, yy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color="#45687d", lw=1.5))

    ax.text(5, .2, "LLM 技术全景：从应用到底层资源", ha="center",
            va="center", fontsize=21, weight="bold", color="#19354a")
    ax.text(5, .70, "向下追踪依赖，向上理解能力；两条路径共享底层资源", ha="center",
            fontsize=13, color="#536a79")
    box(.3, 1.1, 9.4, 1.35, "应用与 Agent · §14.5",
        "RAG 检索证据 · 工作流规定步骤 · Agent 选择工具\n宿主执行动作，校验权限与结果")
    arrow(7.5, 2.45, 7.5, 2.85)
    box(.3, 2.85, 9.4, 1.3, "模型服务 · §11.13",
        "API / 网关 · 路由与限流 · 扩缩容\n输入请求 → 流式结果")
    arrow(7.5, 4.15, 7.5, 4.65)
    box(.3, 4.65, 4.15, 2.25, "训练与后训练 · 第5—8章",
        "数据 → 前向 / 反向 → 参数更新\nSFT / DPO / RLHF · LoRA\nPyTorch · FSDP / DeepSpeed\nMegatron · TRL", "#fff0dd")
    box(5.55, 4.65, 4.15, 2.25, "推理执行 · 第9—11章",
        "请求 → Prefill → 逐轮 Decode\nKV Cache · 连续批处理\nvLLM · SGLang\nTensorRT-LLM", "#e5f3ee")
    arrow(4.5, 5.8, 5.5, 5.8)
    ax.text(5, 5.12, "权重\n配置\n分词器", ha="center", va="center", fontsize=11)
    ax.text(5, 7.24, "两侧都执行模型计算：Attention · MLP · 归一化 · 残差（第2—4章）",
            ha="center", fontsize=12, color="#536a79", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    arrow(2.4, 6.9, 2.4, 7.65)
    arrow(7.6, 6.9, 7.6, 7.65)
    box(.3, 7.65, 9.4, 1.6, "算子、运行时与通信 · §11.14",
        "张量运算 → CUDA / Triton 内核 → GPU 执行\n融合 · FlashAttention · 低精度计算（第7、10章）\n跨 GPU 的集合通信：NCCL")
    arrow(5, 9.25, 5, 9.7)
    box(.3, 9.7, 9.4, 3.3, "硬件与基础设施 · §11.12",
        "GPU / HBM：计算、权重与中间状态\n服务器：CPU / 内存 / 存储 / GPU，PCIe 与 NVLink 互连\n集群：多节点调度、RDMA 网络、共享存储\n数据中心：机房、供电、散热与物理故障域\n集群是逻辑组织；数据中心是物理设施", "#f0edf6")
    box(.3, 13.45, 9.4, 1.35, "贯穿各层，不是额外执行步骤",
        "数据治理 · 质量评估 · 安全与权限 · 可观测性 · 成本\n§5.5 / §8.6 / §11.13 / §14.5", "#f5f4ef")
    ax.text(5, 15.2, "主职责图，不是产品边界图；在线后训练还会调用生成服务。",
            ha="center", fontsize=12, color="#536a79")
    ax.set(xlim=(0, 10), ylim=(15.55, 0))
    ax.axis("off")
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
