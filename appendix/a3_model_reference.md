## A.3 主流模型参数速查表

> 核验日期：2026-07-10。高漂移模型条目（如 GPT、Claude、Gemini、Llama 的上下文窗口、价格和发布时间）应以官方模型页、价格页和发布说明为准；本表只作为阅读时的技术快照，30 天核验门禁见 [A.5](a5_volatile_facts.md)。

| 模型 | 发布时间 | 架构类型 | 参数量 | 层数 | 隐藏维度 | 注意力头 | 上下文长度 | 关键特性 |
|------|---------|---------|-------|------|---------|---------|-----------|---------|
| Transformer | 2017.06 | Enc-Dec | 65M/213M | 6 | 512/1024 | 8/16 | - | 原始架构 |
| BERT-Base | 2018.10 | Encoder | 110M | 12 | 768 | 12 | 512 | MLM + NSP |
| BERT-Large | 2018.10 | Encoder | 340M | 24 | 1024 | 16 | 512 | MLM + NSP |
| GPT-2 | 2019.02 | Decoder | 1.5B | 48 | 1600 | 25 | 1024 | 自回归 LM |
| T5-Large | 2019.10 | Enc-Dec | 770M | 24 | 1024 | 16 | 512 | 文本到文本；2019.10 为 arXiv 预印本时间，正式发表于 2020 年 |
| GPT-3 | 2020.05 | Decoder | 175B | 96 | 12288 | 96 | 2048 | 少样本学习 |
| Llama 2-7B | 2023.07 | Decoder | 7B | 32 | 4096 | 32 | 4096 | RoPE |
| Llama 2-70B | 2023.07 | Decoder | 70B | 80 | 8192 | 64 | 4096 | GQA(8KV头) |
| Llama 3-8B | 2024.04 | Decoder | 8B | 32 | 4096 | 32 | 8192 | 128K词汇表 |
| Llama 3-70B | 2024.04 | Decoder | 70B | 80 | 8192 | 64 | 8192 | GQA |
| Llama 3.1-405B | 2024.07 | Decoder | 405B | 126 | 16384 | 128 | 128K | GQA(8KV头)；开放权重追近 GPT-4 |
| GPT-4o mini | 2024.07 | Decoder | 未公开 | - | - | - | 128K | 极致性价比 |
| Claude 3.5 Sonnet | 2024.06 | Decoder | 未公开 | - | - | - | 200K | Artifacts；Computer Use（2024.10 升级版起） |
| o1 | 2024.12 | Decoder | 未公开 | - | - | - | 200K | 推理时计算扩展 |
| o1-mini | 2024.09.12 | Decoder | 未公开 | - | - | - | 128K | 轻量级推理；o3-mini 已进入弃用表，新项目优先看当前 GPT-5.x |
| Qwen 2.5-72B | 2024.09 | Decoder | 72B | 80 | 8192 | 64 | 128K | 多语言、代码、数学 |
| Mistral 7B | 2023.09 | Decoder | 7B | 32 | 4096 | 32 | 32K | 滑动窗口注意力 |
| DeepSeek-V3 | 2024.12 | MoE-Dec | 671B(37B激活) | 61 | 7168 | 128 | 128K | MoE + FP8 |
| DeepSeek-R1 | 2025.01.20 | MoE-Dec | 671B(37B激活) | 61 | 7168 | 128 | 128K | cold-start + 多阶段训练 |
| Claude 3.7 Sonnet | 2025.02.24 | Decoder | 未公开 | - | - | - | 200K | 混合推理能力 |
| Claude Opus 4 | 2025.05 | Decoder | 未公开 | - | - | - | 200K | 多模态和智能体能力 |
| Claude Sonnet 4.5 | 2025.09.29 | Decoder | 未公开 | - | - | - | 200K | 高性能推理 |
| Claude Haiku 4.5 | 2025.10.15 | Decoder | 未公开 | - | - | - | 200K | 快速轻量级 |
| Gemini 2.5 Pro | 2025.03.25 Experimental；2025.06.17 stable GA | 多模态 | 未公开 | - | - | - | 1M | 原生多模态 |
| Gemini 3 Pro Preview | 2025.11.18 | 多模态 | 未公开 | - | - | - | 1M | 官方文档标记已于 2026-03-09 关闭 |
| Gemini 3.1 Pro Preview | 2026.02.19 | 多模态 | 未公开 | - | - | - | 1M | 当前预览代际 |
| Gemini 3.5 Flash | 2026.05.19 | 多模态 | 未公开 | - | - | - | 1,048,576 | 稳定 ID `gemini-3.5-flash`；65,536 最大输出；智能体、编码、多模态 |
| o3 | 2025.04 | Decoder | 未公开 | - | - | - | 200K | 推理模型 |
| Claude Opus 4.6 | 2026.02.05 | Decoder | 未公开 | - | - | - | 1M | 增强推理 |
| Claude Sonnet 4.6 | 2026.02.17 | Decoder | 未公开 | - | - | - | 1M | 长上下文能力 |
| Claude Opus 4.7 | 2026.04.16 | Decoder | 未公开 | - | - | - | 以官方模型页为准 | 软件工程、视觉理解和复杂多步任务 |
| Claude Opus 4.8 | 2026.05.28 | Decoder | 未公开 | - | - | - | 1M | 当前 Opus 代际；128K 输出、Adaptive Thinking、fast mode |
| Claude Fable 5 | 2026.06.09 | Decoder | 未公开 | - | - | - | 1M | 能力最强的广泛发布模型；128K 输出、Adaptive Thinking 常开、\$10/\$50；2026-07-01 与 Mythos 5 一同恢复访问 |
| Claude Mythos 5 | 2026.06.09 | Decoder | 未公开 | - | - | - | 1M | 与 Fable 5 同规格和价格，Adaptive Thinking 常开；仅限 Project Glasswing 获批客户；2026-07-01 恢复访问 |
| Claude Sonnet 5 | 2026.06.30 | Decoder | 未公开 | - | - | - | 1M | 128K 输出；Adaptive Thinking 默认开启；模型 ID `claude-sonnet-5` |
| Llama 4 Scout | 2025.04 | MoE-Dec | 109B total / 17B active | - | - | - | 10M | MoE 架构 |
| Llama 4 Maverick | 2025.04 | MoE-Dec | 400B total / 17B active | - | - | - | 1M | MoE 架构 |
| GPT-5 | 2025.08.07 | Decoder | 未公开 | - | - | - | 400,000 | 文本/图像输入推理 |
| GPT-5.1 | 2025.11 | Decoder | 未公开 | - | - | - | 400,000 | 迭代更新 |
| GPT-5.2 | 2025.12.11 | Decoder | 未公开 | - | - | - | 400,000 | 旗舰推理模型 |
| GPT-5.3-Codex | 2026.02.05 | Decoder | 未公开 | - | - | - | 400,000 | Codex 编程模型 |
| GPT-5.4 | 2026.03.05 | Decoder | 未公开 | - | - | - | 1,050,000 | 融合推理与编码 |
| GPT-5.4 mini | 2026.03 | Decoder | 未公开 | - | - | - | 400,000 | 极致性价比 |
| GPT-5.4 nano | 2026.03 | Decoder | 未公开 | - | - | - | 400,000 | 超轻量级 |
| GPT-5.5 | 2026.04.23 | Decoder | 未公开 | - | - | - | 1,050,000 | 发布时旗舰，2026-07 起由 GPT-5.6 系列接替；标准短上下文 \$5/\$30，>272K 输入按长上下文加价 |
| GPT-5.6 Sol | 2026.07.09 | Decoder | 未公开 | - | - | - | 1,050,000 | 前沿能力层；128K 最大输出；`gpt-5.6` 别名指向 Sol；Responses/Chat Completions/Batch |
| GPT-5.6 Terra | 2026.07.09 | Decoder | 未公开 | - | - | - | 1,050,000 | 智能与成本平衡层；128K 最大输出；Responses/Chat Completions/Batch |
| GPT-5.6 Luna | 2026.07.09 | Decoder | 未公开 | - | - | - | 1,050,000 | 高吞吐成本敏感层；128K 最大输出；Responses/Chat Completions/Batch |

表 A-1：主流 Transformer 模型参数速查表

### 关键缩写说明

- **Enc-Dec**：编码器-解码器架构
- **MoE-Dec**：混合专家解码器架构
- **MLM**：掩码语言模型
- **NSP**：下一句预测
- **GQA**：分组查询注意力
- **RoPE**：旋转位置编码

### 使用提示

- 表中参数与上下文长度优先用于快速建立数量级直觉；遇到版本迭代较快的闭源模型时，应再结合官方模型页、价格页和正文中的时间线说明一起阅读。
