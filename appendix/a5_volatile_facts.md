## A.5 快变事实核验表

<!-- volatile-facts: verified_at=2026-07-10 expires_at=2026-08-09 ttl_days=30 conflict_status=resolved-conflict -->

> 核验日期：2026-07-10；到期日期：2026-08-09。这里的模型时间线、价格、上下文、硬件可用性和 serving 生态均属于快变事实。30 天到期后，项目检查会拒绝通过，必须重新访问权威入口并更新核验元数据。

| 类别 | 当前维护口径 | 权威入口 | 编辑要求 |
| --- | --- | --- | --- |
| 模型时间线 | OpenAI 于 2026-07-09 将 **GPT-5.6 Sol、Terra、Luna** 发布到 `v1/responses`、`v1/chat/completions` 和 `v1/batch`；`gpt-5.6` 别名指向 Sol。Anthropic 于 2026-06-30 发布 Claude Sonnet 5；Claude Fable 5 / Mythos 5 于 6 月 9 日发布、6 月 12 日暂停，并于 7 月 1 日恢复访问，其中 Fable 5 为 GA、Mythos 5 仅限 Project Glasswing 获批客户。Gemini 3.5 Flash 的稳定模型 ID 为 `gemini-3.5-flash`。 | [OpenAI API changelog](https://developers.openai.com/api/docs/changelog), [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra), [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [Anthropic release notes](https://platform.claude.com/docs/en/release-notes/overview), [Claude Models](https://platform.claude.com/docs/en/about-claude/models/overview), [Fable/Mythos access statement](https://www.anthropic.com/news/fable-mythos-access), [Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash), [Meta Llama](https://www.llama.com/), [Qwen Blog](https://qwenlm.github.io/blog/) | 时间线必须区分 API、ChatGPT/网页产品和预览状态。2026-06-26 的 OpenAI 模型目录抓取曾显示 GPT-5.6 为受信任合作方预览，但更晚的 2026-07-09 官方 changelog 与独立模型详情页明确显示 API 已发布；本冲突状态为 **resolved-conflict**，以后者为准。Claude 侧“最强/旗舰”表述须区分 Fable 5（全系）与 Opus 4.8（Opus 档）。 |
| 参数与上下文 | 参数量、MoE active params、context window、max output 和 tokenizer 变更以模型卡或技术报告为准。GPT-5.6 三个层级的官方模型页均列出 1,050,000 上下文和 128,000 最大输出。 | 官方 model card / technical report；Llama 4、Qwen MoE 路由等细节优先看官方博客、技术报告和模型卡 | 不用传闻数字补表；上下文词元不要直接换算为“百万行代码”。 |
| 推理引擎 | vLLM、TensorRT-LLM、SGLang、TGI 等部署状态以项目文档和 release 为准。 | 项目官方文档 / release | 避免写“某厂商已大规模使用”，除非有公开证据。 |
| 硬件 | Google 已公开 TPU 8t（训练，216 GB HBM、6528 GB/s）与 TPU 8i（推理，288 GB HBM、8601 GB/s），但 Cloud TPU 当前目录仍标为 **Coming soon**；Ironwood TPU7x 已于 2026-03-31 GA。NVIDIA Vera Rubin 已进入 full production，但实际云区域和实例可用性仍需单独核验。 | [TPU 8t/8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive), [Cloud TPU](https://cloud.google.com/tpu), [Cloud TPU release notes](https://docs.cloud.google.com/tpu/docs/release-notes), [Vera Rubin production](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory) | 把“发布/量产”“Cloud SKU 可申请”和“特定区域有库存”分开写；Coming soon 不得写成已可用。 |
| Benchmark | SWE-bench、GPQA、MMLU、long-context eval 等只作为 dated snapshot。 | benchmark 官方站点、论文、仓库 | 说明评测集、版本、是否使用工具和污染风险。 |
| 新近论文快照数字 | 正文引用的 2026 年新近研究（3.5.5 AttnRes 的基准增益、8.3.4 IH-Challenge 的 +10.0% 等）按论文当时版本记录。 | [AttnRes](https://arxiv.org/abs/2603.15031), [IH-Challenge](https://arxiv.org/abs/2603.10521), [Making Claude a chemist](https://www.anthropic.com/research/making-claude-a-chemist) | 复核时对照 arXiv/官方页当前版本；数字变动则更新正文快照口径。 |

`conflict_status=resolved-conflict` 不是“来源永远一致”的声明，而是记录本轮已发现的 GPT-5.6 预览/已发布冲突已有明确裁决。若权威来源再次互相冲突，应先把状态改为 `open-conflict`；项目检查会失败，直到正文、附录和来源链完成重新核验并改回 `resolved-conflict`。
