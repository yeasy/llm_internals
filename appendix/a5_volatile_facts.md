## A.5 快变事实核验表

> Last verified: 2026-06-17. 本表用于维护模型时间线、参数、价格、benchmark、硬件和 serving 生态的高波动事实。高波动条目应尽量在正文或附录参考中保留来源入口和核验日期。

| 类别 | 当前维护口径 | 权威入口 | 编辑要求 |
| --- | --- | --- | --- |
| 模型时间线 | GPT、Claude、Gemini、Llama、DeepSeek、Qwen 等时间线以官方发布和 model card 为准；Claude Fable 5 / Mythos 5 于 2026-06-09 发布，曾于 2026-06-12 起因出口管制暂停访问；截至 2026-07-09，官方模型页已恢复将 Fable 5 列为正常提供（GA）、Mythos 5 为 Project Glasswing 受限可用（未见单独恢复公告，以模型页现状为准）。 | [OpenAI Models](https://developers.openai.com/api/docs/models/all/), [OpenAI ChatGPT retirement](https://help.openai.com/en/articles/20001051), [Claude Models](https://platform.claude.com/docs/en/about-claude/models/overview), [Fable/Mythos access statement](https://www.anthropic.com/news/fable-mythos-access), [Gemini Models](https://ai.google.dev/gemini-api/docs/models), [Meta Llama](https://www.llama.com/), [Qwen Blog](https://qwenlm.github.io/blog/) | 时间线章节必须标注“公开资料快照”，并区分 API 状态与 ChatGPT/网页产品状态；Claude 侧“最强/旗舰”表述须区分发布规格、当前可用性、Fable 5（全系）与 Opus 4.8（Opus 档）。 |
| 参数与上下文 | 参数量、MoE active params、context window、tokenizer 变更以模型卡或技术报告为准。 | 官方 model card / technical report；Llama 4、Qwen MoE 路由等细节优先看官方博客、技术报告和模型卡 | 不用传闻数字补表；上下文 token 不要直接换算为“百万行代码”。 |
| 推理引擎 | vLLM、TensorRT-LLM、SGLang、TGI 等部署状态以项目文档和 release 为准。 | 项目官方文档 / release | 避免写“某厂商已大规模使用”除非有公开证据。 |
| 硬件 | NVIDIA、TPU、AI ASIC 等路线图以厂商公开资料为准。 | 厂商发布页和技术白皮书；TPU 8t/8i、H100/H200/B200 等条目需核对发布时间和可用状态 | 对未来硬件写 planned / announced，不写已可用。 |
| Benchmark | SWE-bench、GPQA、MMLU、long-context eval 等只作为 dated snapshot。 | benchmark 官方站点、论文、仓库 | 说明评测集、版本和污染风险。 |
| 新近论文快照数字 | 正文引用的 2026 年新近研究（3.5.5 AttnRes 的基准增益、8.3.4 IH-Challenge 的 +10.0% 等）按论文当时版本记录。 | [AttnRes](https://arxiv.org/abs/2603.15031), [IH-Challenge](https://arxiv.org/abs/2603.10521), [Making Claude a chemist](https://www.anthropic.com/research/making-claude-a-chemist) | 复核时对照 arXiv/官方页当前版本；数字变动则更新正文快照口径。 |
