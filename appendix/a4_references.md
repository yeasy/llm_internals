## A.4 推荐阅读与参考文献

### 核心论文

1. Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS 2017*.
2. Devlin, J., et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *NAACL 2019*.
3. Radford, A., et al. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). *OpenAI*.
4. Brown, T., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). *NeurIPS 2020*.
5. Raffel, C., et al. (2020). [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683). *JMLR 2020*.
6. Lewis, M., et al. (2020). [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461). *ACL 2020*.
7. Liu, Y., et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).
8. Lan, Z., et al. (2019). [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942).
9. Clark, K., et al. (2020). [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555). *ICLR 2020*.
10. Beltagy, I., et al. (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
11. Zaheer, M., et al. (2020). [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). *NeurIPS 2020*.
12. Kaplan, J., et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
13. Hoffmann, J., et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). *Chinchilla*.

### 架构改进

14. Su, J., et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
15. Press, O., et al. (2021). [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).
16. Peng, B., et al. (2023). [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071). *ICLR 2024*.
17. Geva, M., et al. (2021). [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913). *EMNLP 2021*.
18. Dao, T., et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). *NeurIPS 2022*.
19. Shazeer, N. (2019). [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150).
20. Ainslie, J., et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). *EMNLP 2023*.
21. Jiang, A. Q., et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825).
22. Xiao, G., et al. (2023). [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453). *ICLR 2024*.

### 训练与对齐

23. Kingma, D. P. & Ba, J. (2014). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). *ICLR 2015*.
24. Loshchilov, I. & Hutter, F. (2017). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). *ICLR 2019*.
25. Ouyang, L., et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). *NeurIPS 2022*.
26. Rafailov, R., et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). *NeurIPS 2023*.
27. DeepSeek-AI. (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).
28. Hu, E., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR 2022*.
29. Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). *NeurIPS 2023*.
30. Rajbhandari, S., et al. (2019). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
31. Microsoft DeepSpeed Team. (2021). [DeepSpeed: Accelerating large-scale model inference and training via system optimizations and compression](https://www.deepspeed.ai/2021/05/14/inference-release.html).
32. Shoeybi, M., et al. (2019). [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053).

### 推理优化

33. Kwon, W., et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). *SOSP 2023*.
34. Leviathan, Y., et al. (2023). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). *ICML 2023*.
35. Cai, T., et al. (2024). [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774).
36. Li, Y., et al. (2024). [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077).
37. Nguyen, M., et al. (2024). [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082).
38. Liu, H., et al. (2023). [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889).
39. Milakov, M. & Gimelshein, N. (2018). [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). *NVIDIA*.
40. Zadouri, T., et al. (2026). [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451).

### 前沿架构

41. Gu, A. & Dao, T. (2023; revised 2024). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
42. DeepSeek-AI. (2024). [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434).
43. DeepSeek-AI. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).
44. Fedus, W., et al. (2021). [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). *JMLR 2022*.
45. Wei, J., et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). *NeurIPS 2022*.
46. Yao, S., et al. (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). *ICLR 2023*.
47. Yao, S., et al. (2023). [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601).
48. Besta, M., et al. (2023). [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687).

### 教程与可视化

49. Jay Alammar. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
50. Lilian Weng. [The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/).
51. HuggingFace. [Transformers Documentation](https://huggingface.co/docs/transformers/).

### 推荐书籍

52. Jurafsky, D. & Martin, J.H. *Speech and Language Processing* (3rd ed.). 第10章 Transformer 部分。
53. 邱锡鹏. 《神经网络与深度学习》. 第15章注意力机制与 Transformer。

### 官方模型、硬件与工程资料

54. OpenAI. [GPT-5 Model Documentation](https://developers.openai.com/api/docs/models/gpt-5).
55. OpenAI. [Retiring GPT-4o and other ChatGPT models](https://help.openai.com/en/articles/20001051).
56. Meta AI. [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
57. Google. [Introducing Gemini 1.5](https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/).
58. Google Cloud. [Our eighth generation TPUs: TPU 8t and TPU 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/).
59. NVIDIA. [H100 GPU Product Specifications](https://www.nvidia.com/en-us/data-center/h100/).
60. PyTorch. [DistributedDataParallel Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
61. DeepSpeed. [ZeRO Documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html).
62. Qwen Team. [Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters](https://qwenlm.github.io/blog/qwen-moe/).
63. Qwen. [Qwen2-57B-A14B Model Card](https://huggingface.co/Qwen/Qwen2-57B-A14B).
64. Qwen Team. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).
