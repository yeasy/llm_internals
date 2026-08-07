## 本章小结

**BERT** 开创了双向预训练范式，在理解型任务上取得了突破性成果。**RoBERTa** 证明了充分训练的重要性，**ALBERT** 展示了参数效率优化的可行性，**ELECTRA** 通过替换检测任务大幅提升了训练效率。**Longformer** 和 **BigBird** 通过稀疏注意力使编码器能够高效处理长文本。

既然“没有因果掩码、每个位置都能同时关注前后文”被认定为编码器的核心优势，为什么主流最终倒向了只能从左往右看的解码器？

---

> 📝 **发现错误或有改进建议？** 欢迎提交 [Issue](https://github.com/yeasy/llm_internals/issues) 或 [PR](https://github.com/yeasy/llm_internals/pulls)。
