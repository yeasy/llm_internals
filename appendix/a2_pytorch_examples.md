## A.2 PyTorch 实现示例

以下代码展示了 Transformer 核心组件的 PyTorch 实现。更多与正文数学推导配套的可运行代码示例，可参见[第 2 章](../02_attention/2.2_scaled_dot_product.md)、[第 3 章](../03_components/3.4_feedforward.md)与[第 4 章](../04_position_encoding/4.3_rope.md)中的内嵌代码。

完成本附录后，继续用 [A.6](a6_practice_path.md)核对手算梯度、精确恢复训练、KV Cache 等价性与后训练目标，再用 [A.7](a7_framework_recipes.md)迁移到开源训练框架和推理服务。

### 缩放点积注意力

下面的函数显式区分三类掩码：`is_causal` 表示因果注意力，`key_padding_mask` 用布尔值标记有效键位置，`attn_mask` 支持布尔可见性掩码或加性 bias 掩码。

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, attn_mask=None, key_padding_mask=None, is_causal=False):
    """缩放点积注意力。

    布尔 mask 中 True 表示该位置可见；加性 mask 会直接加到 attention scores 上。
    """
    if Q.dim() not in (3, 4) or K.dim() != Q.dim() or V.dim() != Q.dim():
        raise ValueError("Q, K, V must all be 3D (B, L, D) or 4D (B, H, L, D)")

    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if is_causal:
        q_len, k_len = Q.size(-2), K.size(-2)
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=Q.device).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~causal, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.reshape((1,) * (scores.dim() - 2) + attn_mask.shape)
        elif attn_mask.dim() == 3 and scores.dim() == 4:
            attn_mask = attn_mask[:, None, :, :]
        elif attn_mask.dim() != scores.dim():
            raise ValueError("attn_mask must broadcast to attention scores")

        attn_mask = attn_mask.to(device=scores.device)
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype)

    if key_padding_mask is not None:
        if key_padding_mask.shape != (Q.size(0), K.size(-2)):
            raise ValueError("key_padding_mask must be (batch, key_len)")
        if scores.dim() == 3:
            valid_keys = key_padding_mask[:, None, :]
        else:
            valid_keys = key_padding_mask[:, None, None, :]
        valid_keys = valid_keys.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~valid_keys, float("-inf"))

    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    safe_scores = scores.masked_fill(fully_masked, 0.0)
    attn_weights = F.softmax(safe_scores, dim=-1)
    attn_weights = attn_weights.masked_fill(fully_masked, 0.0)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

### 多头注意力

多头注意力需要 `d_model` 能被注意力头数整除，否则无法把投影后的张量均匀拆成多个头。

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, attn_mask=None, key_padding_mask=None, is_causal=False):
        batch_size = Q.size(0)
        # 投影并拆分为多头
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 计算注意力
        out, _ = scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)
```

### 前馈网络

下面的代码块延续前文导入的 `nn` 和 `F`，展示逐位置前馈网络的最小结构。

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

### RMSNorm

与 [3.6 节](../03_components/3.6_layer_norm.md)的公式对应：不做去均值，仅以均方根缩放后乘可学习增益。

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS(x) = sqrt(mean(x^2) + eps)；对比 LayerNorm 省去了减均值
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

PyTorch 2.4 起也提供内置的 `nn.RMSNorm`，行为与上述实现一致，可直接相互对照验证。

### 旋转位置编码（RoPE）

与 [4.3 节](../04_position_encoding/4.3_rope.md)对应：按 $\theta_i = 10000^{-2i/d}$ 预计算角度表，对 Q、K 的每对相邻维度做二维旋转。

```python
def rope_frequencies(d, max_len, base=10000.0, device=None):
    """预计算各位置、各二维子空间的旋转角度表"""
    if d % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    i = torch.arange(0, d, 2, dtype=torch.float32, device=device)   # 0, 2, ..., d-2
    theta = base ** (-i / d)                         # (d/2,)，即 10000^(-2j/d)（i = 2j）
    m = torch.arange(max_len, dtype=torch.float32, device=device)   # 位置 0, 1, ..., max_len-1
    angles = torch.outer(m, theta)                   # (max_len, d/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """对形如 (..., seq_len, d) 的 Q 或 K 应用 RoPE"""
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    x1, x2 = x[..., 0::2], x[..., 1::2]              # 相邻两维构成一个二维子空间
    c = cos[: x.size(-2)].to(device=x.device, dtype=x.dtype)
    s = sin[: x.size(-2)].to(device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s                 # 逐子空间做二维旋转
    out[..., 1::2] = x1 * s + x2 * c
    return out
```

可以直接验证 4.3 节的核心性质——旋转后的注意力分数只依赖相对位置：

```python
d = 8
cos, sin = rope_frequencies(d, max_len=64)
q, k = torch.randn(1, d), torch.randn(1, d)
rot = lambda v, pos: apply_rope(v, cos[pos:pos + 1], sin[pos:pos + 1])

s1 = rot(q, 3) @ rot(k, 1).T    # 位置 (3, 1)，相对距离 2
s2 = rot(q, 23) @ rot(k, 21).T  # 位置 (23, 21)，相对距离仍为 2
print(torch.allclose(s1, s2, atol=1e-5))  # True：分数只随 m-n 变化
```

注意：Llama、Hugging Face 等生产实现通常采用“前半-后半”配对（`rotate_half`）而非这里的相邻维度交错配对。两种布局在数学上等价（相差一个固定的维度置换），但已训练权重与具体布局绑定，移植权重时不能混用。

### 完整实验：训练、保存、加载与生成

本书源码中的 `examples/tiny_gpt.py` 将前面的组件组成一个可在 CPU 上训练的微型 GPT：2 层、每层 2 个注意力头，模型维度为 32，每个头的维度为 16。模型使用可学习的位置嵌入、前置 LayerNorm、因果自注意力、GELU 前馈网络和残差连接；各层参数独立，Q/K/V 投影参数通过训练学习，不使用恒等变换。

**实验目标是理解训练闭环，不是训练通用语言模型。** 数据只有 `1+1=2。`、`1+2=3。`、`2+2=4。`、`2+3=5。` 四条文本，每个字符作为一个词元，另加结束词元 `<eos>`。这四条文本都用于训练，没有验证集；生成正确只能说明模型记住了这些样本，不能证明它能计算未见过的加法。

#### 1. 准备输入与目标

以 `2+3=5。` 为例，先追加结束词元，再将同一序列错开一位：

```text
原序列：2   +   3   =   5   。   <eos>
输入：  2   +   3   =   5   。
目标：  +   3   =   5   。  <eos>
```

输入中的每个位置都预测下一个词元。例如，位置 `=` 的输出用于预测 `5`，位置 `5` 的输出用于预测 `。`。因果掩码阻止当前位置读取右侧输入，因此训练时即使整条序列一起输入，也不会提前看到要预测的答案。

四条文本等长，可以直接组成一个批次，不需要填充。词表含 9 个词元：结束词元、`+`、数字 `1` 至 `5`、`=` 和 `。`。

#### 2. 前向计算与参数更新

```text
输入词元编号 [4, 6]
    ↓ 词元嵌入 + 位置嵌入
词元向量 [4, 6, 32]
    ↓ 第 1 层：归一化 → 双头注意力 → 残差 → 归一化 → MLP → 残差
中间向量 [4, 6, 32]
    ↓ 第 2 层：重复相同结构，使用另一组参数
中间向量 [4, 6, 32]
    ↓ 最终归一化 + 词表投影
每个位置的词表分数 [4, 6, 9]
    ↓ 与目标编号 [4, 6] 计算交叉熵
标量损失 → 反向传播 → 梯度裁剪 → AdamW 更新参数
```

形状中的 `4` 是样本数，`6` 是每条输入的词元数，`32` 是每个词元的表示维度，`9` 是词表大小。每层中，Q、K、V 拆头后的形状均为 `[4, 2, 6, 16]`，注意力矩阵为 `[4, 2, 6, 6]`；合并双头后恢复为 `[4, 6, 32]`。

令 $p_{b,t}$ 表示模型在第 $b$ 条样本、第 $t$ 个位置赋予正确目标词元的概率，则本批次损失为：

$$
\mathcal{L}=-\frac{1}{4\times 6}\sum_{b=1}^{4}\sum_{t=1}^{6}\log p_{b,t}.
$$

正确目标的概率越大，该位置的损失越小。代码中的 `cross_entropy` 直接接收未经 Softmax 的分数，内部完成所需计算；不要提前对 `logits` 做 Softmax。下面是完整脚本的核心循环，`training_batch` 和 `TinyGPT` 均在脚本中定义：

```python
inputs, targets = training_batch()
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
model.train()
for step in range(200):
    optimizer.zero_grad(set_to_none=True)  # 清除上一次更新的梯度
    logits = model(inputs)               # [4, 6, 9]
    loss = F.cross_entropy(
        logits.reshape(-1, len(VOCAB)),  # [24, 9]：24 个位置的预测
        targets.reshape(-1),            # [24]：各位置的正确词元编号
    )
    loss.backward()                     # 计算损失对各参数的梯度
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()                     # 根据梯度更新参数
```

每次更新都使用全部四条训练样本。训练同时计算所有位置的损失；生成时只用最后一个位置的分数选出下一个词元，两者不要混淆。

#### 3. 运行、保存与重新加载

在本书源码目录中执行以下命令。依赖安装在独立环境中，不需要下载预训练模型或数据集；示例使用 Python 3.11、PyTorch 2.8.0 验证。

```bash
python3.11 -m venv /tmp/llm-training-env
/tmp/llm-training-env/bin/python -m pip install torch==2.8.0 numpy
/tmp/llm-training-env/bin/python examples/tiny_gpt.py --checkpoint /tmp/tiny-gpt.pt
```

一次 CPU 运行的输出如下；不同运行环境下损失末位可能略有差异。

```text
loss: 2.3766 -> 0.1184
generation: 2+3=5。
```

本例的训练损失无法降到零。例如，文本开头同为 `1+` 的两条样本，下一个词元分别是 `1` 和 `2`；模型仅凭这个前缀无法确定哪一个才是本条样本的目标。

脚本保存参数和词表，再创建一个新模型加载参数，最后运行生成。以下命令只加载，不训练；保存路径已存在时，训练命令会拒绝覆盖，请改用新路径。

```bash
/tmp/llm-training-env/bin/python examples/tiny_gpt.py --checkpoint /tmp/tiny-gpt.pt --load-only
```

参数通过 `state_dict` 保存，读取时使用 `weights_only=True`；文件用于本例推理，不包含优化器状态，不能精确恢复中断时的训练进度。只加载可信来源的文件。保存与加载机制可参见 [PyTorch 官方教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。

#### 4. 观察逐词元生成

```text
输入「2+3=」   → 最后位置的分数 → 选择「5」
输入「2+3=5」  → 最后位置的分数 → 选择「。」
输入「2+3=5。」→ 最后位置的分数 → 选择 <eos> → 停止
```

每轮通过全部两层后，才选出一个新词元；选出的词元追加到输入，再开始下一轮。生成期间参数不更新，使用 `eval()` 和 `no_grad()`；脚本采用贪心选择，遇到结束词元、长度上限或新增词元数量上限即停止。

为便于逐行阅读，本实验每轮重算完整前缀，**没有实现 KV Cache**。它演示自回归依赖关系，但不能据此理解为高效 Decode 也会重算整个历史；使用缓存时只输入最新词元，详见 [3.8 节](../03_components/3.8_gpt_inference_flow.md)。

配套测试检查目标错位、因果掩码、参数更新、损失下降、保存前后输出一致及生成结果：

```bash
/tmp/llm-training-env/bin/python -m unittest tests.test_tiny_gpt -v
```
