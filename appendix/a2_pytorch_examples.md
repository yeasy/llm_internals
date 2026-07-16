## A.2 PyTorch 实现示例

以下代码展示了 Transformer 核心组件的 PyTorch 实现。更多与正文数学推导配套的可运行代码示例，可参见[第2章](../02_attention/2.2_scaled_dot_product.md)、[第3章](../03_components/3.4_feedforward.md)与[第4章](../04_position_encoding/4.3_rope.md)中的内嵌代码。

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

与 [4.3 节](../04_position_encoding/4.3_rope.md)对应：按 $$\theta_i = 10000^{-2i/d}$$ 预计算角度表，对 Q、K 的每对相邻维度做二维旋转。

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

### 训练循环示例

下面是训练循环骨架，省略了具体模型类、数据加载器和 epoch 配置，重点展示损失计算、梯度裁剪和优化器更新的位置。

```python
model = MyTransformerModel(vocab_size=32000, d_model=512, n_heads=8, n_layers=6)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, target_ids = batch
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```
