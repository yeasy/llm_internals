## A.1 数学基础速查

### 线性代数基础

**矩阵乘法**：$C = AB$，其中 $C_{ij} = \sum_k A_{ik} B_{kj}$。复杂度为 $O(M \times N \times K)$。

**点积**：$a \cdot b = \sum_i a_i b_i = \|a\|\|b\|\cos\theta$，衡量两个向量的方向一致性。

**转置**：$(AB)^T = B^T A^T$。

### Softmax 函数

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

将实数向量转换为概率分布（非负且和为 1）。具有平移不变性：$\text{softmax}(z + c) = \text{softmax}(z)$。

### 交叉熵

$$H(p, q) = -\sum_x p(x) \log q(x)$$

当 $p$ 为独热分布时，$H(p, q) = -\log q(x_{\text{true}})$。

### 概率与统计

**期望**：$E[X] = \sum_x x \cdot P(x)$

**方差**：$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$

**独立随机变量之和的方差**：$\text{Var}(\sum_i X_i) = \sum_i \text{Var}(X_i)$

### 导数与链式法则

**链式法则**：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$

**Softmax 的导数**：$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j)$
