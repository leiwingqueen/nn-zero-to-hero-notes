# 从零构建 GPT

## 视频链接


## 核心概念


## 关键代码 / 实现要点

### BigramLanguageModel（最简单的语言模型基线）

图解见 [bigram-diagram.html](./bigram-diagram.html)。

- 核心结构：`nn.Embedding(vocab_size, vocab_size)`，权重矩阵形状是 `(vocab_size, vocab_size)`。
  - 没有隐藏层、没有注意力——输入 token 的 id **直接索引出一行**，这一行就是预测下一个 token 的 logits。
- `forward(idx, targets=None)`：
  - `idx` 形状 `(B, T)`，查表后得到 `logits` 形状 `(B, T, C)`，`C = vocab_size`。
  - `targets` 为 `None` 时（推理/生成阶段）：`loss = None`，只返回 logits。
  - `targets` 给出时（训练阶段）：把 `logits.view(B*T, C)`、`targets.view(B*T)` 拍平后传入 `F.cross_entropy` 算出标量 loss。
- `generate(idx, max_new_tokens)`：自回归采样循环，每次迭代：
  1. `logits, _ = self(idx)` 重新跑一遍前向传播 → `(B, T, C)`
  2. `logits[:, -1, :]` 只取最后一个时间步 → `(B, C)`
  3. `F.softmax(logits, dim=-1)` 转成概率分布 → `(B, C)`
  4. `torch.multinomial(probs, num_samples=1)` **按概率随机采样**（不是 argmax）→ `(B, 1)`
  5. `torch.cat((idx, idx_next), dim=1)` 拼接到序列末尾 → `(B, T+1)`，重复直到生成满 `max_new_tokens`
- 关键局限：每一步预测只依赖上一个字符，没有更长的上下文窗口——这正是后续引入 self-attention 要解决的问题。

## 疑问 & 待深入

