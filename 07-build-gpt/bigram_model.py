"""
BigramLanguageModel —— 从零构建 GPT 的最简单基线模型

只用 nn.Embedding(vocab_size, vocab_size) 直接查表预测下一个字符，
没有注意力、没有隐藏层。包含数据准备、训练、验证 loss 评估、生成测试的完整流程。
"""

import os
import urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------ 超参数 ------------------------------
batch_size = 32       # 每个 batch 并行处理的序列数量 B
block_size = 8         # 每个序列的长度（上下文窗口）T
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# ------------------------------ 数据准备 ------------------------------
data_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(data_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符级 tokenizer：字符 <-> 整数 id 的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 训练/验证集划分（前 90% 训练，后 10% 验证）
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # TODO: 随机采样一个 batch
    # 1. 根据 split 选择 train_data 或 val_data
    # 2. 用 torch.randint 随机选出 batch_size 个起始位置 ix（注意上界要减去 block_size，防止越界）
    # 3. x 是从每个起始位置取长度为 block_size 的片段，堆叠成 (batch_size, block_size)
    # 4. y 是 x 整体往右移一位的片段（即下一个字符），同样堆叠成 (batch_size, block_size)
    # 5. 把 x, y 搬到 device 上并返回

    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model):
    # TODO: 对 train / val 各采样 eval_iters 个 batch，计算平均 loss
    # 1. model.eval() 切换到评估模式
    # 2. 对 'train' 和 'val' 两个 split，分别循环 eval_iters 次：
    #    调用 get_batch(split) 拿到 X, Y，跑一次前向拿到 loss，记录下来
    # 3. 对每个 split 记录的 loss 求平均，存进一个 dict（key 是 'train'/'val'）
    # 4. model.train() 切回训练模式，返回这个 dict
    raise NotImplementedError


# ------------------------------ 模型定义 ------------------------------
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # TODO: 定义一个 nn.Embedding(vocab_size, vocab_size)
        # 每个 token 直接通过查表得到下一个 token 的 logits（没有隐藏层）
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # TODO:
        # 1. 用 self.token_embedding_table(idx) 得到 logits，形状 (B,T,C)
        # 2. 如果 targets 为 None，loss 设为 None
        # 3. 否则，把 logits reshape 成 (B*T, C)，targets reshape 成 (B*T,)，
        #    用 F.cross_entropy 计算 loss
        # 4. 返回 (logits, loss)
        raise NotImplementedError

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # TODO: 循环 max_new_tokens 次，每次：
        # 1. 调用 self(idx) 得到 logits（忽略 loss）
        # 2. 只取最后一个时间步的 logits，形状变成 (B, C)
        # 3. 用 F.softmax 得到概率分布
        # 4. 用 torch.multinomial 从概率分布中采样出下一个 token（形状 (B, 1)）
        # 5. 把采样出的 token 拼接到 idx 后面（沿时间维度），继续下一轮
        # 6. 循环结束后返回 idx
        raise NotImplementedError


# ------------------------------ 训练 ------------------------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# TODO: 写训练循环，共 max_iters 步
# 1. 每隔 eval_interval 步（或最后一步），调用 estimate_loss(m) 打印 train/val loss
# 2. 每一步：用 get_batch('train') 采样一个 batch
# 3. 前向计算 logits, loss = m(xb, yb)
# 4. optimizer.zero_grad(set_to_none=True) 清空梯度
# 5. loss.backward() 反向传播
# 6. optimizer.step() 更新参数

# ------------------------------ 测试 / 生成 ------------------------------
# 用一个空字符（id=0）作为起点，生成 500 个新字符，看看训练后的模型能"编"出什么
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
