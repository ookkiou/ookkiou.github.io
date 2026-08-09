---
title: minimind-day1
date: 2026-08-08 18:43:00
excerpt: 基本代码内容和框架总览
tags: [大模型,minimind]
categories: 
  - 大模型
  - minimind
---

重点是四个模块：模型结构 model/model_minimind.py
数据 dataset/lm_dataset.py
训练循环 trainer/train_pretrain.py
工具函数 trainer/trainer_utils.py

## 模型是什么，我们在训练什么？
模型 = 架构（代码定义的结构） + 权重（一堆数字）

| 部分  | 是什么                                   | 在哪里                                                    |
| --- | ------------------------------------- | ------------------------------------------------------ |
| 架构  | 一堆数学公式和层叠结构（Attention、FFN、Embedding…） | model/model_minimind.py 这个 Python 类                    |
| 权重  | 公式里那些参数的具体数值（每个 Linear 层的矩阵）          | out/pretrain_768.pth 文件，或 minimind-3/model.safetensors |
只有架构没权重 → 模型存在但"啥都不会"（输出乱码）。 只有权重没架构 → 一堆数字，不知道怎么用。 两者合起来 → 能用的模型。

一开始只有架构，权重都是随机的。我们训练的就是权重。不断调整那些随机权重，让模型输出越来越像人话 。

## Transformer
输入 x（token 的特征）是"原始信息"，同一个 token 的特征 x，经过三个不同投影变成三种角色 ：

```
x  →  q_proj (×Wq)  →  xq   (query)
x  →  k_proj (×Wk)  →  xk   (key)
x  →  v_proj (×Wv)  →  xv   (value)
W是权重矩阵，就是训练出来的参数
```
- 投影成 q → 这个 token 作为"提问者"该问什么
- 投影成 k → 这个 token 作为"被查者"能被什么样的问题匹配
- 投影成 v → 这个 token 作为"信息源"能提供什么内容

投影之后：
1️⃣拆头，拆成8维的96个头，每个维度关注不同的方面。RoPE位置编码加上位置信息。拼kvcache：
```
xq = xq.view(bsz, seq_len, n_local_heads, head_dim)      # 拆成 8 个头
xk = xk.view(bsz, seq_len, n_local_kv_heads, head_dim)   # 拆成 4 个头
xq, xk = self.q_norm(xq), self.k_norm(xk)                # QK-Norm
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)          # 加位置信息
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)        # 拼历史 k
    xv = torch.cat([past_key_value[1], xv], dim=1)        # 拼历史 v
```
2️⃣算注意力：
```
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)   # q · k^T / √d
output = F.softmax(scores, dim=-1) @ xv    # 用权重对 v 加权求和
```
拿着 q（搜索词）去和每个 token 的 k（标签）做点积。点积越大 = 越匹配 = 这个 token 对当前 token 越重要。除以 √d 是为了数值稳定（防止点积过大导致 softmax 饱和）。然后 softmax 归一化成权重，再用这个权重对 v 加权求和。"我关心的 token 的内容(v) 被加权聚合过来了" ——这就是注意力机制的本质： 每个 token 根据自己的 q 去挑选相关 token 的 v，加权融合成新特征 。
3️⃣输出投影：8 个头的结果拼回 768 维，再经 o_proj 线性层整合，输出最终的注意力结果。
```
output = output.transpose(1, 2).reshape(bsz, seq_len, -1)   # 多头结果拼回去
output = self.resid_dropout(self.o_proj(output))             # 再过一个线性层
```
## 注意力机制
- **多头注意力（MHA, Multi-Head Attention）**：每个查询（Query）都有自己独立的键（Key）和值（Value）头。**优点**是质量最高、表达能力强；**缺点**是显存占用大、推理速度慢（尤其在长文本时）。
- **多查询注意力（MQA, Multi-Query Attention）**：所有查询头**共享**同一组键和值头（通常只有 1 个 KV 头）。**优点**是推理速度极快、显存占用极小；**缺点**是模型质量下降明显（因为共享导致表达能力受限）。
- **GQA（Grouped-Query Attention）**，KV 头数只有 Q 的一半，省显存。minimind使用的就是GQA



## kvcache
### kvcache
在大模型生成文本时，比如说生成了‘我爱吃’，模型下一步需要预测，只用根据‘吃’的概率预测即可。不需要‘我’和‘爱’的概率。
根据注意力计算公式，我们不需要‘我’和‘爱’的q，只需要‘吃’的q和整个k、v矩阵，即可计算出吃的下一token的概率。
有了kvcache，模型每次只用输入一个token，只需要算他自己的qkv，他之前的k和v都存在cache中待用，节省时间。
若没有，那么模型就要输入完整的‘我爱吃’，并且计算每个字的qkv。

### 训练阶段不用kvcache
#### 不划算
训练时：在训练阶段你手里有完整序列 [什么是爱是一种情感] ，一次性喂进去，所有 token 的 q/k/v 同时算出来、同时算注意力—— 天然并行 。用 flash attention（ L126 ）做大矩阵乘法，GPU 满载最快。这时用 cache 反而徒增内存开销，没意义。
推理时 ：用户问"什么是爱"，模型要 一个字一个字往外吐 ——先出"是"，再出"一"，再出"种"... 每生成一个字，新 token 的 q 都要去查所有历史 token 的 k/v。如果没 cache，每生成一个字都要把前面所有字重算一遍投影，O(n²) 复杂度，越长越慢。 有了 cache，每步只算 1 个新 token 的 k/v，O(n) 复杂度 。

#### 算法上的不可行：
- **推理**只有**前向传播**，存 KV 只是为了省计算。
    
- **训练**包含**前向 + 反向传播**。在反向传播时，梯度不仅要传给当前位置的 Q，**还要传给之前所有位置的 K 和 V**。
    

如果你使用了 KV Cache（即复用之前算好的 K 和 V），在反向传播时，梯度流会回溯到缓存中的 K 和 V。这意味着：

- **缓存中的 K/V 必须保留计算图（Computational Graph）**，否则梯度传不回去。
    
- 保留计算图 = 缓存的不再是“纯数值”，而是“带梯度钩子的张量”，显存占用直接翻好几倍。
    

最终，你为了“省前向计算”而缓存的 KV，在反向传播时反而要消耗更多显存来保存中间变量。**这笔账算下来是巨亏的**。



## Decoder-only架构

