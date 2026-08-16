---
title: Decoder-only
date: 2026-08-12 13:19:47
excerpt: 什么是Decoder-only
tags:
  - 大模型
  - Decoder-only
categories:
  - 大模型
---
现在主流大模型都是用的是Decoder-only架构。**从结构上说**，它只保留 Transformer 的 Decoder 部分，去掉了 Encoder。整个模型就是一堆相同的 Decoder Block 堆叠起来，每个 Block 包含自注意力层和前馈网络。
T5、BART 这类 Encoder-Decoder 模型，Encoder 负责把输入完整编码成一组向量表示，Decoder 再基于这组表示来生成输出，天然适合翻译、摘要这类"输入一段、输出一段"的任务。而 Decoder-only 把输入和输出放在同一个序列里处理，不做这种分离，架构更简单，也更容易 scale up。
## 两个阶段 
### Prefill 阶段

Prefill 处理的是 prompt 部分。用户发来的 prompt 比如 "请解释什么是KV Cache"，经过 tokenizer 变成一串 token ID，这些 token 全部是已知的、确定的。

因为所有输入 token 都已知，可以把它们一次性打包成一个 batch 送进模型。每一层做自注意力时，每个 token 的 Query 可以和序列中所有 token 的 Key 做点积，这是一个标准的矩阵乘法（Q × K^T），维度是 [seq_len, seq_len]。矩阵乘法是 GPU 最擅长的事，所以这一步的 GPU 算力利用率很高，属于**计算密集型（compute-bound）**。

Prefill 阶段还有一个重要任务：构建 KV Cache。每一层注意力算出的 Key 和 Value 矩阵会被保存下来，供后续 Decode 阶段复用。如果不保存，每生成一个新 token 都要把整个 prompt 重新算一遍，代价不可接受。

### Decode 阶段

Prefill 结束后，prompt 的 KV 都已经缓存好了。接下来进入逐 token 生成。

每生成一个新 token，流程是这样的：先拿到上一个 token 的 embedding，过每一层时，只需要用这一个 token 的 Query（维度 [1, head_dim]）去和 KV Cache 里所有历史 token 的 Key、Value（维度 [seq_len, head_dim]）做注意力。然后把新 token 的 K、V 追加进 Cache，再送到下一层。

这里的问题是：每次只处理 1 个 token，计算量很小，但每一步都要把整个 KV Cache 从显存读一遍。随着序列变长，Cache 越来越大，读取代价越来越高。瓶颈从"算力"转移到了"显存带宽"，所以这个阶段是**访存密集型（memory-bound）**。

### 两者的核心差异对比

Prefill 是"一次算很多 token，计算量大，GPU 忙得团团转"；Decode 是"一次算一个 token，计算量小，GPU 大部分时间在等数据从显存搬过来"。用一个比喻：Prefill 像工厂流水线全速运转，Decode 像快递员一趟一趟跑，每趟只送一个包裹。

### 这对实际系统意味着什么

正因为两个阶段特性差异这么大，现代推理框架会分别优化。比如 Prefill 和 Decode 可以跑在不同的 GPU 上（disaggregated serving，DistServe 的思路），或者用不同的 batch 策略——Prefill 倾向大 batch 充分利用算力，Decode 则需要 careful batching 来缓解访存瓶颈。KV Cache 的管理（PagedAttention 等）也主要是为 Decode 阶段服务的。

理解了 Prefill 和 Decode 的区别，后面遇到 chunked prefill、speculative decoding 这些更进阶的优化技术就很好理解了——本质上都是在想办法让 Decode 阶段不那么"闲"，或者让 Prefill 阶段不至于阻塞 Decode。


