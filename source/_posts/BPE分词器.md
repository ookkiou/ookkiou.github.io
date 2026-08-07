---
title: BPE分词器
date: 2026-08-07 16:17:05
excerpt: 介绍BPE分词器
tags: [大模型,BPE]
categories: 大模型
---

cs336第一节课介绍了不同类型的分词器
效率的关键：字节数/token数，越高越好
character字符分割——按照Unicode分割，显得token太长，词汇量太长
byte字节分割——按照utf8，token被控制在0-255，token更长了
word词语分割——按照空格之类的分割，词汇量不稳定，可能出现没见过的token，有很多词语很罕见，对训练无用、训练中未见过的新词会被赋予一个特殊的UNK标记，这很不优雅，并且会搞乱困惑度的计算。

## BPE分割——字节对编码

先是处理special tokens，比如如果文本里面有ENDOFTEXT，得先切出来，不经过PBE直接查询id
然后我们要进行pre-tokenize，把文本切成"Hello"、","、" how"。  用的是GPT-2 的一个固定正则 pat，按照这个规则先切成大块tokens。这样一来可以提高泛用性。如果不pre，那么字母就会和后面的空格组合在一起，没有意义。把空格放在前面，可以分辨什么在开头，什么在中间。
之后把字符串转化成bytes，开始字节对编码，出现概率高的merge到一起，作为一个新的token，放到词表里面。然后我们一直重复这个过程。

token 是词表里的一个条目，有 id（整数编号）和 bytes（字节内容）两个属性
vocab: dict[int, bytes] 记录 id → bytes，inverse_vocab 是反向
encode 本质：文本 → bytes → 切成词表里的 token → 查 id
decode 本质：id → 查 bytes → 拼接 → UTF-8 解码

### 如何merge：在序列里找最高优先级的可合并对
在当前字节列表里，扫描所有相邻对，查 merge rank dict，找出rank 最小（优先级最高）的那个可合并对，合并它。
重复，直到没有任何相邻对在 merges 里。
优点：只处理当前序列里实际出现的 merge，高效。
缺点：每次合并后要重新扫描找最小 rank。

每次合并一个对后，受影响的只是合并位置附近，不需要重新扫描整个序列，只看局部即可。
假设序列是 [..., X, a, b, Y, ...]，你把相邻对 (a, b) 合并成 ab，序列变成 [..., X, ab, Y, ...]。请你确认受影响的相邻对是哪几个：

原来的 (X, a) 变成了 (X, ab) —— 新相邻对
原来的 (b, Y) 变成了 (ab, Y) —— 新相邻对
中间的 (a, b) 消失了
所以合并一次后，只有 2 个新的相邻对需要重新查 merge rank dict，不用全扫描。你的直觉完全对。


### encode：
"hello" 
  → UTF-8 编码 → b"hello" 
  → 按 BPE 规则切成字节片段 [b'he', b'l', b'l', b'o']   # 这些片段都必须是词表里已有的 token
  → 每个片段查反向 vocab(bytes→id) → [256, 108, 108, 111]


输入: text (str)
  │
  ├─ 1. 如果有 special_tokens，先按 special tokens 切分文本（长者优先匹配）
  │     → 得到若干片段，标记每段是 special 还是非 special
  │
  ├─ 2. 对每个片段:
  │     ├─ special 片段: 直接查反向 vocab(bytes→id)，不经过 BPE
  │     └─ 非 special 片段:
  │          ├─ 2a. pre-tokenize（GPT-2 正则）切成若干 pre-token
  │          ├─ 2b. 每个 pre-token: str → UTF-8 bytes → 拆成单字节列表
  │          ├─ 2c. 对单字节列表做 BPE 合并（思路 B：找最高优先级可合并对，重复直到不能合并）
  │          └─ 2d. 合并后的每个字节片段查反向 vocab 得 id
  │
  └─ 3. 按顺序拼接所有 id → list[int]

### decode：
[256, 108, 108, 111]
  → 每个 id 查正向 vocab(id→bytes) → [b'he', b'l', b'l', b'o']
  → 拼接所有 bytes → b"hello"
  → UTF-8 解码 → "hello"

一定要先拼接：比如 😃 可能会拆分成3个token[333，334，335]。查表之后是[b'xxx',b'aaa',b''ccc]
然后拼接得到b"xxxaaaccc",然后转化成😃
如果你先解码再拼接的话，那就只是单纯的得到三个分开的东西：xxx，aaa，ccc

### id是怎么得到的？
id 是从预先训练好的词表里查的，不是现创造的
词表在训练阶段就固定下来了，使用阶段（encode）只是查表。

两个阶段对比：

|        | 训练阶段                         | 使用阶段             |
| ------ | ---------------------------- | ---------------- |
| 对应函数   | run_train_bpe（你后面要写的）        | encode（你现在在写的）   |
| 做什么    | 从语料统计字节对频率，反复合并，构建词表和 merges | 用已有的词表把文本查表转成 id |
| 创造 id？ | 是，每次合并生成新 token，分配新 id       | 否，只能查已有的 vocab   |

所以一个 AI 模型有一个固定的词表（比如 GPT-2 是 50257 个 token），在训练模型之前就用 BPE 训练好了。之后所有输入文本都映射到这些已有的 id 上，encode 不会创造任何新 id。



### run train BPE
