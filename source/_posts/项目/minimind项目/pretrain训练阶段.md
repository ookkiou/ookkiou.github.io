---
title: pretrain训练阶段
date: 2026-08-010 16:35:05
excerpt: 预训练阶段都干了什么
tags: [大模型,预训练,minimind]
categories: 
- 大模型
- minimind


---

## 训练流程
训练的文件：![](Snapzy_2026-08-10_13-37-19_263.png)

产物：out/pretrain_768.pth

通过pretrain训练出来的是一个只会接龙，按上文生成下文的模型，不具备回答问题的能力，之后还需要进行SFT才可以。
为什么他只会接龙，原因就在 lm_dataset.py:53-54 ：
```
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```
预训练的 labels = input_ids ， 整段文本每个位置都参与学习 ，模型学的是"接龙"而不是"回答"。
要让它听懂指令、会对话，是阶段 2（SFT）的事——那时 labels 只保留 assistant 回答段（ generate_labels ），模型才学会"听到问题→给出回答"的格式。
### 完整一次更新
```
1. 从训练集取 32 条句子 → 拼成 [bos]+tokens+[eos]+[pad] 矩阵
2. 整个 batch 一次前向 → 每个位置输出 6400 维概率分布
3. 和 shift 后的标签算交叉熵（pad 位置 -100 跳过）→ loss
4. loss ÷ 8 → 反向传播累积梯度    ← 重复 8 次
5. 第 8 次累积完 → 裁剪梯度 → AdamW 更新参数 → 清零梯度
6. 回到第 1 步，取下一个 batch
```
训练阶段一次并行，同时算出整个序列的loss
#### 为什么要8次batch之后更新权重？
想要的效果:  batch=256 一次算 → grad = 平均(256个样本梯度)
实际的做法:  batch=32 跑8次  → grad = Σ(每次32个梯度)/8 = 平均(256个梯度)
显存占用:    只有32的量        ✅

实际上 batch 越大，学习率更新的也就越稳定。但是我们并没有那么多的内存，所以我们只能让 batch 小一点，然后让它跑 8 次，这样来计算权重的时候呢，它跟 256 的结果基本上是差不多的。

## eval_llm.py
加载训练好的权重，做推理生成，验证模型学得怎么样。

- 加载权重 init_model L12-30 ：根据 --load_from 路径判断走原生 .pth 还是 transformers 格式
- 构造输入 L73-76 ：预训练权重直接 bos（begin of sequece） + 文本 （续写），SFT 权重套对话模板
- 流式生成 L82-87 ：调模型的 generate ，用 TextStreamer 边生成边打印

padding token id = 0、bos=1、eos=2
### 为什么padding的labels被设置成-100
```
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```
在计算loss时，凡是 target（labels）等于 -100 的位置，直接跳过 ，不参与 loss 计算，也不参与梯度。这是写死的逻辑。我们不想让模型计算padding的loss，就设置了一个不会和真实tokenid（大于等于0）相同的数字-100，约定俗成。
只有真实 token 之间的接龙关系参与学习，padding 这个"假数据"不会污染 loss，也不会产生梯度去误导模型参数。如果不忽略，padding（token id=0）会变成一个"真实目标"，模型会努力学"bos+你好+eos 之后应该接 0、0"，这是噪声，会拉低 loss 的信号质量。设成 -100 就让模型对这些位置"视而不见"。

在 SFT 里， prompt 段 （用户问的问题）也会被设成 -100——因为 SFT 只想学"怎么回答"，不想学"怎么提问"。机制完全一样，只是 -100 的应用范围从 padding 扩大到"非回答段"。
## 实际训练日志 
```
cd ~/autodl-tmp/minimind/trainer

python train_pretrain.py \
  --epochs 2 \
  --batch_size 32 \
  --accumulation_steps 8 \
  --max_seq_len 340 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --device cuda:0 \
  --save_dir ../out \
  --data_path ../dataset/pretrain_t2t_mini.jsonl
```
![](Snapzy_2026-08-10_13-49-14_058.png)
总损失loss=logits_loss + aux_loss。我们用的是dense模型，没有用moe，所以aux_loss=0
- logits_loss （纯语言模型损失） 就是预测下一个 token 的 交叉熵 。模型在每个位置输出一个 6400 维概率分布，和真实下一个 token 算交叉熵（ model_minimind.py:252 ）。这是模型"学语言"的 真正指标 ——越低说明预测越准。
- aux_loss （MoE 辅助损失） 只在 MoE 架构 下才有意义。MoE 有多个"专家"网络，路由器把 token 分给不同专家。如果不加约束，路由器可能把所有 token 都塞给同一个专家，其他专家闲着。 aux_loss 惩罚这种 负载不均 ，强迫路由器把 token 均匀分配（ model_minimind.py:171-173 ）。


每1000步覆盖一次out/pretrain_768.pth
开个新终端验证一下中途的成果
![](Snapzy_2026-08-10_13-50-40_018.png)现在的效果不是很好
三小时之后终于训练出来了
![](Snapzy_2026-08-10_16-48-01_854.png)