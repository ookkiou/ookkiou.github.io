---
title: SFT训练阶段
date: 2026-08-010 17:36:05
excerpt: SFT训练阶段都干了什么
tags: [大模型,SFT,minimind]
categories: 
- 大模型
- minimind


---
## SFT阶段和预训练的区别
SFT 训练流程和预训练几乎一模一样。训练循环、梯度累积、cosine lr、混合精度、保存逻辑全部相同。唯一本质区别在 数据集的 labels 构造 （只学 assistant 回答段）和 初始权重 （从 pretrain 接着训）。

数据集的差别：
![](Snapzy_2026-08-10_17-02-46_220.png)五个关键参数的差异：![](Snapzy_2026-08-10_17-03-02_930.png)lr 差 50 倍是最关键的认知 ：SFT 是微调，不是从头学，用大 lr 会把预训练学到的语言能力冲掉。

## labels
在我们的预训练阶段，当时提过是把Padding设成-100，这样我们就不想让模型学习到关于Padding的事情,就不会在这一方面参与梯度计算。
在SFT阶段呢，我们更进一步只保留 assistant 回答段，其余全部设 -100 （包括 system、user、以及 chat template 的格式标记）。我们只想让模型回答助手回答的问题，不需要让它去提问或者去干什么其他的事情
```
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)          # ① 先全部设 -100（都不学）
    i = 0
    while i < len(input_ids):
        if input_ids[i:i+len(self.bos_id)] == self.bos_id:   # ② 找到 assistant 开头
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end+len(self.eos_id)] == self.eos_id:  # ③ 找到结尾
                    break
                end += 1
            for j in range(start, min(end+len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]       # ④ 只把回答段设成真实 token（要学）
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```
### 举例子
假设多轮对话经过 chat_template 拼接后：
```
[system] 你是助手 [user] 你好 [assistant] 你好，我是AI [user] 今天天气 [assistant] 今天晴天
```
tokenize 后 labels 是这样：
```
位置:    system段    user段    assistant回答段    user段    assistant回答段
labels:  -100...    -100...   你好，我是AI<eos>   -100...   今天晴天<eos>
				                    ↑保留                      ↑保留
```
每个 assistant 回答段都保留 ，user 和 system 段全是 -100。
### labels 和 inputs_id
input_ids 和 labels 是同一序列的两个视角——前者是"模型看到什么"，后者是"模型学什么"。先 tokenize 得到 input_ids，再在其上标记哪些位置参与学习（设 labels）。-100 就是"看但别学"的标记。
① tokenize 文本 → input_ids → ② 基于 input_ids 构造 labels 。
模型用 input_ids 看前面的序列做预测；算 loss 时对照 labels ，凡是 labels 里是 -100 的位置（pad、user 段等）就 跳过不算 loss、不产生梯度 ，只有真实 token 位置才算 loss 并反向传播更新参数。


## 训练日志
```
cd ~/autodl-tmp/minimind/trainer

python train_full_sft.py \
  --from_weight pretrain \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --epochs 1 \ 由于训练时间过长，这里 epochs 我改成了 1。
  --max_seq_len 768 \
  --device cuda:0
```
![](Snapzy_2026-08-10_19-43-37_727.png)
训练之后可以发现，相比预训练阶段，模型可以回答问题了，并不是单纯的词语接龙。
![](Snapzy_2026-08-10_19-55-36_921.png)