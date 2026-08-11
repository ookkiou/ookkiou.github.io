---
title: LoRa
date: 2026-08-11 20:44:11
excerpt: 什么是 LoRA
tags:
  - 大模型
  - minimind
  - LoRA
categories:
  - 大模型
  - minimind
---
## lora 训练日志
![](Snapzy_2026-08-11_12-23-17_849.png)
![](Snapzy_2026-08-11_12-34-17_142.png)在医学领域上的回答：
![](Snapzy_2026-08-11_12-34-44_102.png)我惊讶的发现，在问他其他问题时，模型居然也扯到了医学领域。按理来说 lora 不会改变原基础模型，可能是过拟合了。模型把所有输入都当医学问题处理了，无论问什么，都往医学方向答。这就是过拟合到医学领域的典型表现。
为什么？
![](Snapzy_2026-08-11_12-37-04_998.png)小数据+多 epoch 导致过拟合了，我们重训调参：
```
python train_lora.py \
  --from_weight full_sft \
  --lora_name lora_medical \
  --data_path ../dataset/lora_medical.jsonl \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --device cuda:0
```
训练完之后可以看到回答基本正常：
![](Snapzy_2026-08-11_12-48-34_194.png)为什么效果不理想
① 模型能力上限 64M 参数的 minimind 本身知识储备有限，医学理解浅。它学到的是"医学问答的 格式和词汇 "，不是真正的医学知识。所以回答看起来像医学，但内容有错误（"血液进入血管"这种胡话）。
② 数据规模小 lora_medical.jsonl 数据量小，LoRA 容易记住数据分布而非泛化。即使 3 epoch，对小数据来说可能还是偏多。