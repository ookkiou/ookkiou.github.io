---
title: SwanLab
date: 2026-08-12 16:33:42
excerpt: 介绍SwanLab
tags:
  - 大模型
categories:
  - 大模型
---
SwanLab训练过程的实时仪表盘。你训练时产生的所有数字（loss、学习率、reward、KL……）它都帮你记录，画成曲线图，在网页上实时看。当然国外也有wandb。SwanLab 的用法和 wandb 几乎一模一样，但服务器在国内，AutoDL 连起来稳定得多。
MiniMind 项目里面，本身就是自己写训练循环，在代码里面已经有了接入 SwanLab 的代码。我们只需要在训练的时候加上参数即可。
```
# ========== 4. 配wandb ==========

wandb = None

if args.use_wandb and is_main_process():

import swanlab as wandb

wandb_id = ckp_data.get('wandb_id') if ckp_data else None

resume = 'must' if wandb_id else None

wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"

wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
```

如果模型用 HuggingFace Trainer，就不像 MiniMind 一样自己手敲代码的话。配置就用下面的。
```
import swanlab
from swanlab.integration.huggingface import SwanLabCallback   # 内置适配，不用自己写

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[SwanLabCallback(
        project="Gemma-Finetune",
        experiment_name="sft-med",
        config=vars(args),          # ← 命令行所有参数自动转成 dict
    )],
)
trainer.train()
```
## 使用方案
下载安装并登录：
```
pip install swanlab
swanlab login # 按提示注册/登录拿 API key
```
进行 Lora 训练，就在参数最后加一个--use_wandb即可：
```
python train_lora.py --from_weight full_sft --lora_name lora_medical_v2 --data_path ../dataset/lora_medical.jsonl --epochs 2 --batch_size 32 --learning_rate 5e-5 --device cuda:0 --use_wandb
```
之后按照提示即可在网页端看到可视化的数据：
![](Snapzy_2026-08-12_16-39-42_948.png)
