---
title: 
date: 2026-08-14 17:52:19
excerpt:
tags:
categories:
  - 大模型
---
更换完实例之后发现，之前能测评，现在又测评不了了，说的是 VLLM 框架有问题。
之前我们用基座模型测评那个 CMB，就可以用 VLLM 来测，但是换完实例之后就不可以了。
然后之后测 Cxexam 就只能用 Transformers 来测。这样效率就比较慢。
然后现在是先做 SFT 微调。等到模型微调好了之后，然后再看看这个 VLLM 怎么弄。不然再用 Transformers 来测速度太慢了。

然后就是 SFT 微调的那个脚本，只要是有一个环境升级了或者变动了，它有可能会导致很多的东西都会变动，比如说 TRL这个版本比我预想还要更激进一点，一直在修参数，参数名字它就会更新，它更新一个版本，参数名字就会更新。你比如说 warmup ratio，它就改成了 warmup set steps。


之前 `--report-to swanlab` 只告诉 transformers "用 SwanLab"，但 transformers v5 的集成要求先存在一个活跃 Run。现在我在 `trainer.train()` 前手动调 `swanlab.init()`，读取你设的 `SWANLAB_PROJECT` 和 `SWANLAB_WORKSPACE`，这样：

- Run 先创建好
- transformers 后续记录指标就能挂到这个 Run 上
- 数据会同步到网页端 `ookkiou/med-rlhf`

这次应该能真正进入训练循环出 loss 了。如果 SwanLab 初始化或上传在国内网络卡住，把日志贴给我，我们可以临时切 `--report-to none` 先把训练跑通。

在经历了种种环境修改，然后改代码，之后这个脚本终于跑起来了。但是训练时长可能在我预期之外，它要训练 14.5 小时。因为我 batch size 只设了 1，然后数据量 45000，有两个 epoch，总步数达到了 5625，预估耗时 14.5 小时。所以这个时长可能有点长，所以我改了一下组合，批量加大，数据量适量的减少。你比如说我 batch size 改到了，增加到了 4，然后数据量从 45000 变成了 3 万，3 万条已经能注入足够的领域知识了，不会有太明显的损失效果。epoch 还是保留了 2，SFT 数据多过一遍的话，数据更稳。

训练完成之后，还需要跑合并。
```
cd ~/autodl-tmp/med-rlhf

python3 - <<'PYEOF'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_path = "out/sft_medical_v2"
merged_path = "out/sft_medical_v2_merged"

print("加载基础模型 (bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

print("加载 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("合并 LoRA 权重...")
model = model.merge_and_unload()

print(f"保存: {merged_path}")
model.save_pretrained(merged_path, safe_serialization=True)
tokenizer.save_pretrained(merged_path)
print("合并完成!")
PYEOF
```


训练完成之后，它比基模甚至还要更垃圾。它在第一个测评集上的正确率是 25%。在第二个测评集上的正确率是 27%。那么一开始我以为是那个 Lora 的权重它合并有问题，因为说这个伽马 3 它是一个多模态模型，有可能是给它合并到了视觉上。结果查它确实是挂在语言模型上，那么就继续考虑是不是过拟合。然后看日志发现，这个模型它的选择大多数都集中在 A 上面。
模型预测分布: A: 329 (65.8%) B: 90 (18.0%) C: 26 (5.2%) D: 23 (4.6%) E: 32 (6.4%) 
正确答案分布: A: 94 (18.8%) B: 106 (21.2%) C: 113 (22.6%) D: 99 (19.8%) E: 88 (17.6%)

结果训练出来过拟合了，模型基本上都会选 A。所有语言模型对选择题的第一个选项都有微弱偏好——因为 A 是最先看到的选项，注意力权重稍高。但基座模型的推理能力足够强，能克服这个偏向，根据题目内容选出正确答案。这个 Lora 破坏了它模型原来的思考能力，所以我现在需要重新训练。r=64 + lr=2e-4 + 2 epochs 改了太多参数，模型的内部表示空间被过度压缩（representation collapse）。通俗说就是模型"变笨了"——不再理解题目在问什么，无法通过推理区分哪个选项对。
![](Snapzy_2026-08-14_18-48-36_008.png)

之后更换了训练集

之后再进行 DPO

也换了新的评测集