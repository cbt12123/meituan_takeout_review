# 美团评论智能分析助手

## 1. 项目背景

现在的打工人，如果公司没有食堂，不想下楼跑，也不想做饭带饭，那必然会在美团，饿了么，京东这黄蓝红三色中进行选择。但各行各业遍地开花，新店新品层出不穷，平台无法做到精确的把控每一道菜品的质量，每一单外卖的质量。由此就会产生冲突。 

见过不少外卖员因配送问题背差评的情况，也见到过一些曝光的不良商家。有些时候我们义愤填膺，在评论区慷慨激昂，诉说自己的不满，同时尝试给其他客户示警避雷；更多的时候，我们是那个看到4.8+评分，掉入陷阱的新客。随着我和一些商家的沟通，我发现有些时候，并不是商家不理会，反而是一个让我诧异的情况：**商家不知道菜出现了问题**。 

我们有时遇到大问题，如漏餐等，第一反应是发消息而不是打电话，有时候商家在后厨忙前忙后，我们一个电话过去，商家耳边大火爆炒，听不清我们说话；我们在电话另一边听着锅碗翻飞，也失去了沟通的欲望。于是开始尝试联系商家，发信息。但是这个时候，商家电话都不一定接的过来，根本很难第一时间回消息，顾客看着未读陷入了沉思... 

有好一点的商家，设置了智能回复。但智能回复似乎只是一些机械的提取关键词，问外卖就回复努力配送，问其他就抱歉，夸好吃...似乎也没什么回复，或者偶尔感谢一下认可？ 

正向的我们暂时不讨论，毕竟从我个人而言，消息区我是反馈问题的，很少会去表扬（除非老顾客了，下次点送个配菜），那么客户满意度就会在这种情况下不断被拉低。 

这个项目就是在这种场景下开始的，也是恰好被外卖气饱，在浏览一些数据集，恰好看到了"waimai_10k.csv"，就想着要不基于这个方向做点东西，一方面练手，一方面万一真有用呢？

## 2. 项目结构

```
.
├── README.md
├── data
│   ├── raw
│   │   ├── meituan.csv
│   │   └── waimai_10k.csv
│   ├── train-00000-of-00001.parquet
│   ├── train.jsonl
│   └── val.jsonl
├── data_processor.py
├── data_v2
│   ├── train.jsonl
│   └── val.jsonl
├── evaluate.py
├── infer.py
├── output_eval
│   └── evaluation_results_v1.0
│       ├── confusion_matrix.png
│       ├── core_metrics.png
│       ├── error_distribution.png
│       ├── raw_evaluation_results.csv
│       ├── success_evaluation_results.csv
│       └── validation_report_en.md
├── output_model
│   └── output_v1.0
│       ├── checkpoint-231
│       ├── checkpoint-264
│       │   ├── README.md
│       │   ├── adapter_config.json
│       │   ├── adapter_model.safetensors
│       │   ├── added_tokens.json
│       │   ├── chat_template.jinja
│       │   ├── merges.txt
│       │   ├── optimizer.pt
│       │   ├── rng_state.pth
│       │   ├── scaler.pt
│       │   ├── scheduler.pt
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer.json
│       │   ├── tokenizer_config.json
│       │   ├── trainer_state.json
│       │   ├── training_args.bin
│       │   └── vocab.json
│       ├── core_word_utils.py
│       ├── lora_weights
│       │   ├── README.md
│       │   ├── adapter_config.json
│       │   └── adapter_model.safetensors
│       └── tokenizer
│           ├── added_tokens.json
│           ├── chat_template.jinja
│           ├── merges.txt
│           ├── special_tokens_map.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           └── vocab.json
└── train.py
```

## 3. 环境依赖
```
python=3.10.8
pandans=2.3.1
torch=2.1.2+cu121
transformers=4.55.1
```

## 4. 快速开始

### 4.1 数据准备
在`./data/raw`里已准备好了`meituan.csv`，这个是自己扩充的，包括个人搜集+大模型生成+淘宝购买标注服务，可以按照格式扩充。 
在`./data/raw`里已准备好了`waimai_10k.csv`，这是原先的开源数据，可以尝试做一些处理。

准备好之后可以直接使用代码做处理：
```
python data_processor.py
```
### 4.2 模型训练

启动训练：
```
python train.py
```

### 4.3 模型测试

运行测试，生成测试报告
```
python evaluate.py
```

### 4.4 模型应用

运行代码，观察效果：
```
python infer.py
```
可以尝试简单修改，做其他输入测试

## 5. 实验结果
对训练的Lora进行验证，得到混淆矩阵如下： 

![图片描述](/output_eval/evaluation_results_v1.0/confusion_matrix.png)

由于数据原因，只准备了36个数据进行验证，其中正确的为35个，情感分类准确率达到了97.2%，达到较满意的结果。

核心评估指标的输出如下所示，大部分指标达到较高水平： 

![核心评估指标](/output_eval/evaluation_results_v1.0/core_metrics.png)

运行`infer.py`，测试结果如下：

![infer推理结果](/docs/infer.jpg)


## 6. 备注
- 如需要已训练好的模型验证发出的实验结果，请联系作者。
- Email：1318053467@qq.com