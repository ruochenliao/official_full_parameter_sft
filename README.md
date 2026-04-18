# Official Full Parameter SFT

基于 Qwen3-0.6B 的全参数监督微调（SFT）实战项目，使用 Hugging Face TRL 框架对大语言模型进行关键词提取任务的微调训练。

## 项目简介

本项目是一个学习示例，参考 [Hugging Face LLM Course - SFT](https://huggingface.co/learn/llm-course/chapter11/3)，实现了：

1. **数据准备**：将自定义对话数据集转换为 SFTTrainer 标准 messages 格式
2. **全参数微调**：使用 TRL 的 SFTTrainer 对 Qwen3-0.6B 进行全参数 SFT 训练
3. **模型推理**：加载微调后的模型进行关键词提取

## 项目结构

```
official_full_parameter_sft/
├── train.py                    # SFT 训练脚本
├── use_model.py                # 模型推理脚本
├── generate_tb_logs.py         # TensorBoard 日志生成工具
├── keywords_data_train.jsonl   # 训练数据（49,500 条）
├── keywords_data_test.jsonl    # 测试数据（500 条）
├── .gitignore
├── README.md
└── sft_output/                 # 训练输出（不纳入 Git）
    ├── checkpoint-xxx/         # 各步骤的模型快照
    │   ├── model.safetensors   # 微调后的模型权重
    │   ├── tokenizer.json      # 分词器
    │   ├── config.json         # 模型配置
    │   └── ...
    └── logs/                   # TensorBoard 日志
```

## 环境要求

- Python 3.11+
- Apple M 系列芯片（MPS 加速）或 NVIDIA GPU（CUDA）

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers trl datasets tensorboard
```

### 2. 训练模型

```bash
# Mac 用户建议加上环境变量，避免 MPS 内存溢出
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train.py
```

### 3. 使用模型

```bash
python use_model.py
```

### 4. 查看训练曲线（TensorBoard）

```bash
# 如果训练时未开启 TensorBoard，可从 trainer_state.json 生成日志
python generate_tb_logs.py

# 启动 TensorBoard
tensorboard --logdir ./sft_output/logs
# 浏览器打开 http://localhost:6006
```

## 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 基座模型 | Qwen/Qwen3-0.6B | 通义千问 6 亿参数模型 |
| batch_size | 1 | 受限于 Mac 内存 |
| gradient_accumulation | 4 | 等效 batch_size = 4 |
| learning_rate | 5e-5 | |
| max_length | 1024 | 最大序列长度（token） |

## 训练效果

| 指标 | 训练前 | 训练后 |
|---|---|---|
| Loss | 3.17 | ~2.31 |
| Token 准确率 | 42.4% | ~52.9% |

## 微调前后对比

**输入**：抽取出文本中的关键词（人工神经网络在猕猴桃种类识别上的应用）

| | 输出 |
|---|---|
| **微调前** | 关键词：人工神经网络、猕猴桃、种类识别、模式识别、特征参数、果品、介电特性、品质、研究方法 |
| **微调后** | 猕猴桃;人工神经网络;果品;品种;新鲜等级 |

微调后模型学会了用分号分隔的简洁关键词格式输出。

## 参考

- [Hugging Face LLM Course - Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/3)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
