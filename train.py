from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import torch

# Set device (Mac 使用 MPS 加速)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load dataset
from datasets import load_dataset
dataset_dict = load_dataset("json", data_files={"train": "keywords_data_train.jsonl", "test": "keywords_data_test.jsonl"})
def map_func(example):
    conversation = example["conversation"]
    messages = []
    for item in conversation:
        messages.append({"role": "user", "content": item["human"]})
        messages.append({"role": "assistant", "content": item["assistant"]})
    return {"messages": messages}

dataset_dict = dataset_dict.map(map_func, batched=False, remove_columns=["conversation_id", "category", "conversation", "dataset"])

# Configure model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# Qwen3-0.6B 自带 chat template，SFTTrainer 会自动应用，无需手动 setup_chat_format

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    max_length=1024,                          # 进一步缩短，降低显存
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100, 
    eval_strategy="steps",
    # eval_strategy="no",                      # 先关闭 eval，跑通后再开
    dataloader_pin_memory=False,             # MPS 不支持 pin_memory
    logging_dir="./sft_output/logs",          # TensorBoard 日志目录
    report_to="tensorboard",                  # 启用 TensorBoard
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
)

dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))
print(batch)
batch["input_ids"]

decode_input_first_item = tokenizer.decode(batch["input_ids"][0])
print(decode_input_first_item)

# Start training
trainer.train()
