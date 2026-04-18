"""从 trainer_state.json 生成 TensorBoard 日志，用于可视化已完成的训练"""
import json
import os
from torch.utils.tensorboard import SummaryWriter

# 找到最新的 checkpoint
checkpoints = sorted(
    [d for d in os.listdir("./sft_output") if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1])
)

if not checkpoints:
    print("没有找到 checkpoint 目录")
    exit(1)

latest = checkpoints[-1]
state_path = f"./sft_output/{latest}/trainer_state.json"
print(f"使用: {state_path}")

with open(state_path) as f:
    state = json.load(f)

# 写入 TensorBoard 日志
writer = SummaryWriter(log_dir="./sft_output/logs")

for entry in state["log_history"]:
    step = entry["step"]
    if "loss" in entry:
        writer.add_scalar("train/loss", entry["loss"], step)
    if "mean_token_accuracy" in entry:
        writer.add_scalar("train/mean_token_accuracy", entry["mean_token_accuracy"], step)
    if "entropy" in entry:
        writer.add_scalar("train/entropy", entry["entropy"], step)
    if "learning_rate" in entry:
        writer.add_scalar("train/learning_rate", entry["learning_rate"], step)
    if "grad_norm" in entry:
        writer.add_scalar("train/grad_norm", entry["grad_norm"], step)

writer.close()
print(f"已生成 TensorBoard 日志到 ./sft_output/logs/")
print("刷新 http://localhost:6006 即可查看")
