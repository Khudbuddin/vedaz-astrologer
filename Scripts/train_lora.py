"""
train_lora.py

Fine-tunes Qwen2.5-1.5B-Instruct with LoRA on the Vedaz astrologer chat dataset.

Run: python Scripts/train_lora.py
Output: outputs/lora_adapter/  (the small trained adapter files)
        outputs/logs/          (training logs / loss history)
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Using project root: {PROJECT_ROOT}")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = str(PROJECT_ROOT / "data" / "processed" / "train_dataset.jsonl")
ADAPTER_OUT = str(PROJECT_ROOT / "outputs" / "lora_adapter")
LOG_OUT = str(PROJECT_ROOT / "outputs" / "logs")

os.makedirs(ADAPTER_OUT, exist_ok=True)
os.makedirs(LOG_OUT, exist_ok=True)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Loaded {len(dataset)} training examples")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}


dataset = dataset.map(format_example)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir=LOG_OUT,
    num_train_epochs=10,          # bumped from 3 -> 10 for stronger persona learning on a small dataset
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,           # keep only the last 2 checkpoints, avoid clutter
    bf16=True,
    max_length=1024,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(ADAPTER_OUT)
tokenizer.save_pretrained(ADAPTER_OUT)
print(f"Adapter saved to {ADAPTER_OUT}")