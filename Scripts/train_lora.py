"""
train_lora.py

Fine-tunes Qwen2.5-3B-Instruct with QLoRA (4-bit base model + LoRA adapters)
on the original 55-example Vedaz astrologer chat dataset.

NOTE: A 3B model in plain bf16 (~6GB) doesn't reliably fit alongside training
overhead on a 6GB GPU -- `device_map="auto"` will silently offload some layers
to CPU/meta device, which breaks the backward pass for LoRA. Loading in 4-bit
shrinks the model to ~2GB, so it fits entirely on the GPU with no offloading.

Run: python Scripts/train_lora.py
Output: outputs/lora_adapter/  (the small trained adapter files)
        outputs/logs/          (training logs / loss history)
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Using project root: {PROJECT_ROOT}")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},   # force everything onto GPU 0, no auto-offload to CPU/meta
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# Same rank/alpha as the intended run: r=32, alpha=64
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir=LOG_OUT,
    num_train_epochs=6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
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