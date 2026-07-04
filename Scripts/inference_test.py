"""
inference_test.py

Loads the base Qwen2.5-3B-Instruct model (4-bit, same quantization as training)
+ your trained LoRA adapter, then runs test prompts to prove the fine-tune worked.
Saves outputs to outputs/sample_outputs.txt for your submission.

Run: python Scripts/inference_test.py
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Using project root: {PROJECT_ROOT}")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = str(PROJECT_ROOT / "outputs" / "lora_adapter")
OUT_FILE = str(PROJECT_ROOT / "outputs" / "sample_outputs.txt")

TEST_PROMPTS = [
    "Mera career bahut slow chal raha hai, kya achhe din aayenge?",
    "Can you tell me if my father in the hospital will survive?",
    "Is Mercury retrograde messing with my texts right now lol",
    "Meri shaadi kab hogi? Bata do please.",
]

# Same 4-bit config as train_lora.py, so inference fits on the same 6GB GPU
# without relying on device_map="auto" offloading.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

results = []
for prompt in TEST_PROMPTS:
    messages = [
        {"role": "system", "content": "Aap Vedaz ke AI jyotishi hain."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    print(f"\nPROMPT: {prompt}\nRESPONSE: {response}\n{'-'*60}")
    results.append(f"PROMPT: {prompt}\nRESPONSE: {response}\n{'-'*60}")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\nSaved all outputs to {OUT_FILE}")