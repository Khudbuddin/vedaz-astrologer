"""
inference_test.py

Loads the base Qwen2.5-1.5B-Instruct model + your trained LoRA adapter,
then runs a few test prompts to prove the fine-tune worked.
Saves outputs to outputs/sample_outputs.txt for your submission.

Run: python scripts/inference_test.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join("outputs", "lora_adapter")
OUT_FILE = os.path.join("outputs", "sample_outputs.txt")

TEST_PROMPTS = [
    "Mera career bahut slow chal raha hai, kya achhe din aayenge?",
    "Can you tell me if my father in the hospital will survive?",
    "Is Mercury retrograde messing with my texts right now lol",
    "Meri shaadi kab hogi? Bata do please.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # load from base model (has full tokenizer files)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
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

os.makedirs("outputs", exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\nSaved all outputs to {OUT_FILE}")