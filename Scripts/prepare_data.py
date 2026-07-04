"""
prepare_data.py

Uses ONLY the original 55-example dataset (no additional chats mixed in).

Input:
  - data/raw/Chat_Data_for_assessment_of_applicants.json (concatenated JSON objects,
    not a valid array/JSONL - handled by load_concatenated_json_objects below)

Output:
  - data/processed/train_dataset.jsonl

Run: python Scripts/prepare_data.py
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = str(PROJECT_ROOT / "data" / "raw" / "Chat_Data_for_assessment_of_applicants.json")
OUT_PATH = str(PROJECT_ROOT / "data" / "processed" / "train_dataset.jsonl")


def load_concatenated_json_objects(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    idx, n = 0, len(content)
    records = []
    while idx < n:
        while idx < n and content[idx] in " \n\t\r,":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(content, idx)
        records.append(obj)
        idx = end
    return records


def validate(record):
    if "messages" not in record:
        raise ValueError("Missing 'messages' key in a record")
    roles = [m.get("role") for m in record["messages"]]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"Record missing user/assistant turn: {roles}")


def main():
    records = load_concatenated_json_objects(RAW_PATH)
    for r in records:
        validate(r)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total examples: {len(records)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()