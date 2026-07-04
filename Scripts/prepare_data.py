"""
prepare_data.py

Merges:
  - data/raw/Chat_Data_for_assessment_of_applicants.json  (the original 55 examples,
    stored as multiple concatenated JSON objects, NOT a clean array/JSONL)
  - data/new_chats/vedaz_5_chats.jsonl                     (your 5 new examples)

Outputs:
  - data/processed/train_dataset.jsonl   (one clean JSON object per line,
    each with a "messages" key - ready for the trainer)

Run: python scripts/prepare_data.py
"""

import json
import os

RAW_PATH = os.path.join("data", "raw", "Chat_Data_for_assessment_of_applicants.json")
NEW_CHATS_PATH = os.path.join("data", "new_chats", "vedaz_5_chats.jsonl")
OUT_PATH = os.path.join("data", "processed", "train_dataset.jsonl")


def load_concatenated_json_objects(path):
    """
    The original file is NOT a valid JSON array - it's multiple {...} objects
    concatenated with commas/newlines between them (no wrapping brackets).
    This walks through the raw text and pulls out each object one by one.
    """
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


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate(record, source):
    """Basic sanity check on each record before it goes into training data."""
    if "messages" not in record:
        raise ValueError(f"Missing 'messages' key in a record from {source}")
    roles = [m.get("role") for m in record["messages"]]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"Record from {source} missing user/assistant turn: {roles}")


def main():
    original = load_concatenated_json_objects(RAW_PATH)
    new_chats = load_jsonl(NEW_CHATS_PATH)

    all_records = original + new_chats
    for r in all_records:
        validate(r, "dataset")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Original examples: {len(original)}")
    print(f"New examples:      {len(new_chats)}")
    print(f"Total written:     {len(all_records)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
