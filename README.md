# Vedaz Astrologer Fine-tune

Fine-tunes **Qwen2.5-3B-Instruct** with LoRA on Vedaz's original 55-example astrologer
chat dataset. Built and trained on WSL2 (Ubuntu inside Windows), using VS Code connected
to WSL, on an RTX 4050 (6GB VRAM).

## Model & training configuration
- Base model: **Qwen2.5-3B-Instruct** (larger than the 1.5B variant, more capacity to
  pick up stylistic/domain patterns from a small dataset)
- LoRA rank/alpha: **r=32, alpha=64** (higher rank than a smaller-model baseline run,
  giving the adapter more parameters to learn the astrologer persona)
- Epochs: **6**
- Dataset: the original **55 chats only** (`Chat_Data_for_assessment_of_applicants.json`) -
  no additional synthetic examples mixed in, to keep this run's results a clean read on
  what the provided data alone can teach the model
- Effective batch size: 8 (batch size 1 x gradient accumulation 8)
- Precision: bf16, gradient checkpointing enabled (fits 3B comfortably in 6GB VRAM)

## One-time machine setup (WSL2 + Python 3.12)

```powershell
wsl --install -d Ubuntu
```

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

Work from WSL's native filesystem, not `/mnt/c` or `/mnt/d` (Windows-mounted drives don't
support the permissions/symlinks `venv` needs):
```bash
cd ~
git clone <this-repo-url> vedaz-astrologer-finetune
cd vedaz-astrologer-finetune
```

```bash
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl datasets accelerate bitsandbytes sentencepiece
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Files needed
Put the original dataset at:
`data/raw/Chat_Data_for_assessment_of_applicants.json`

## Run order

```bash
python Scripts/prepare_data.py      # -> data/processed/train_dataset.jsonl (55 examples)
python Scripts/train_lora.py        # -> outputs/lora_adapter/
python Scripts/inference_test.py    # -> outputs/sample_outputs.txt
```

## Deliverables
- `outputs/lora_adapter/` - trained LoRA adapter
- `outputs/logs/` - training checkpoints + loss history
- `outputs/sample_outputs.txt` - generated responses to 4 test prompts

## Notes on this configuration's tradeoffs
Using a larger base model (3B vs 1.5B) with a higher LoRA rank gives the adapter more
capacity, at the cost of longer training time and more VRAM/compute per step. Fewer
epochs (6 vs a higher count on a smaller model) were used here partly to manage total
training time on a single consumer GPU, and because a larger base model with higher-rank
LoRA tends to need somewhat less repetition over the same data to pick up patterns
compared to a smaller model with a lower-rank adapter.