# Vedaz Astrologer Fine-tune

Fine-tunes Qwen2.5-1.5B-Instruct with LoRA on Vedaz astrologer chat data.
Runs on **WSL2 (Ubuntu) inside Windows**, using VS Code connected to WSL. Native Windows was tried
first but hit unfixable `bitsandbytes`/CUDA crashes (see "Why WSL" below) — WSL is the supported
path for this project.

---

## One-time machine setup (WSL2 + Python 3.12)

Run these in a normal Windows PowerShell/CMD (not inside a venv):

```powershell
wsl --install -d Ubuntu
```

Restart if prompted, then finish Ubuntu's first-run setup (create a Linux username/password).

Open the Ubuntu terminal (or connect VS Code: bottom-left `><` icon -> "Connect to WSL"), then:

```bash
# System Python in WSL is often too new (e.g. 3.14) and lacks torch/bitsandbytes wheels.
# Install Python 3.12 specifically via the deadsnakes PPA:
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

python3.12 --version   # should print Python 3.12.x
```

**Important — use your Linux home directory, not `/mnt/c/...` or `/mnt/d/...`.**
Windows-mounted drives inside WSL don't support the file permissions/symlinks that
`venv` needs, and are much slower for I/O. Copy the project into WSL's native filesystem:

```bash
cp -r /mnt/d/vedaz-astrologer-finetune ~/vedaz-astrologer-finetune
cd ~/vedaz-astrologer-finetune
```

From here on, always work from `~/vedaz-astrologer-finetune`, and open this path
(not the D: drive path) as your VS Code folder when connected to WSL.

## Verify GPU passthrough

```bash
nvidia-smi
```

Should show your GPU (confirmed working: RTX 4050, 6GB VRAM). If this fails, WSL's
NVIDIA GPU support needs a driver-level fix on the Windows side (rare on modern Windows 11).

## Create the virtual environment

```bash
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate
python --version   # should print Python 3.12.x
```

You'll need to run `source .venv-wsl/bin/activate` every time you open a new terminal.

## Install dependencies

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl datasets accelerate bitsandbytes sentencepiece
```

Confirm GPU is visible to PyTorch:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Should print `True`.

Exact versions confirmed working on this project (see `requirements.txt`):
torch 2.6.0+cu124, transformers 5.12.1, peft 0.19.1, trl 1.7.0, datasets 5.0.0,
accelerate 1.14.0, bitsandbytes 0.49.2, sentencepiece 0.2.1.

---

## Files needed before you start

1. Original dataset -> `data/raw/Chat_Data_for_assessment_of_applicants.json`
2. New 5 chats -> `data/new_chats/vedaz_5_chats.jsonl`

(Note: folder is `Scripts/` with a capital S in this project — Linux is
case-sensitive, so commands below match that exactly.)

## Run order

```bash
python Scripts/prepare_data.py      # merges + cleans data -> data/processed/train_dataset.jsonl
python Scripts/train_lora.py        # trains LoRA adapter -> outputs/lora_adapter/  (~15-20 min for 10 epochs)
python Scripts/inference_test.py    # tests the model -> outputs/sample_outputs.txt
```

## What each step produces (deliverables)

- `outputs/lora_adapter/` — the trained LoRA adapter (small files, this is the actual model artifact)
- `outputs/logs/` — training checkpoints + `trainer_state.json` (loss history, proof it trained).
  Only the last 2 checkpoints are kept (`save_total_limit=2`) to avoid clutter.
- `outputs/sample_outputs.txt` — example responses from the fine-tuned model for 4 test prompts

---

## Why WSL instead of native Windows

Native Windows (VS Code + PowerShell venv) hit a hard, inconsistent crash
(`0xC0000005` access violation, no Python traceback) tied to `bitsandbytes`/CUDA
kernel loading on Windows — a known fragile combination. WSL2 runs the same
libraries on Linux, which is the platform they're actually developed and tested
against, and resolved the crash completely with no code changes needed.

## Training result (first run, 3 epochs, before the epoch bump)

- Loss: 2.377 -> 2.136 over 3 epochs (24 steps, ~5 minutes on RTX 4050)
- `mean_token_accuracy`: 0.48 -> 0.54
- Pipeline confirmed working end-to-end (adapter saves, loads, generates)
- Response quality was inconsistent (generic "as an AI language model" refusals
  leaking through instead of the trained astrologer persona) — expected with
  only 60 examples over 3 epochs on a 1.5B model. Epochs bumped to 10 to
  strengthen persona adherence; rerun and compare `sample_outputs.txt` before
  finalizing for submission.