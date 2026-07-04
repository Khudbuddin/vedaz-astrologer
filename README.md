# Vedaz Astrologer Fine-tune

This project fine-tunes **Qwen2.5-3B-Instruct** with **QLoRA** (4-bit quantized base
model + LoRA adapters) on Vedaz's 55-example astrologer chat dataset. It runs on
**WSL2 (Ubuntu on Windows)**, developed in VS Code connected to WSL, on an
**RTX 4050 (6GB VRAM)**.

The project has three stages, each a standalone script:
1. `Scripts/prepare_data.py` — parses and validates the raw chat dataset
2. `Scripts/train_lora.py` — fine-tunes the base model with a LoRA adapter
3. `Scripts/inference_test.py` — loads the base model + adapter and runs test prompts

Note: the folder is `Scripts/` with a capital S. Linux is case-sensitive, so commands
below match that exactly.

## Model & training configuration
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Base model quantization: 4-bit (NF4, double quantization), compute dtype bf16
- LoRA: r=32, alpha=64, dropout=0.05, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Epochs: 6
- Batch size: 1, gradient accumulation: 8 (effective batch size 8)
- Learning rate: 2e-4
- Max sequence length: 1024
- Gradient checkpointing: enabled
- Dataset: 55 chat examples (`Chat_Data_for_assessment_of_applicants.json`), no
  additional data mixed in

`inference_test.py` loads the base model with the same 4-bit `BitsAndBytesConfig` used
in `train_lora.py`, so training and inference use the same quantization and fit the
same 6GB GPU.

## Why WSL2 instead of native Windows

Native Windows (VS Code + PowerShell venv) hit a hard, inconsistent crash
(`0xC0000005` access violation, no Python traceback) tied to `bitsandbytes`/CUDA
kernel loading on Windows — a known fragile combination, since these libraries are
developed and tested primarily against Linux. WSL2 runs the same libraries on Linux
and resolved the crash completely with no code changes needed. WSL2 is the supported
path for this project.

## Environment

- OS: Ubuntu 24.04 (via WSL2 on Windows)
- Python: 3.12
- GPU: NVIDIA RTX 4050, 6GB VRAM, CUDA 12.4

### Set up WSL2 and Python

Run this in a normal Windows PowerShell/CMD (not inside a venv):

```powershell
wsl --install -d Ubuntu
```

Restart if prompted, then finish Ubuntu's first-run setup (create a Linux
username/password). Open the Ubuntu terminal (or connect VS Code: bottom-left `><`
icon -> "Connect to WSL"), then install Python 3.12 — WSL's system Python is often
too new and lacks prebuilt `torch`/`bitsandbytes` wheels, so install 3.12 specifically
via the deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

python3.12 --version   # should print Python 3.12.x
```

Work from WSL's native filesystem, not `/mnt/c` or `/mnt/d` (Windows-mounted drives
don't support the permissions/symlinks `venv` needs, and are much slower for I/O):

```bash
cd ~
git clone <this-repo-url> vedaz-astrologer-finetune
cd vedaz-astrologer-finetune
```

From here on, always work from `~/vedaz-astrologer-finetune`, and open this path (not
a Windows drive path) as your VS Code folder when connected to WSL.

### Verify GPU passthrough

```bash
nvidia-smi
```

Should show your GPU (confirmed working: RTX 4050, 6GB VRAM). If this fails, WSL's
NVIDIA GPU support needs a driver-level fix on the Windows side (rare on modern
Windows 11).

### Create the virtual environment and install dependencies

```bash
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate
python --version   # should print Python 3.12.x
```

You'll need to run `source .venv-wsl/bin/activate` every time you open a new terminal.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins every dependency, and includes the `--extra-index-url` needed
to install the CUDA build of torch, so this single command installs everything.
Exact versions used: torch 2.6.0+cu124, transformers 5.12.1, peft 0.19.1, trl 1.7.0,
datasets 5.0.0, accelerate 1.14.0, bitsandbytes 0.49.2, sentencepiece 0.2.1,
protobuf 5.29.3.

Confirm GPU is visible to PyTorch:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Should print `True`.

## Libraries used

Each library below is imported directly by at least one script in `Scripts/`.

| Library | Used for | Link |
|---|---|---|
| `os` | filesystem operations (standard library) | https://docs.python.org/3/library/os.html |
| `json` | reading/writing the chat dataset (standard library) | https://docs.python.org/3/library/json.html |
| `pathlib` | resolving project paths (standard library) | https://docs.python.org/3/library/pathlib.html |
| `torch` | tensor ops, CUDA/bf16 support | https://pytorch.org/docs/stable/index.html |
| `datasets` | loading the JSONL training set | https://huggingface.co/docs/datasets |
| `transformers` | base model, tokenizer, 4-bit quantization config | https://huggingface.co/docs/transformers |
| `peft` | LoRA adapter config and application | https://huggingface.co/docs/peft |
| `trl` | supervised fine-tuning trainer (`SFTTrainer`/`SFTConfig`) | https://huggingface.co/docs/trl |
| `bitsandbytes` | 4-bit quantization backend used by `transformers` | https://github.com/bitsandbytes-foundation/bitsandbytes |
| `sentencepiece` | tokenizer backend required by the Qwen tokenizer | https://github.com/google/sentencepiece |
| `protobuf` | serialization format required by `sentencepiece`/`transformers` | https://protobuf.dev/ |
| `accelerate` | device placement and training loop utilities used by `trl`/`transformers` | https://huggingface.co/docs/accelerate |

## Files needed before you start

- Original dataset -> `data/raw/Chat_Data_for_assessment_of_applicants.json`

The file is a sequence of concatenated JSON objects (not a single array or JSONL).
`prepare_data.py` parses this format directly and writes a normalized JSONL file to
`data/processed/train_dataset.jsonl`, validating that every record has both a `user`
and an `assistant` turn.

## Run order

```bash
python Scripts/prepare_data.py      # -> data/processed/train_dataset.jsonl (55 examples)
python Scripts/train_lora.py        # -> outputs/lora_adapter/
python Scripts/inference_test.py    # -> outputs/sample_outputs.txt
```

## What each step produces (deliverables)

- `outputs/lora_adapter/` — the trained LoRA adapter (small files, the actual model
  artifact)
- `outputs/logs/` — training checkpoints + `trainer_state.json` (loss history, proof
  it trained). Only the last 2 checkpoints are kept (`save_total_limit=2`) to avoid
  clutter.
- `outputs/sample_outputs.txt` — generated responses to 4 test prompts