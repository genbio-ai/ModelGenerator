# GB.RNA

Standalone GB.RNA model repo using HuggingFace handles.

## Installation

```bash
pip install -e .
```

## Quickstart

1. **Edit the configuration** in `embed.py`:

```python
# CONFIGURATION - Set these variables
MODEL_NAME = "genbio-ai/GB.RNA-1M-MARS"  # Or "genbio-ai/GB.RNA-1.6B"
INPUT_SEQS = [
    "AACTTTTTGGTTTCGAGCT",
    "GGGAAACCCCTTTGGGAAA",
]
DEVICE = "cuda"
BATCH_SIZE = 32
```

2. **Run the script**:

```bash
python embed.py
```

## Finetune with LoRA

1. **Edit the configuration** in `finetune.py`:

```python
# CONFIGURATION - Set these variables
MODEL_NAME = "genbio-ai/GB.RNA-1M-MARS" # HuggingFace model handle
NUM_CLASSES = 5                           # Number of classification classes
DEVICE = "cuda"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
FREEZE_BACKBONE = False                  # Set True to freeze GB.Cell weights (ignored if USE_LORA=True)

# LoRA/PEFT Configuration
USE_LORA = True                          # Set True to use LoRA for parameter-efficient fine-tuning
LORA_R = 8                               # LoRA rank (higher = more parameters, default: 8)
LORA_ALPHA = 16                          # LoRA alpha (scaling factor, default: 16)
LORA_DROPOUT = 0.1                       # LoRA dropout
LORA_TARGET_MODULES = ["query", "value"] # Modules to apply LoRA (query, key, value, dense)
```

2. **Run the fine-tuning script**:

```bash
python finetune.py
```

3. **Load your model**

After training, load your fine-tuned model:

```python
# Todo
```

## Quirks

GB.RNA was pre-trained on a fixed DNA-based vocabulary directly from sequence readouts, so all input sequences must be DNA (A, C, G, T).

## Package Structure

Todo

## Available Models

GB.Cell models on HuggingFace:
- `genbio-ai/GB.RNA-1.6B`
- `genbio-ai/GB.RNA-1.6B-CDS`
- `genbio-ai/GB.RNA-650M`
- `genbio-ai/GB.RNA-650M-CDS`
- `genbio-ai/GB.RNA-300M-MARS`
- `genbio-ai/GB.RNA-25M-MARS`
- `genbio-ai/GB.RNA-1M-MARS`

Check the [GB.RNA HuggingFace page](https://huggingface.co/genbio-ai) for the latest models.