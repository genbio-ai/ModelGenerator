# AIDO.Cell

Standalone AIDO.Cell model repo using HuggingFace handles.

## Installation

```bash
# Barebones installation
pip install -e .

# FlashAttention2 support
pip install -e ".[flash_attn]"

# PEFT/LoRA support
pip install -e ".[peft]"

# All optional dependencies
pip install -e ".[flash_attn,peft]"
```

## Quickstart

1. **Edit the configuration** in `embed.py`:

```python
# CONFIGURATION - Set these variables
MODEL_NAME = "genbio-ai/AIDO.Cell-3M"  # Or "genbio-ai/AIDO.Cell-100M"
INPUT_FILE = "temp_adata.h5ad"          # Path to your input file
OUTPUT_FILE = None                       # Auto-generates: input_embeddings.h5ad
DEVICE = "cuda"                          # "cuda" or "cpu"
BATCH_SIZE = 32
EMBEDDING_KEY = "X_aido_cell"
```

2. **Run the script**:

```bash
python embed.py
```

## Finetune with LoRA

> **Note**: Fine-tuning requires the `peft` optional dependency. Install with: `uv pip install -e ".[peft]"`

1. **Edit the configuration** in `finetune.py`:

```python
# CONFIGURATION - Set these variables
MODEL_NAME = "genbio-ai/AIDO.Cell-3M"   # HuggingFace model handle
NUM_CLASSES = 5                          # Number of classification classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
FREEZE_BACKBONE = False                  # Set True to freeze AIDO.Cell weights (ignored if USE_LORA=True)

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
from finetune import CellFoundationClassifier

model = CellFoundationClassifier(MODEL_NAME, NUM_CLASSES, FREEZE_BACKBONE, USE_LORA, lora_config)
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Quirks

AIDO.Cell was pre-trained on a fixed set of 19,264 genes using a read depth-aware objective function.
All inputs should be processed using the `aido_cell.utils.gene_alignment` and `aido_cell.utils.preprocessing` tools.

1. Gene alignment 
  1. Removes genes in your data that aren't in AIDO.Cell's gene set
  2. Adds zero-filled entries for genes in AIDO.Cell's set that are missing from your data
  3. Reorders genes to match AIDO.Cell's expected order
  4. Creates attention masks so the model knows which genes are actually present
2. Preprocessing
  1. Calculates log10 of total counts per cell (minimum 5) for depth tokens
  2. Normalizes counts to log1p(CPM) where CPM = counts per 10,000
  3. Appends two depth tokens (rawcountsidx, inputcountidx) to the sequence. 
  In pretraining these indicated the input and desired output depth, but in this script they are fixed to be equal.
  4. Clips values at 20
  5. Converts to bfloat16

## Package Structure

```
aido.cell/
├── embed.py                    # Embedding generation script
├── finetune.py                 # Fine-tuning script with LoRA
├── pyproject.toml              # Package configuration
├── aido_cell/                  # Python package
│   ├── __init__.py
│   ├── models/                 # CellFoundation model implementations
│   │   ├── __init__.py
│   │   ├── configuration_cellfoundation.py
│   │   ├── modeling_cellfoundation.py
│   │   └── gene_lists/         # Reference gene set (19,264 genes)
│   │       └── OS_scRNA_gene_index.19264.tsv
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── gene_alignment.py   # Gene alignment utilities
│       └── preprocessing.py    # Data normalization (log1p CPM + depth tokens)
```

## Available Models

AIDO.Cell models on HuggingFace:
- `genbio-ai/AIDO.Cell-3M`
- `genbio-ai/AIDO.Cell-10M`
- `genbio-ai/AIDO.Cell-100M`

Check the [AIDO.Cell HuggingFace page](https://huggingface.co/genbio-ai) for the latest models.
