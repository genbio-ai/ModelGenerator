#!/usr/bin/env python3
"""
AIDO.Cell Embedding Script

This script loads pre-trained AIDO.Cell models from HuggingFace and generates
cell embeddings from single-cell RNA-seq data in AnnData format.

Set the configuration variables below and run: python embed.py
"""

import os
import sys
import torch
import anndata as ad
import numpy as np
from tqdm import tqdm

from aido_cell.models import CellFoundationModel, CellFoundationConfig
from aido_cell.utils import align_adata, preprocess_counts

# ============================================================
# CONFIGURATION - Set these variables
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.Cell-3M"  # HuggingFace model handle
INPUT_FILE = "temp_adata.h5ad"          # Path to input AnnData file
OUTPUT_FILE = None                       # Output file (None = auto-generate from input)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect device
BATCH_SIZE = 32
EMBEDDING_KEY = "X_aido_cell"           # Key to store embeddings in adata.obsm
# ============================================================

# Set output path if not specified
if OUTPUT_FILE is None:
    base_name = os.path.splitext(INPUT_FILE)[0]
    OUTPUT_FILE = f"{base_name}_embeddings.h5ad"

# Print configuration
print("\n" + "="*60)
print("AIDO.Cell Embedding Generation")
print("="*60)
print(f"Input data: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print("="*60 + "\n")

# Load input data
print("Loading input data...")
try:
    adata = ad.read_h5ad(INPUT_FILE)
    print(f"✓ Loaded data with {adata.n_obs} cells and {adata.n_vars} genes\n")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Align data to AIDO.Cell gene set
adata_aligned, attention_mask = align_adata(adata)

# Load model
print(f"\n{'='*60}")
print(f"Loading model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"{'='*60}\n")

try:
    config = CellFoundationConfig.from_pretrained(MODEL_NAME)
    model = CellFoundationModel.from_pretrained(MODEL_NAME, config=config)
    model = model.to(DEVICE)

    # Convert model to bfloat16 if using CUDA (required for Flash Attention)
    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)
        print("✓ Model converted to bfloat16 for Flash Attention")

    model.eval()
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Generate embeddings
print(f"\n{'='*60}")
print("Generating embeddings")
print(f"Number of cells: {adata_aligned.n_obs}")
print(f"Batch size: {BATCH_SIZE}")
print(f"{'='*60}\n")

n_cells = adata_aligned.n_obs
all_embeddings = []

# Convert attention mask to torch tensor once
attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(DEVICE)

# Process in batches
with torch.no_grad():
    for start_idx in tqdm(range(0, n_cells, BATCH_SIZE), desc="Processing batches"):
        end_idx = min(start_idx + BATCH_SIZE, n_cells)

        # Get batch of raw counts
        batch_counts = adata_aligned.X[start_idx:end_idx]
        if hasattr(batch_counts, 'toarray'):
            batch_counts = batch_counts.toarray()

        # Preprocess counts (normalize, add depth tokens)
        batch_processed = preprocess_counts(batch_counts, device=DEVICE)

        # Expand attention mask to match batch size
        batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)

        # Need to add 2 to attention mask for the two depth tokens
        depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=DEVICE)
        batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

        # Forward pass through model
        outputs = model(
            input_ids=batch_processed,
            attention_mask=batch_attn_mask,
            output_hidden_states=True
        )

        # Extract embeddings from last hidden state
        # Take the mean over all gene positions (excluding padding)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Average pooling over sequence dimension (weighted by attention mask)
        mask_expanded = batch_attn_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        batch_embeddings = sum_embeddings / sum_mask

        # Move to CPU and store
        all_embeddings.append(batch_embeddings.cpu().numpy())

# Concatenate all batches
embeddings = np.vstack(all_embeddings)
print(f"\n✓ Generated embeddings with shape: {embeddings.shape}\n")

# Store embeddings in original adata object
print("Saving embeddings...")
adata.obsm[EMBEDDING_KEY] = embeddings

# Save to output file
adata.write_h5ad(OUTPUT_FILE)
print(f"✓ Saved embeddings to {OUTPUT_FILE}")
print(f"  Embeddings stored in adata.obsm['{EMBEDDING_KEY}']\n")

print("="*60)
print("✓ Embedding generation complete!")
print("="*60 + "\n")
