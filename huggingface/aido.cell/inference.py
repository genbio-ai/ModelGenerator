#!/usr/bin/env python3
"""
AIDO.Cell Gene Expression Reconstruction Script

This script uses the pretrained decoder head from AIDO.Cell to reconstruct
gene expression from single-cell RNA-seq data in AnnData format.

Set the configuration variables below and run: python inference.py
"""

import os
import sys
import torch
import anndata as ad
import numpy as np
from tqdm import tqdm
from scipy import stats

from aido_cell.models import CellFoundationConfig
from aido_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
from aido_cell.utils import align_adata, preprocess_counts

# ============================================================
# CONFIGURATION - Set these variables
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.Cell-3M"     # HuggingFace model handle
INPUT_FILE = "temp_adata.h5ad"            # Path to input AnnData file
OUTPUT_FILE = None                         # Output file (None = auto-generate)
OUTPUT_LAYER = "reconstructed"             # Key for adata.layers to store results
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
SAVE_AS_CPM = False                        # If True, convert from log1p(CPM) to CPM
                                           # If False, keep as log1p(CPM)
COMPUTE_METRICS = True                     # Calculate reconstruction quality metrics
# ============================================================

# Set output path if not specified
if OUTPUT_FILE is None:
    base_name = os.path.splitext(INPUT_FILE)[0]
    OUTPUT_FILE = f"{base_name}_reconstructed.h5ad"

# Print configuration
print("\n" + "="*60)
print("AIDO.Cell Gene Expression Reconstruction")
print("="*60)
print(f"Input data: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Output format: {'CPM' if SAVE_AS_CPM else 'log1p(CPM)'}")
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

# Load model with decoder head
print(f"\n{'='*60}")
print(f"Loading model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"{'='*60}\n")

try:
    config = CellFoundationConfig.from_pretrained(MODEL_NAME)
    # Use CellFoundationForMaskedLM for reconstruction (has decoder head)
    model = CellFoundationForMaskedLM.from_pretrained(MODEL_NAME, config=config)
    model = model.to(DEVICE)

    # Convert model to bfloat16 if using CUDA (required for Flash Attention)
    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)
        print("✓ Model converted to bfloat16 for Flash Attention")

    model.eval()
    print("✓ Model loaded successfully")
    print("✓ Model has decoder head for reconstruction\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Perform gene expression reconstruction
print(f"\n{'='*60}")
print("Performing gene expression reconstruction")
print(f"Number of cells: {adata_aligned.n_obs}")
print(f"Batch size: {BATCH_SIZE}")
print(f"{'='*60}\n")

n_cells = adata_aligned.n_obs
all_reconstructions = []

# Convert attention mask to torch tensor once
attn_mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).to(DEVICE)

# Process in batches
with torch.no_grad():
    for start_idx in tqdm(range(0, n_cells, BATCH_SIZE), desc="Reconstructing"):
        end_idx = min(start_idx + BATCH_SIZE, n_cells)

        # Get batch of raw counts
        batch_counts = adata_aligned.X[start_idx:end_idx]
        if hasattr(batch_counts, 'toarray'):
            batch_counts = batch_counts.toarray()

        # Preprocess counts (normalize, add depth tokens)
        batch_processed = preprocess_counts(batch_counts, device=DEVICE)

        # Expand attention mask to match batch size
        batch_attn_mask = attn_mask_tensor.repeat(batch_processed.shape[0], 1)

        # Add 2 positions to attention mask for depth tokens
        depth_token_mask = torch.ones((batch_processed.shape[0], 2), device=DEVICE)
        batch_attn_mask = torch.cat([batch_attn_mask, depth_token_mask], dim=1)

        # Forward pass through model
        outputs = model(
            input_ids=batch_processed,
            attention_mask=batch_attn_mask,
            return_dict=True
        )

        # Extract gene predictions from logits
        # logits shape: [batch, seq_len, 1] where seq_len = 19266 (19264 genes + 2 depth)
        logits = outputs.logits  # [batch, 19266, 1]

        # Remove last 2 positions (depth token predictions)
        gene_logits = logits[:, :-2, :]  # [batch, 19264, 1]

        # Squeeze out the last dimension
        gene_predictions = gene_logits.squeeze(-1)  # [batch, 19264]

        # Denormalize if requested (convert log1p(CPM) to CPM)
        if SAVE_AS_CPM:
            # Reverse log1p: expm1(x) = exp(x) - 1
            gene_predictions = torch.expm1(gene_predictions)

        # Move to CPU and store (convert to float32 first, numpy doesn't support bfloat16)
        all_reconstructions.append(gene_predictions.float().cpu().numpy())

# Concatenate all batches
reconstructed = np.vstack(all_reconstructions)
print(f"\n✓ Reconstructed expression with shape: {reconstructed.shape}")
print(f"✓ Values are in {'CPM' if SAVE_AS_CPM else 'log1p(CPM)'} space\n")

# Calculate reconstruction quality metrics
if COMPUTE_METRICS:
    print("="*60)
    print("Computing reconstruction quality metrics")
    print("="*60 + "\n")

    # Get original data in same space as predictions
    original_data = adata_aligned.X
    if hasattr(original_data, 'toarray'):
        original_data = original_data.toarray()

    # Normalize original data to log1p(CPM) for fair comparison
    total_counts = original_data.sum(axis=1, keepdims=True)
    original_log1p_cpm = np.log1p(original_data / total_counts * 10000)

    # If we saved as CPM, convert back to log1p(CPM) for metrics
    if SAVE_AS_CPM:
        reconstructed_log1p_cpm = np.log1p(reconstructed)
    else:
        reconstructed_log1p_cpm = reconstructed

    # Calculate metrics
    # 1. Mean Squared Error
    mse = np.mean((original_log1p_cpm - reconstructed_log1p_cpm) ** 2)

    # 2. Pearson correlation (per-cell average)
    cell_correlations = []
    for i in range(original_log1p_cpm.shape[0]):
        # Only correlate non-zero genes to avoid artificial high correlation
        mask = (original_log1p_cpm[i] > 0) | (reconstructed_log1p_cpm[i] > 0)
        if mask.sum() > 1:
            corr, _ = stats.pearsonr(
                original_log1p_cpm[i][mask],
                reconstructed_log1p_cpm[i][mask]
            )
            cell_correlations.append(corr)

    mean_correlation = np.mean(cell_correlations)

    # 3. Gene-wise correlation
    gene_correlations = []
    for j in range(original_log1p_cpm.shape[1]):
        mask = (original_log1p_cpm[:, j] > 0) | (reconstructed_log1p_cpm[:, j] > 0)
        if mask.sum() > 1:
            corr, _ = stats.pearsonr(
                original_log1p_cpm[mask, j],
                reconstructed_log1p_cpm[mask, j]
            )
            gene_correlations.append(corr)

    mean_gene_correlation = np.mean(gene_correlations)

    print(f"Mean Squared Error (log1p(CPM) space): {mse:.4f}")
    print(f"Mean per-cell Pearson correlation: {mean_correlation:.4f}")
    print(f"Mean per-gene Pearson correlation: {mean_gene_correlation:.4f}")
    print(f"Number of cells evaluated: {len(cell_correlations)}")
    print(f"Number of genes evaluated: {len(gene_correlations)}\n")

# Store reconstructed expression in original adata object
print("Saving reconstructed expression...")
adata.layers[OUTPUT_LAYER] = reconstructed

# Add metadata about reconstruction
adata.uns['reconstruction_params'] = {
    'model': MODEL_NAME,
    'device': DEVICE,
    'batch_size': BATCH_SIZE,
    'output_format': 'CPM' if SAVE_AS_CPM else 'log1p(CPM)',
}

if COMPUTE_METRICS:
    adata.uns['reconstruction_metrics'] = {
        'mse': float(mse),
        'mean_cell_correlation': float(mean_correlation),
        'mean_gene_correlation': float(mean_gene_correlation),
    }

# Save to output file
adata.write_h5ad(OUTPUT_FILE)
print(f"✓ Saved results to {OUTPUT_FILE}")
print(f"  Reconstructed expression in adata.layers['{OUTPUT_LAYER}']")
print(f"  Metadata in adata.uns['reconstruction_params']")
if COMPUTE_METRICS:
    print(f"  Metrics in adata.uns['reconstruction_metrics']")
print()

print("="*60)
print("✓ Reconstruction complete!")
print("="*60 + "\n")
