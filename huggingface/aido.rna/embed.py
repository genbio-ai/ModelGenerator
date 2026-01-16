#!/usr/bin/env python3
"""
AIDO.RNA Embedding Script

This script loads pre-trained AIDO.RNA models from HuggingFace and generates embeddings.

Set the configuration variables below and run: python embed.py
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Import from aido_rna package
from aido_rna import RNABertModel, RNABertTokenizer, validate_sequences, get_vocab_filepath
import aido_rna

# ============================================================
# CONFIGURATION - Set these variables
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.RNA-1M-MARS"  # HuggingFace model handle
INPUT_SEQS = [
    "AACTTTTTGGTTTCGAGCT",
    "GGGAAACCCCTTTGGGAAA",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect device
BATCH_SIZE = 32
# ============================================================


def main():
    print(f"AIDO.RNA Embedding Script")
    print(f"=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Number of sequences: {len(INPUT_SEQS)}")
    print(f"=" * 60)

    # Validate sequences
    print("\n[1/4] Validating sequences...")
    validated_seqs = validate_sequences(INPUT_SEQS)
    print(f"✓ Validated {len(validated_seqs)} sequences")

    # Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    tokenizer = RNABertTokenizer(get_vocab_filepath(), version="v2")
    print(f"✓ Tokenizer loaded from {get_vocab_filepath()}")

    # Load model
    print("\n[3/4] Loading model...")
    model = RNABertModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)

    # Convert to bfloat16 if using CUDA for Flash Attention compatibility
    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)

    model.eval()
    print(f"✓ Model loaded and moved to {DEVICE}")

    # Generate embeddings
    print("\n[4/4] Generating embeddings...")
    all_embeddings = []

    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(validated_seqs), BATCH_SIZE), desc="Processing batches"):
            batch_seqs = validated_seqs[i:i + BATCH_SIZE]

            # Tokenize
            inputs = tokenizer(
                batch_seqs,
                max_length=2048,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Move to device if specified
            if DEVICE is not None:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

            # Extract last hidden state
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

            # Mean pooling (weighted by attention mask)
            attention_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)

            all_embeddings.append(embeddings.cpu().float().numpy())

    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)

    # Save embeddings
    output_file = "embeddings.npy"
    np.save(output_file, embeddings_array)

    print(f"\n✓ Embeddings generated successfully!")
    print(f"  Shape: {embeddings_array.shape}")
    print(f"  Saved to: {output_file}")
    print(f"\nExample embeddings (first sequence, first 5 dimensions):")
    print(f"  {embeddings_array[0, :5]}")


if __name__ == "__main__":
    main()
