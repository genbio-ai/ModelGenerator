#!/usr/bin/env python3
"""
AIDO.RNA Inference/Generation Script

This script uses the pretrained decoder head from AIDO.RNA to reconstruct
masked nucleotides from input sequences.

Set the configuration variables below and run: python inference.py
"""

import torch
from tqdm import tqdm

# Import from aido_rna package
from aido_rna import RNABertForMaskedLM, RNABertTokenizer
from aido_rna.utils import get_vocab_filepath

# ============================================================
# CONFIGURATION - Set these variables
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.RNA-1M-MARS"     # HuggingFace model handle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SEQS = [
    "AACTTTTTGGTT.CGAGCT",
    "GGGAA...CCCTTTGGGAAA",
]
# ============================================================


def main():
    print(f"AIDO.RNA Inference Script")
    print(f"=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Number of sequences: {len(INPUT_SEQS)}")
    print(f"=" * 60)

    # Step 1: Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    vocab_file = get_vocab_filepath()
    tokenizer = RNABertTokenizer(vocab_file, version="v2")
    print(f"✓ Tokenizer loaded from {vocab_file}")

    # Step 2: Load model
    print("\n[2/3] Loading model...")
    model = RNABertForMaskedLM.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()
    print(f"✓ Model loaded and moved to {DEVICE}")

    # Step 3: Reconstruct masked sequences
    print("\n[3/3] Reconstructing masked sequences...")
    print()

    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id

    for i, masked_seq in enumerate(tqdm(INPUT_SEQS, desc="Reconstructing")):
        # Tokenize sequence (space-separate nucleotides including dots)
        spaced_seq = " ".join(masked_seq)
        encoded = tokenizer(
            spaced_seq,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        # Replace dots (UNK tokens) with [MASK] tokens
        input_ids = encoded["input_ids"].clone()
        mask_positions = (input_ids == unk_token_id)
        input_ids[mask_positions] = mask_token_id

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=encoded["attention_mask"])
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Decode predictions back to sequence
        predicted_tokens = predictions[0].cpu().tolist()
        reconstructed = []
        for j, token_id in enumerate(predicted_tokens):
            if mask_positions[0, j]:
                # Masked position - use prediction
                predicted_token = tokenizer.decode([token_id]).strip()
                if predicted_token in ['A', 'C', 'G', 'T', 'U', 'N']:
                    reconstructed.append(predicted_token)
                else:
                    reconstructed.append('N')
            else:
                # Not masked - use original token
                original_token = tokenizer.decode([input_ids[0, j].item()]).strip()
                if original_token in ['A', 'C', 'G', 'T', 'U', 'N']:
                    reconstructed.append(original_token)

        reconstructed_seq = ''.join(reconstructed)

        # Display results
        print(f"\nSequence {i + 1}:")
        print(f"  Original:      {masked_seq}")
        print(f"  Reconstructed: {reconstructed_seq}")

        # Highlight predicted positions
        if len(reconstructed_seq) == len(masked_seq):
            diff = []
            for orig, recon in zip(masked_seq, reconstructed_seq):
                if orig == '.':
                    diff.append("^")
                elif orig == recon:
                    diff.append(" ")
                else:
                    diff.append("X")
            print(f"  Predictions:   {''.join(diff)}")

    print(f"\n{'=' * 60}")
    print(f"Reconstruction completed!")
    print(f"\nNote: '^' indicates positions where predictions were made")
    print(f"      'X' indicates mismatches (unexpected)")


if __name__ == "__main__":
    main()