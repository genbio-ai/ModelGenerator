#!/usr/bin/env python3
"""
AIDO.RNA Fine-tuning Script

This script demonstrates how to fine-tune AIDO.RNA with a linear classification head.
Uses random synthetic data for demonstration purposes.

Set the configuration variables below and run: python finetune.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType

# Import from aido_rna package
from aido_rna import RNABertModel, RNABertTokenizer
from aido_rna.utils import get_vocab_filepath
import aido_rna

# ============================================================
# CONFIGURATION - Set these variables
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.RNA-1M-MARS"   # HuggingFace model handle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5                           # Number of classification classes
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
FREEZE_BACKBONE = False                  # Set True to freeze AIDO.RNA weights (ignored if USE_LORA=True)
NUM_TRAIN_SAMPLES = 100                  # Number of synthetic training samples
NUM_VAL_SAMPLES = 20                     # Number of synthetic validation samples
SEQ_LENGTH = 100                         # Length of synthetic sequences

# LoRA/PEFT Configuration
USE_LORA = True                          # Set True to use LoRA for parameter-efficient fine-tuning
LORA_R = 8                               # LoRA rank (higher = more parameters, default: 8)
LORA_ALPHA = 16                          # LoRA alpha (scaling factor, default: 16)
LORA_DROPOUT = 0.1                       # LoRA dropout
LORA_TARGET_MODULES = ["query", "value"] # Modules to apply LoRA (query, key, value, dense)
# ============================================================


class SyntheticSequenceDataset(Dataset):
    """Generate random DNA sequences with random labels."""

    def __init__(self, num_samples, seq_length, num_classes):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Generate random sequences
        nucleotides = ['A', 'C', 'G', 'T']
        self.sequences = [
            ''.join(np.random.choice(nucleotides, size=seq_length))
            for _ in range(num_samples)
        ]

        # Generate random labels
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class RNABertClassifier(nn.Module):
    """RNABert model with a classification head."""

    def __init__(self, backbone, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        # Get embeddings from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classification
        logits = self.classifier(pooled_output)
        return logits


def train_epoch(model, dataloader, criterion, optimizer, tokenizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in tqdm(dataloader, desc="Training"):
        # Tokenize
        inputs = tokenizer(
            [" ".join(seq) for seq in sequences],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, tokenizer, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            # Tokenize
            inputs = tokenizer(
                [" ".join(seq) for seq in sequences],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    print(f"AIDO.RNA Fine-tuning Script")
    print(f"=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Use LoRA: {USE_LORA}")
    print(f"=" * 60)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    vocab_file = get_vocab_filepath()
    tokenizer = RNABertTokenizer(vocab_file, version="v2")
    print(f"✓ Tokenizer loaded from {vocab_file}")

    # Load model
    print("\n[2/5] Loading model...")
    backbone = RNABertModel.from_pretrained(MODEL_NAME)
    hidden_size = backbone.config.hidden_size

    # Apply LoRA if enabled
    if USE_LORA:
        print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()
    elif FREEZE_BACKBONE:
        print("Freezing backbone parameters...")
        for param in backbone.parameters():
            param.requires_grad = False

    # Create classifier
    model = RNABertClassifier(backbone, hidden_size, NUM_CLASSES)
    model = model.to(DEVICE)

    # Convert to bfloat16 if using CUDA
    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)

    print(f"✓ Model loaded and moved to {DEVICE}")

    # Create datasets
    print("\n[3/5] Creating synthetic datasets...")
    train_dataset = SyntheticSequenceDataset(NUM_TRAIN_SAMPLES, SEQ_LENGTH, NUM_CLASSES)
    val_dataset = SyntheticSequenceDataset(NUM_VAL_SAMPLES, SEQ_LENGTH, NUM_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✓ Created {NUM_TRAIN_SAMPLES} training samples and {NUM_VAL_SAMPLES} validation samples")

    # Training setup
    print("\n[4/5] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"✓ Optimizer: AdamW (lr={LEARNING_RATE})")

    # Training loop
    print(f"\n[5/5] Training for {NUM_EPOCHS} epochs...")
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, tokenizer, DEVICE
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, tokenizer, DEVICE
        )
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"✓ Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: best_model.pt")


if __name__ == "__main__":
    main()
