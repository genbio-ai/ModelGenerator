#!/usr/bin/env python3
"""
AIDO.Cell Fine-tuning Script

This script demonstrates how to fine-tune AIDO.Cell with a linear classification head.
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


from aido_cell.models import CellFoundationModel, CellFoundationConfig
from aido_cell.utils import preprocess_counts


# ============================================================
# CONFIGURATION - Set these variables
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.Cell-3M"   # HuggingFace model handle
NUM_CLASSES = 5                          # Number of classification classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
FREEZE_BACKBONE = False                  # Set True to freeze AIDO.Cell weights (ignored if USE_LORA=True)
NUM_TRAIN_SAMPLES = 100                  # Number of synthetic training samples
NUM_VAL_SAMPLES = 20                     # Number of synthetic validation samples
NUM_GENES = 19264                        # AIDO.Cell gene count

# LoRA/PEFT Configuration
USE_LORA = True                          # Set True to use LoRA for parameter-efficient fine-tuning
LORA_R = 8                               # LoRA rank (higher = more parameters, default: 8)
LORA_ALPHA = 16                          # LoRA alpha (scaling factor, default: 16)
LORA_DROPOUT = 0.1                       # LoRA dropout
LORA_TARGET_MODULES = ["query", "value"] # Modules to apply LoRA (query, key, value, dense)
# ============================================================

print("\n" + "="*60)
print("AIDO.Cell Fine-tuning Example")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Number of classes: {NUM_CLASSES}")
if USE_LORA:
    print(f"Training mode: LoRA (rank={LORA_R}, alpha={LORA_ALPHA})")
else:
    print(f"Training mode: {'Linear probing' if FREEZE_BACKBONE else 'Full fine-tuning'}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print("="*60 + "\n")


# ============================================================
# Create Synthetic Dataset
# ============================================================
class SyntheticCellDataset(Dataset):
    """Synthetic single-cell dataset for demonstration."""

    def __init__(self, num_samples, num_genes, num_classes, seed):
        np.random.seed(seed)
        # Generate random count data (log-normal distribution simulates gene expression)
        self.counts = np.random.lognormal(mean=2.0, sigma=1.5, size=(num_samples, num_genes))
        self.counts = np.clip(self.counts, 0, 10000).astype(np.float32)
        # Generate random labels
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'counts': torch.from_numpy(self.counts[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


print("Generating synthetic training data...")
train_dataset = SyntheticCellDataset(NUM_TRAIN_SAMPLES, NUM_GENES, NUM_CLASSES, seed=42)
print(f"✓ Created training dataset with {len(train_dataset)} samples")

print("Generating synthetic validation data...")
val_dataset = SyntheticCellDataset(NUM_VAL_SAMPLES, NUM_GENES, NUM_CLASSES, seed=123)
print(f"✓ Created validation dataset with {len(val_dataset)} samples\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# Define Model with Classification Head
# ============================================================
class CellFoundationClassifier(nn.Module):
    """AIDO.Cell model with classification head."""

    def __init__(self, model_name, num_classes, freeze_backbone, use_lora=False, lora_config=None):
        super().__init__()
        # Load pre-trained AIDO.Cell model
        config = CellFoundationConfig.from_pretrained(model_name)
        self.backbone = CellFoundationModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.use_lora = use_lora

        # Apply LoRA if specified
        if use_lora:
            if lora_config is None:
                raise ValueError("lora_config must be provided when use_lora=True")
            self.backbone = get_peft_model(self.backbone, lora_config)
            print(f"✓ LoRA applied to backbone")
            self.backbone.print_trainable_parameters()
        # Freeze backbone if specified (only used when not using LoRA)
        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Get embeddings from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Mean pooling (weighted by attention mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1)
        if last_hidden_state.dtype == torch.bfloat16:
            mask_expanded = mask_expanded.to(torch.bfloat16)
        else:
            mask_expanded = mask_expanded.float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        pooled = sum_embeddings / sum_mask

        # Classification
        logits = self.classifier(pooled)
        return logits


print(f"Loading model: {MODEL_NAME}")

# Setup LoRA config if using LoRA
lora_config = None
if USE_LORA:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

model = CellFoundationClassifier(
    MODEL_NAME,
    NUM_CLASSES,
    FREEZE_BACKBONE,
    use_lora=USE_LORA,
    lora_config=lora_config
)
model = model.to(DEVICE)

if FREEZE_BACKBONE and not USE_LORA:
    print("✓ Backbone weights frozen")

# Convert to bfloat16 if using CUDA
if DEVICE == "cuda":
    model = model.to(torch.bfloat16)
    print("✓ Model converted to bfloat16 for Flash Attention")

print(f"✓ Model loaded with {NUM_CLASSES} output classes")
print(f"✓ Hidden size: {model.hidden_size}\n")


# ============================================================
# Setup Training
# ============================================================
# Create attention mask (all genes present in synthetic data)
attention_mask_base = torch.ones(NUM_GENES, dtype=torch.float32)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Optimizer: AdamW (lr={LEARNING_RATE})")
print(f"Loss: CrossEntropyLoss\n")


# ============================================================
# Training Loop
# ============================================================
print("="*60)
print("Starting training...")
print("="*60 + "\n")

best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    # ========== Training ==========
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_loader, desc="Training"):
        counts = batch['counts'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        # Preprocess counts
        input_ids = preprocess_counts(counts, device=DEVICE)

        # Create attention mask (add 2 for depth tokens)
        batch_size = input_ids.shape[0]
        attn_mask = attention_mask_base.unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        depth_mask = torch.ones((batch_size, 2), device=DEVICE)
        attention_mask = torch.cat([attn_mask, depth_mask], dim=1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        pred = logits.argmax(dim=1)
        train_correct += (pred == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # ========== Validation ==========
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            counts = batch['counts'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Preprocess counts
            input_ids = preprocess_counts(counts, device=DEVICE)

            # Create attention mask
            batch_size = input_ids.shape[0]
            attn_mask = attention_mask_base.unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
            depth_mask = torch.ones((batch_size, 2), device=DEVICE)
            attention_mask = torch.cat([attn_mask, depth_mask], dim=1)

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Track metrics
            val_loss += loss.item()
            pred = logits.argmax(dim=1)
            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    # Print epoch results
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pt')
        print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")

    print()

print("="*60)
print("Training complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("="*60 + "\n")

print("Model saved to: best_model.pt")
print("\nTo load the fine-tuned model:")
print("  model = CellFoundationClassifier(MODEL_NAME, NUM_CLASSES, FREEZE_BACKBONE)")
print("  checkpoint = torch.load('best_model.pt')")
print("  model.load_state_dict(checkpoint['model_state_dict'])")
