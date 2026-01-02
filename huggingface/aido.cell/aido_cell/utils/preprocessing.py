import torch
import numpy as np


def preprocess_counts(counts, device='cpu'):
    """Preprocesses raw gene counts for AIDO.Cell model input.

    This function performs the following steps:
    1. Calculates log10 of total counts per cell (minimum 5) for depth tokens
    2. Normalizes counts to log1p(CPM) where CPM = counts per 10,000
    3. Appends two depth tokens (rawcountsidx, inputcountidx) to the sequence
    4. Clips values at 20
    5. Converts to bfloat16

    Args:
        counts: Raw gene expression counts. Can be:
            - numpy array of shape (n_cells, n_genes)
            - torch tensor of shape (n_cells, n_genes)
        device: Device to place the tensor on ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Preprocessed gene expression tensor of shape (n_cells, n_genes + 2)
            ready for model input, in bfloat16 format
    """
    # Convert to torch tensor if needed
    if isinstance(counts, np.ndarray):
        X = torch.from_numpy(counts).to(dtype=torch.float32, device=device)
    else:
        X = counts.to(dtype=torch.float32, device=device)

    # Calculate depth tokens (log10 of total counts per cell, minimum 5)
    # rawcountsidx and inputcountidx are both calculated the same way
    total_counts = X.sum(dim=1, keepdim=True)
    rawcountsidx = torch.maximum(
        torch.log10(total_counts), torch.tensor(5.0, device=X.device)
    )
    inputcountidx = torch.maximum(
        torch.log10(total_counts), torch.tensor(5.0, device=X.device)
    )

    # Normalize to log1p(CPM) where CPM = counts per 10,000
    X = torch.log1p(X / total_counts * 10000)

    # Concatenate depth tokens to the end
    X = torch.cat(
        (
            X,
            rawcountsidx.to(X.device),
            inputcountidx.to(X.device),
        ),
        axis=1,
    )

    # Clip values at 20
    X = torch.clamp(X, max=20)

    # Convert to bfloat16 if on CUDA (Flash Attention requires bf16/fp16)
    # Keep float32 for CPU compatibility
    if X.is_cuda:
        X = X.to(torch.bfloat16)

    return X
