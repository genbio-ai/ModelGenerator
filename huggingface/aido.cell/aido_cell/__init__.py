"""AIDO.Cell: Standalone package for cell foundation models."""

from aido_cell.models import CellFoundationModel, CellFoundationConfig
from aido_cell.utils.gene_alignment import align_adata
from aido_cell.utils.preprocessing import preprocess_counts

__version__ = "0.1.0"

__all__ = [
    "CellFoundationModel",
    "CellFoundationConfig",
    "align_adata",
    "preprocess_counts",
]
