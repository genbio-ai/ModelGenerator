"""Deprecated compatibility shim for the ``gb_cell`` package.

``aido_cell`` was renamed to ``gb_cell`` in the AIDO.Cell -> GB.Cell rebrand.
Importing ``aido_cell`` re-exports everything from ``gb_cell`` and emits a
``DeprecationWarning``. Use ``gb_cell`` directly in new code.
"""

import warnings

warnings.warn(
    "The 'aido_cell' package is deprecated and will be removed in a future "
    "release; import 'gb_cell' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gb_cell import *  # noqa: E402,F401,F403
from gb_cell import (  # noqa: E402,F401
    CellFoundationConfig,
    CellFoundationModel,
    align_adata,
    preprocess_counts,
)

__version__ = "0.1.0"

__all__ = [
    "CellFoundationModel",
    "CellFoundationConfig",
    "align_adata",
    "preprocess_counts",
]
