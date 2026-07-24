"""Deprecated compatibility shim for the ``gb_rna`` package.

``aido_rna`` was renamed to ``gb_rna`` in the AIDO.RNA -> GB.RNA rebrand.
Importing ``aido_rna`` re-exports everything from ``gb_rna`` and emits a
``DeprecationWarning``. Use ``gb_rna`` directly in new code.
"""

import warnings

warnings.warn(
    "The 'aido_rna' package is deprecated and will be removed in a future "
    "release; import 'gb_rna' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gb_rna import *  # noqa: E402,F401,F403
from gb_rna import (  # noqa: E402,F401
    RNABertConfig,
    RNABertForMaskedLM,
    RNABertModel,
    RNABertTokenizer,
    get_vocab_filepath,
    validate_sequences,
)

__version__ = "0.1.0"

__all__ = [
    "RNABertModel",
    "RNABertConfig",
    "RNABertTokenizer",
    "RNABertForMaskedLM",
    "validate_sequences",
    "get_vocab_filepath",
]
