"""Preprocessing and validation utilities for AIDO.RNA sequences."""

import warnings
from typing import List


def get_vocab_filepath() -> str:
    """
    Abspath of the vocab.txt file.

    Returns:
        Abspath of the vocab.txt file
    """
    import aido_rna
    import os
    package_dir = os.path.dirname(aido_rna.__file__)
    vocab_file = os.path.join(package_dir, "models", "vocab.txt")
    return vocab_file


def validate_sequences(sequences: List[str], mask_char: str = ".") -> List[str]:
    """
    Validate and clean DNA/RNA sequences.

    This function:
    - Converts sequences to uppercase
    - Converts U (RNA) to T (DNA) with a warning
    - Validates that sequences only contain allowed characters
    - Converts a mask_char (.) to '[MASK]'

    Args:
        sequences: List of DNA/RNA sequences
        mask_char: Character used to represent masked positions (default: ".")

    Returns:
        List of validated and cleaned sequences

    Raises:
        ValueError: If sequences contain invalid characters

    Example:
        >>> sequences = ["acgt", "gcua.ta"]
        >>> validated = validate_sequences(sequences)
        >>> print(validated)
        ['ACGT', 'GCTA[MASK]TA']
    """
    validated = []
    u_found = False
    vocab = []
    with open(get_vocab_filepath(), "r") as vf:
        for line in vf:
            vocab.append(line.strip())

    for i, seq in enumerate(sequences):
        # Replace mask character with [MASK]
        seq = seq.replace(mask_char, "[MASK]")

        # Convert to uppercase
        seq = seq.upper()
        
        # Convert U to T
        if "U" in seq:
            u_found = True
        seq = seq.replace("U", "T")

        # Check for substrings not in valid characters
        temp_seq = seq
        for char in vocab:
            temp_seq = temp_seq.replace(char, "")
        if len(temp_seq) > 0:
            raise ValueError(
                f"Invalid characters found in sequence '{seq}': {set(temp_seq)}. "
                f"Allowed characters are: {vocab}."
            )

        validated.append(" ".join(seq))

    if u_found:
        warnings.warn(
            "Found 'U' nucleotides in input sequences. Converting U → T for DNA input. "
            "AIDO.RNA models expect DNA sequences (A, C, G, T).",
            UserWarning,
        )
    return validated
