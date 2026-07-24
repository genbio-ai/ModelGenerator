import torch
import pandas as pd
import ast
import argparse
from modelgenerator.structure_tokenizer.utils.constants import residue_constants as RC


def struct_token_tsv_to_pt(tsv_file, output_file):
    """
    Converts a TSV file containing protein sequence and predicted structure tokens
    into a PyTorch-compatible format (.pt) for downstream use in a decoder.

    Parameters:
    - tsv_file (str): Path to the input TSV file containing protein sequences and structure tokens.
    - output_file (str): Path to the output .pt file to store the converted data.

    TSV file format:
    Each row should contain:
        - uid: Unique identifier for the protein sequence.
        - sequences: Protein sequence (e.g., "LRTPTT").
        - labels: Ground truth labels for structure (not used in this function).
        - predictions: Predicted structure tokens as a list (e.g., "[164, 287, 119, ...]").

    Output .pt format:
    A dictionary where:
        - Keys are "<uid>_nan".
        - Values are dictionaries with:
            - "aatype": Tensor of amino acid types (integer encoding for each residue).
            - "struct_tokens": Tensor of predicted structure tokens.
            - "residue_index": Tensor of residue indices (from 0 to n-1).
    """
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, sep="\t")
    ret_dict = {}  # Dictionary to store converted data

    # Process each row of the DataFrame
    for row in df.iloc:
        uid = row["uid"]  # Unique identifier for the protein sequence
        # Convert amino acid sequence to integer encoding using residue_constants
        aatype = [RC.restype_1toidx[aa] for aa in row["sequences"]]
        # Parse the predicted structure tokens from the string representation
        struct_token = ast.literal_eval(row["predictions"])
        # Generate a residue index list (0-based indices for each residue)
        residue_index = list(range(len(aatype)))
        # Add the converted data to the dictionary
        ret_dict[f"{uid}_nan"] = {
            "aatype": torch.tensor(aatype, dtype=torch.int64),  # Tensor of amino acid types
            "struct_tokens": torch.tensor(struct_token, dtype=torch.int64),  # Tensor of structure tokens
            "residue_index": torch.tensor(residue_index, dtype=torch.int64),  # Tensor of residue indices
        }

    # Save the dictionary to the output .pt file
    torch.save(ret_dict, output_file)


def main():
    """
    Main function to parse command-line arguments and execute the conversion process.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert the format of structure token from TSV to PyTorch (.pt) format.\n"
            "This script processes a TSV file containing predicted structure tokens from "
            "a language model and converts it into a format compatible with a decoder.\n"
            "Input TSV columns:\n"
            "  - uid: Unique identifier.\n"
            "  - sequences: Protein sequences (e.g., 'LRTPTT').\n"
            "  - labels: Ground truth structure labels (not used).\n"
            "  - predictions: Predicted structure tokens (e.g., '[164, 287, ...]').\n\n"
            "Output .pt file format:\n"
            "  A dictionary with keys '<uid>_nan' and values containing 'aatype',\n"
            "  'struct_tokens', and 'residue_index'."
        )
    )
    parser.add_argument("tsv_file", type=str, help="Path to the input TSV file.")
    parser.add_argument("output_file", type=str, help="Path to the output PyTorch (.pt) file.")
    args = parser.parse_args()

    # Call the conversion function with provided arguments
    struct_token_tsv_to_pt(args.tsv_file, args.output_file)


if __name__ == "__main__":
    # Entry point for the script
    main()
