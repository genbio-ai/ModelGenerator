import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from modelgenerator.structure_tokenizer.utils.data_parsing import register_mmcif_chains, register_pdb_chains
from tqdm import tqdm


def register_dataset(
    folder_path: str | Path,
    format: Literal["cif.gz", "cif", "ent.gz", "pdb"],
    output_file: str | Path,
):
    """
    Registers a dataset of protein structures from a specified folder and saves the metadata to a CSV or Parquet file.

    Parameters:
    ----------
    folder_path : str | Path
        The path to the folder containing the protein structure files.
    format : Literal["cif.gz", "cif", "ent.gz", "pdb"]
        The format of the protein structure files.
    output_file : str | Path
        The path to the output file where the metadata will be saved. Must be a .csv or .parquet file.

    Returns:
    -------
    None

    Notes
    -----
    mmcif
    -----
    schema={
        "filename": str,
        "pdb_id": str,
        "entity": 'uint16',
        "chain": str,
        "nb_residues": 'uint16',
        "nb_complete_backbones": 'uint16',
    }

    pdb
    ---
    schema={
        "filename": str,
        "pdb_id": str,
        "chain": str,
        "nb_residues": 'uint16',
        "nb_complete_backbones": 'uint16',
    }
    """
    folder_path = Path(folder_path)
    output_file = Path(output_file)
    assert output_file.suffix in [".csv", ".parquet"], "Output file must be .csv or .parquet"
    pdb_files = list(folder_path.rglob(f"*.{format}"))
    match format:
        case "cif.gz" | "cif":
            df = pd.concat([register_mmcif_chains(cif_file_path=pdb_file) for pdb_file in tqdm(pdb_files)])
        case "ent.gz" | "pdb":
            df = pd.concat([register_pdb_chains(pdb_file_path=pdb_file) for pdb_file in tqdm(pdb_files)])
        case _:
            raise ValueError(f"Unknown format: {format}")
    match output_file.suffix:
        case ".csv":
            df.to_csv(str(output_file), index=False)
        case ".parquet":
            df.to_parquet(str(output_file))
        case _:
            raise ValueError("Output file must be .csv or .parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register a dataset of protein structures and save the metadata to a CSV or Parquet file."
    )
    parser.add_argument(
        "--folder_path", type=str, help="The path to the folder containing the protein structure files.", required=True
    )
    parser.add_argument(
        "--format",
        type=str,
        help="The format of the protein structure files.",
        choices=["cif.gz", "cif", "ent.gz", "pdb"],
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to the output file where the metadata will be saved. Must be a .csv or .parquet file.",
        required=True,
    )

    args = parser.parse_args()
    register_dataset(
        folder_path=args.folder_path,
        format=args.format,
        output_file=args.output_file,
    )
