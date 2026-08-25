import logging
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


from modelgenerator.structure_tokenizer.configs.data_configs import ProteinDatasetConfig
from modelgenerator.structure_tokenizer.datasets.protein import Protein
from modelgenerator.structure_tokenizer.utils.constants.residue_constants import (
    unknown_restype_idx,
)
from modelgenerator.structure_tokenizer.utils.shape_utils import (
    stack_variable_length_tensors,
)
from modelgenerator.structure_tokenizer.utils.types import PathLike

logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    def __init__(
        self,
        name: str,  # name of the dataset for the logs etc.
        registry_path: PathLike,
        folder_path: PathLike,  # must point to the parent folder of the data (structures, proteomes, ...)
        max_nb_res: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.name = name
        self.registry_path = Path(registry_path)
        self.folder_path = Path(folder_path)
        self.max_nb_res = max_nb_res
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(self.seed + epoch)

    def protein_to_input_crop(
        self, protein: Protein
    ) -> dict[str, str | torch.Tensor | None]:
        input = protein.to_torch_input()
        atom_positions = input["atom_positions"]
        aatype = input["aatype"]
        atom_mask = input["atom_mask"]
        residue_index = input["residue_index"]

        if self.max_nb_res is not None and len(atom_positions) > self.max_nb_res:
            # sequential cropping
            start_idx = self.rng.integers(
                low=0, high=max(len(atom_positions) - self.max_nb_res, 0), endpoint=True
            )
            end_idx = start_idx + self.max_nb_res
            atom_positions = atom_positions[start_idx:end_idx, ...]
            residue_index = residue_index[start_idx:end_idx]
            aatype = aatype[start_idx:end_idx]
            atom_mask = atom_mask[start_idx:end_idx]

        return {
            "id": input["id"],
            "entity_id": input["entity_id"],
            "chain_id": input["chain_id"],
            "resolution": input["resolution"],
            "atom_positions": atom_positions,
            "aatype": aatype,
            "atom_mask": atom_mask,
            "residue_index": residue_index,
        }

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor | str | None]],
    ) -> dict[str, torch.Tensor | list[str | None]]:
        batch_ids = [b["id"] for b in batch]
        batch_entity_ids = [b["entity_id"] for b in batch]
        batch_chain_ids = [b["chain_id"] for b in batch]
        batch_resolution = torch.stack([b["resolution"] for b in batch])
        batch_aatype = stack_variable_length_tensors(
            sequences=[b["aatype"] for b in batch],
            constant_value=unknown_restype_idx,
            return_mask=False,
        )
        batch_atom_positions, attention_mask = stack_variable_length_tensors(
            sequences=[b["atom_positions"] for b in batch],
            constant_value=0.0,
            return_mask=True,
        )
        attention_mask = attention_mask[..., 0, 0]
        batch_atom_masks = stack_variable_length_tensors(
            sequences=[b["atom_mask"] for b in batch],
            constant_value=0,
            return_mask=False,
        )
        batch_residue_index = stack_variable_length_tensors(
            sequences=[b["residue_index"] for b in batch],
            constant_value=0,
            return_mask=False,
        )

        return {
            "id": batch_ids,
            "entity_id": batch_entity_ids,
            "chain_id": batch_chain_ids,
            "resolution": batch_resolution,
            "atom_positions": batch_atom_positions,
            "atom_masks": batch_atom_masks,
            "aatype": batch_aatype,
            "residue_index": batch_residue_index,
            "attention_mask": attention_mask,
        }


class ProteinCSVParquetDataset(ProteinDataset):
    def __init__(
        self,
        name: str,  # name of the dataset for logs
        registry_path: PathLike,
        folder_path: PathLike,  # must point to the parent folder of the data, structures, proteomes, etc.
        max_nb_res: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            registry_path=registry_path,
            folder_path=folder_path,
            max_nb_res=max_nb_res,
        )
        assert self.registry_path.suffix in [".csv", ".parquet"]
        self.proteins_df = self._read_registry()
        assert len(self.proteins_df) > 0

    def _read_registry(self) -> pd.DataFrame:
        match self.registry_path.suffix:
            case ".parquet":
                df = pd.read_parquet(str(self.registry_path)).reset_index(drop=True)
            case ".csv":
                df = pd.read_csv(str(self.registry_path)).reset_index(drop=True)
            case _:
                raise ValueError(f"Invalid csv parquet file: {self.registry_path.name}")
        return df

    @classmethod
    def from_config(
        cls, protein_dataset_config: ProteinDatasetConfig
    ) -> "ProteinCSVParquetDataset":
        return cls(
            name=protein_dataset_config.name,
            registry_path=protein_dataset_config.registry_path,
            folder_path=protein_dataset_config.folder_path,
            max_nb_res=protein_dataset_config.max_nb_res,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | None]:
        row = self.proteins_df.iloc[index]
        filename = row["filename"]
        chain = row["chain"] if not pd.isna(row["chain"]) else "nan"
        match Path(filename).suffix:
            case ".cif.gz" | ".cif":
                entity = row["entity"]
                protein = Protein.from_cif_file_path(
                    cif_file_path=self.folder_path / filename,
                    entity_id=entity,
                    chain_id=chain,
                )
            case ".ent.gz" | ".pdb":
                protein = Protein.from_pdb_file_path(
                    pdb_file_path=self.folder_path / filename,
                    chain_id=chain,
                )
            case _:
                raise ValueError(f"{Path(filename).suffix} is not supported.")

        return self.protein_to_input_crop(protein=protein)

    def __len__(self) -> int:
        return len(self.proteins_df)
