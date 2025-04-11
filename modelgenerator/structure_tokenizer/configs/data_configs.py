import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from modelgenerator.structure_tokenizer.utils.types import PathLike

logger = logging.getLogger(__name__)


REPO_PATH = Path(__file__).parents[3]


@dataclass
class ProteinDatasetConfig:
    name: str  # name of the dataset
    registry_path: PathLike
    folder_path: PathLike
    max_nb_res: int | None
    batch_size: int
    seed: int = 0

    def __post_init__(self):
        self.registry_path = Path(self.registry_path)
        self.folder_path = Path(self.folder_path)
        #
        if not self.registry_path.is_absolute():
            self.registry_path = REPO_PATH / self.registry_path
        if not self.folder_path.is_absolute():
            self.folder_path = REPO_PATH / self.folder_path
        if self.max_nb_res is not None:
            logger.info(
                f"{self.name} dataset has {self.max_nb_res} max residues. Random cropping is activated."
            )


@dataclass
class ProteinDataConfig:
    proteins_datasets_configs: list[ProteinDatasetConfig]
    num_workers: int  # num_workers should be chosen per GPU, https://github.com/Lightning-AI/pytorch-lightning/issues/18149
    seed: int = 0

    def __post_init__(self):
        for protein_dataset_config in self.proteins_datasets_configs:
            protein_dataset_config.seed = self.seed

    @cached_property
    def configs(self) -> dict[str, ProteinDatasetConfig]:
        return {config.name: config for config in self.proteins_datasets_configs}

    @cached_property
    def dataloader_idx_to_name(self) -> dict[int, str]:
        return {idx: config.name for idx, config in enumerate(self.configs.values())}


@dataclass
class StructTokensDatasetConfig:
    name: str  # name of the dataset
    struct_tokens_path: PathLike
    codebook_path: PathLike
    batch_size: int

    def __post_init__(self):
        self.struct_tokens_path = Path(self.struct_tokens_path)
        self.codebook_path = Path(self.codebook_path)


@dataclass
class StructTokensDataConfig:
    struct_tokens_datasets_configs: list[StructTokensDatasetConfig]
    num_workers: int  # num_workers should be chosen per GPU, https://github.com/Lightning-AI/pytorch-lightning/issues/18149

    @cached_property
    def configs(self) -> dict[str, StructTokensDatasetConfig]:
        return {config.name: config for config in self.struct_tokens_datasets_configs}

    @cached_property
    def dataloader_idx_to_name(self) -> dict[int, str]:
        return {idx: config.name for idx, config in enumerate(self.configs.values())}
