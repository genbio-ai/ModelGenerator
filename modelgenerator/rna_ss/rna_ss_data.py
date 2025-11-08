from modelgenerator.data import *
from typing import Union, Optional
from pathlib import Path

from modelgenerator.rna_ss.rna_ss_utils import parse_sec_struct_file


class RNASSPairwiseTokenClassification(DataInterface):
    """A PyTorch Lightning DataModule for pairwise token classification in RNA secondary structure prediction.
    This module is designed to facilitate data loading, preprocessing, and batch preparation for supported datasets.

    Attributes:
        TRAIN_DIR_NAME (str): Name of the training data directory. Default is "train".
        VAL_DIR_NAME (str): Name of the validation data directory. Default is "valid".
        TEST_DIR_NAME (str): Name of the testing data directory. Default is "test".
        SUPPORTED_DATASETS (list[str]): List of RNA datasets supported by the module.

    Args:
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 0.
        pin_memory (bool, optional): If True, DataLoader will copy Tensors into CUDA pinned memory before returning them. Default is True.
        persistent_workers (bool, optional): If True, DataLoader workers are persistent across epochs. Default is True.
        min_seq_len (int, optional): Minimum sequence length for filtering input samples. Default is 0.
        max_seq_len (int, optional): Maximum sequence length for filtering input samples. Default is 999_999_999.
        dataset (str, optional): The name of the RNA dataset to use. Must be one of SUPPORTED_DATASETS. Default is "bpRNA".
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/data/downstream/secondary_structure/datamodule.py

    TRAIN_DIR_NAME = "train"
    VAL_DIR_NAME = "valid"
    TEST_DIR_NAME = "test"

    SUPPORTED_DATASETS = [
        "bpRNA",
        "archiveII_5s",
        "archiveII_16s",
        "archiveII_23s",
        "archiveII_grp1",
        "archiveII_srp",
        "archiveII_telomerase",
        "archiveII_RNaseP",
        "archiveII_tmRNA",
        "archiveII_tRNA",
    ]

    def __init__(
        self,
        path: str,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        min_seq_len: int = 0,
        max_seq_len: int = 999_999_999,
        dataset: str = "bpRNA",
        *args,
        **kwargs,
    ):
        super().__init__(path, *args, **kwargs)

        self.dataset = dataset
        self.data_root = Path(self.path)

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.batch_size = 1

        if dataset == "bpRNA":
            self.train_dir = f"bpRNA/{self.TRAIN_DIR_NAME}"
            self.val_dir = f"bpRNA/{self.VAL_DIR_NAME}"
            self.test_dir = f"bpRNA/{self.TEST_DIR_NAME}"
        elif dataset in self.SUPPORTED_DATASETS and dataset.startswith("archiveII"):
            test_rna_family = dataset.split("_")[-1]

            self.train_dir = f"archiveII/{test_rna_family}/{self.TRAIN_DIR_NAME}"
            self.val_dir = f"archiveII/{test_rna_family}/{self.VAL_DIR_NAME}"
            self.test_dir = f"archiveII/{test_rna_family}/{self.TEST_DIR_NAME}"
        else:
            # Use user custom dataset
            self.train_dir = f"{self.path}/{self.dataset}/{self.TRAIN_DIR_NAME}"
            self.val_dir = f"{self.path}/{self.dataset}/{self.VAL_DIR_NAME}"
            self.test_dir = f"{self.path}/{self.dataset}/{self.TEST_DIR_NAME}"

    def prepare_dataset(
        self,
        data_dir: Union[str, Path],
        min_seq_len: int = 0,
        max_seq_len: int = 999_999_999,
        ss_file_extensions: list[str] = ["ct", "bpseq", "st"],
    ):
        # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/data/downstream/secondary_structure/datamodule.py
        data_dir = Path(data_dir)

        # Collect secondary structure file paths
        ss_ids = []
        sequences = []
        sec_structures = []
        for ss_file_ext in ss_file_extensions:
            for ss_file_path in list(data_dir.glob(f"**/*.{ss_file_ext}")):
                seq, sec_struct = parse_sec_struct_file(ss_file_path)

                if len(seq) >= min_seq_len and len(seq) <= max_seq_len:
                    seq = seq.upper().replace("T", "U")
                    seq = re.sub(r"[^AUCG]", "N", seq)

                    ss_ids.append(ss_file_path.stem)
                    sequences.append(seq)
                    sec_structures.append(sec_struct)

        return ss_ids, sequences, sec_structures

    def setup(self, stage: Optional[str] = None):
        tmp = self.prepare_dataset(
            data_dir=self.data_root / self.train_dir,
            min_seq_len=self.min_seq_len,
            max_seq_len=self.max_seq_len,
        )
        self.train_dataset = AnyDataset(
            ss_ids=tmp[0], sequences=tmp[1], labels=tmp[2]
        )

        tmp = self.prepare_dataset(
            data_dir=self.data_root / self.val_dir,
            min_seq_len=self.min_seq_len,
            max_seq_len=self.max_seq_len,
        )
        self.val_dataset = AnyDataset(
            ss_ids=tmp[0], sequences=tmp[1], labels=tmp[2]
        )

        tmp = self.prepare_dataset(
            data_dir=self.data_root / self.test_dir,
            min_seq_len=self.min_seq_len,
            max_seq_len=self.max_seq_len,
        )
        self.test_dataset = AnyDataset(
            ss_ids=tmp[0], sequences=tmp[1], labels=tmp[2]
        )

    @property
    def provided_columns(self) -> List[str]:
        """List of columns provided by the dataset."""
        return ["ss_ids", "sequences", "labels"]