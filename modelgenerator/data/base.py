import os
import datasets
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from datasets import load_dataset
from docstring_inheritance import GoogleDocstringInheritanceInitMeta


class KFoldMixin:
    """Provides methods for splitting datasets into k-folds for cross-validation"""

    def __init__(self):
        self.cv_splits = None

    def get_split_by_fold_id(
        self, train_dataset, val_dataset, test_dataset, fold_id, val_idx_offset=1
    ):
        """Split the dataset into training, validation, and test sets based on the fold id for test"""
        if self.cv_num_folds <= 1:
            return train_dataset, val_dataset, test_dataset
        if len(val_dataset) or len(test_dataset):
            rank_zero_warn(
                "Redundant val or test splits are not expected and will be ignored during training. "
                "Disable this warning by setting {test,val}_split_size=0 and {test,val}_split_name=None"
            )
        if self.cv_fold_id_col is not None:
            splits = self.read_kfold_split(train_dataset)
        else:
            splits = self.generate_kfold_split(len(train_dataset), self.cv_num_folds)
        test_idx = splits[fold_id]
        val_idx = (
            splits[(fold_id + val_idx_offset) % self.cv_num_folds]
            if self.cv_enable_val_fold
            else []
        )
        if not self.cv_enable_val_fold and self.cv_replace_val_fold_as_test_fold:
            val_idx = test_idx
        train_idx = list(set(range(len(train_dataset))) - set(test_idx) - set(val_idx))
        return (
            train_dataset.select(train_idx),
            train_dataset.select(val_idx),
            train_dataset.select(test_idx),
        )

    def generate_kfold_split(
        self, num_samples: int, num_folds: int, shuffle: bool = True
    ):
        """Randomly split n_samples into n_splits folds and return list of fold_idx

        Args:
            num_samples (int): Number of samples in the data.
            num_folds (Optional[int]): Number of folds for cross validation, must be > 1. Defaults to 10.
            shuffle (Optional[bool]): Whether to shuffle the data before splitting into batches. Defaults to True.

        Returns:
            list of list containing indices for each fold
        """
        if self.cv_splits is not None:
            return self.cv_splits
        idx = np.arange(num_samples)
        if shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(idx)
        fold_samples = num_samples // num_folds
        kfold_split_idx = []
        for k in range(num_folds - 1):
            kfold_split_idx.append(
                idx[k * fold_samples : (k + 1) * fold_samples].tolist()
            )
        kfold_split_idx.append(idx[(k + 1) * fold_samples :].tolist())
        self.cv_splits = kfold_split_idx
        return kfold_split_idx

    def read_kfold_split(self, dataset: datasets.Dataset):
        """Read the fold ids from a pre-split dataset and return list of fold_idx"""
        fold_ids = sorted(dataset.unique(self.cv_fold_id_col))
        if list(range(self.cv_num_folds)) != fold_ids:
            raise ValueError(f"Fold ids {fold_ids} do not match the expected range")
        kfold_split_idx = []
        for fold_id in fold_ids:
            kfold_split_idx.append(
                np.where(np.array(dataset[self.cv_fold_id_col], dtype=int) == fold_id)[
                    0
                ]
            )
        self.cv_splits = kfold_split_idx
        return kfold_split_idx


class HFDatasetLoaderMixin:
    """Provides methods for loading datasets using the Huggingface datasets library."""

    def load_dataset(self, **kwargs) -> Tuple[datasets.Dataset]:
        split_names = [
            self.train_split_name,
            self.valid_split_name,
            self.test_split_name,
        ]
        data_files = {}
        if self.train_split_files:
            data_files["train"] = self.train_split_files
        if self.valid_split_files:
            data_files["valid"] = self.valid_split_files
        if self.test_split_files:
            data_files["test"] = self.test_split_files
        splits = ()
        for split_name in split_names:
            if split_name is None:
                splits += (None,)
            else:
                try:
                    ds = load_dataset(
                        self.path,
                        data_files=None if not data_files else data_files,
                        name=self.config_name,
                        streaming=False,
                        split=split_name,
                        **kwargs,
                    )
                except ValueError as e:
                    rank_zero_warn(
                        f"Could not load split='{split_name}': {e}. Setting to None. You may ignore if you are not using split '{split_name}'."
                    )
                    ds = None
                splits += (ds,)
        return splits

    def load_and_split_dataset(self, **kwargs) -> Tuple[datasets.Dataset]:
        train_dataset, valid_dataset, test_dataset = self.load_dataset(**kwargs)
        if train_dataset is not None:
            if test_dataset is None and self.test_split_size > 0:
                rank_zero_info(
                    f"> Randomly split {self.test_split_size} of train for testing, Random seed: {self.random_seed}"
                )
                train_test_split = train_dataset.train_test_split(
                    test_size=self.test_split_size, seed=self.random_seed
                )
                train_dataset = train_test_split["train"]
                test_dataset = train_test_split["test"]
            if valid_dataset is None and self.valid_split_size > 0:
                rank_zero_info(
                    f"> Randomly split {self.valid_split_size} of train for validation. Random seed: {self.random_seed}"
                )
                train_test_split = train_dataset.train_test_split(
                    test_size=self.valid_split_size, seed=self.random_seed
                )
                train_dataset = train_test_split["train"]
                valid_dataset = train_test_split["test"]
        first_non_empty = train_dataset or valid_dataset or test_dataset
        if first_non_empty is None:
            raise ValueError("All splits are empty")
        # return empty datasets instead of None for easier handling
        if train_dataset is None:
            train_dataset = datasets.Dataset.from_dict(
                {k: [] for k in first_non_empty.column_names}
            )
        if valid_dataset is None:
            valid_dataset = datasets.Dataset.from_dict(
                {k: [] for k in first_non_empty.column_names}
            )
        if test_dataset is None:
            test_dataset = datasets.Dataset.from_dict(
                {k: [] for k in first_non_empty.column_names}
            )
        return train_dataset, valid_dataset, test_dataset


class DataInterface(pl.LightningDataModule, KFoldMixin, metaclass=GoogleDocstringInheritanceInitMeta):
    """Base class for all data modules in this project. Handles the boilerplate of setting up data loaders.

    Note:
        Subclasses must implement the setup method.
        All datasets should return a dictionary of data items.
        To use HF loading, add the HFDatasetLoaderMixin.
        For any task-specific behaviors, implement transformations using `torch.utils.data.Dataset` objects.
        See [MLM](./#modelgenerator.data.MLMDataModule) for an example.

    Args:
        path: Path to the dataset, can be (1) a local path to a data folder or (2) a Huggingface dataset identifier
        config_name: The name of the HF dataset configuration.
            Affects how the dataset is loaded.
        train_split_name: The name of the training split.
        test_split_name: The name of the test split. Also used for `mgen predict`.
        valid_split_name: The name of the validation split.
        train_split_files: Create a split called "train" from these files.
            Not used unless referenced by the name "train" in one of the split_name arguments.
        test_split_files: Create a split called "test" from these files.
            Not used unless referenced by the name "test" in one of the split_name arguments.
            Also used for `mgen predict`.
        valid_split_files: Create a split called "valid" from these files.
            Not used unless referenced by the name "valid" in one of the split_name arguments.
        test_split_size: The size of the test split.
           If test_split_name is None, creates a test split of this size from the training split.
        valid_split_size: The size of the validation split.
           If valid_split_name is None, creates a validation split of this size from the training split.
        random_seed: The random seed to use for splitting the data.
        extra_reader_kwargs: Extra kwargs for dataset readers.
        batch_size: The batch size.
        shuffle: Whether to shuffle the data.
        sampler: The sampler to use.
        num_workers: The number of workers to use for data loading.
        collate_fn: The function to use for collating data.
        pin_memory: Whether to pin memory.
        persistent_workers: Whether to use persistent workers.
        cv_num_folds: The number of cross-validation folds, disables cv when <= 1.
        cv_test_fold_id: The fold id to use for cross-validation evaluation.
        cv_enable_val_fold: Whether to enable a validation fold.
        cv_replace_val_fold_as_test_fold: Replace validation fold with test fold. Only used when cv_enable_val_fold is False.
        cv_fold_id_col: The column name containing the fold id from a pre-split dataset. Setting to None to enable automatic splitting.
        cv_val_offset: The offset applied to cv_test_fold_id to determine val_fold_id.
    """

    def __init__(
        self,
        path: str,
        config_name: Optional[str] = None,
        train_split_name: Optional[str] = "train",
        test_split_name: Optional[str] = "test",
        valid_split_name: Optional[str] = None,
        train_split_files: Optional[Union[str, List[str]]] = None,
        test_split_files: Optional[Union[str, List[str]]] = None,
        valid_split_files: Optional[Union[str, List[str]]] = None,
        test_split_size: float = 0.2,
        valid_split_size: float = 0.1,
        random_seed: int = 42,
        extra_reader_kwargs: Optional[dict] = None,
        batch_size: int = 128,
        shuffle: bool = True,
        sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[callable] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        # TODO: cv params will be deprecated and will be handled by trainer directly
        cv_num_folds: int = 1,
        cv_test_fold_id: int = 0,
        cv_enable_val_fold: bool = True,
        cv_replace_val_fold_as_test_fold: bool = False,
        cv_fold_id_col: Optional[str] = None,
        cv_val_offset: int = 1,
    ):
        super().__init__()
        if os.path.isfile(path):
            raise ValueError(
                "Path must be a directory or a Huggingface dataset repo. "
                "If you want to pass only one file, set the path to the directory "
                "containing the file and set `*_split_files` to `[filename]`."
            )
        self.path = path
        self.config_name = config_name
        self.train_split_name = train_split_name
        self.test_split_name = test_split_name
        self.valid_split_name = valid_split_name
        self.train_split_files = train_split_files
        self.test_split_files = test_split_files
        self.valid_split_files = valid_split_files
        self.test_split_size = test_split_size
        self.valid_split_size = valid_split_size
        self.random_seed = random_seed
        self.extra_reader_kwargs = extra_reader_kwargs or {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cv_num_folds = cv_num_folds
        self.cv_test_fold_id = cv_test_fold_id
        self.cv_enable_val_fold = cv_enable_val_fold
        self.cv_replace_val_fold_as_test_fold = cv_replace_val_fold_as_test_fold
        self.cv_fold_id_col = cv_fold_id_col
        self.cv_val_offset = cv_val_offset

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module. This method should be overridden by subclasses.

        Args:
            stage (Optional[str], optional): training, validation, or test if these need to be setup separately. Defaults to None.
        """
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader

        Returns:
            DataLoader: The training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader

        Returns:
            DataLoader: The validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader

        Returns:
            DataLoader: The test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get the dataloader for predictions for the test set

        Returns:
            DataLoader: The predict data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
