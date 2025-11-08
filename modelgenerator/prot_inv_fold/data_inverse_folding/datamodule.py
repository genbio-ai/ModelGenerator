import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import lightning.pytorch as pl
import json
import pickle

from typing import Union, Optional
from pathlib import Path

from .dataset import ProteinInverseFoldingDataset

from modelgenerator.data import DataInterface


class ProteinInverseFoldingDataModule(DataInterface):
    def __init__(
        self,
        path,
        embed_dim: int = 512,
        batch_size: int = 1,
        init_masking_rate: float = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        overfitting_run: bool = False,
        overfit_on_sample_count: int = 50,
        **kwargs,
    ):
        if self.__class__ is ProteinInverseFoldingDataModule:
            self.save_hyperparameters()

        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.init_masking_rate = init_masking_rate
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.overfitting_run = overfitting_run
        self.overfit_on_sample_count = overfit_on_sample_count

        self.test_only = True

        super().__init__(path=path, **kwargs)

        self.chain_set_map_fullpath = self.path + "chain_set_map.pkl"
        self.chain_set_splits_json_fullpath = self.path + "chain_set_splits.json"
        with open(self.chain_set_map_fullpath, "rb") as f:
            self.chain_set_map = pickle.load(f)

        with open(self.chain_set_splits_json_fullpath, "r") as f:
            self.dataset_splits = json.load(f)

    def setup(self, stage: Optional[str] = None):
        ## for only sampling/inference, do not load the train set (which wastes time).
        self.stage = stage
        if stage == "test":
            self.test_only = True

        self.train_dataset = ProteinInverseFoldingDataset(
            split="train", parent_datamodule=self
        )
        print(
            "[INFO]\tTrain dataset generated! Size=",
            (self.train_dataset.dataset_length),
        )

        self.val_dataset = ProteinInverseFoldingDataset(
            split="val", parent_datamodule=self
        )
        print(
            "[INFO]\tValidation dataset generated! Size=",
            (self.val_dataset.dataset_length),
        )

        self.test_dataset = ProteinInverseFoldingDataset(
            split="test", parent_datamodule=self
        )
        print(
            "[INFO]\tTest dataset generated! Size=", (self.test_dataset.dataset_length)
        )

        # if len(self.train_dataset) in {len(self.test_dataset), len(self.val_dataset)}:
        #     print(
        #         "\n[WARNING] Running overfitting experiment for debugging purposes only.\n\n"
        #     )

        print("[INFO]\tRunning training dataloader sanity check...")
        a = self.train_dataset.__getitem__(0)
        b = self.train_dataset.__getitem__(len(self.train_dataset) - 1)
        assert (a[-1].shape) == (b[-1].shape)

        print("self.train_dataset.max_seq_len =", self.train_dataset.max_seq_len)

        print("\n[INFO]\tAll datasets generated and sanity-checked!\n")

    def train_dataloader(self):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, shuffle=False
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,  ## NOTE: we shuffle data in the dataset.__getitem__() function to control "low discrepancy sampling". Ref: MDLM, Appendix C.3 Low discrepancy sampler.
            sampler=train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  ## TODO: setting it to 1 for now for easy sampling. later I will do batch sampling.
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  ## TODO: setting it to 1 for now for easy sampling. later I will do batch sampling.
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
