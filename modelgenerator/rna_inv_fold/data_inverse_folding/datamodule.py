import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from .dataset import InverseFoldingDataset

from dataclasses import dataclass


@dataclass
class RNAInvFoldConfig:
    # init_params: str = None  # Path to the '.pt' file containing model weights
    # pretrained_LM_location: str = None
    seed: int = None
    embed_dim: int = 512
    num_blocks: int = 1  # 3
    drop_prob: float = 0.1
    add_str_enc_logits: bool = False
    lm_logit_update_weight: float = 1.0
    inverse_folding: bool = True
    multihead_attn_predhead: bool = False
    unconditional: bool = False

    test_only: bool = False

    # diffusion
    num_denoise_steps: int = 1  # best score was with 3
    diffusion_verbose: int = 1
    init_masking_rate: float = 0.15
    num_full_seq_mask_epoch: int = 0
    sample_seq: bool = False
    sampling_temperature: float = 0.1

    lm_modality: str = "rna"

    tokenizer_version: str = "v2"  ##  NOTE: legacy rna-IF-diffusion uses v1

    # class OptimizerConfig:
    class_path: str = "torch.optim.AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01

    # class OverfittingConfig:
    overfitting_run: bool = False
    overfit_on_sample_count: int = 50

    # class LoRAConfig:
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # class SchedulerConfig:
    lr_scheduler_total_iters: int = 500
    lr_scheduler_end_factor: float = 0.1

    # class DataloaderConfig:
    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = True


default_config = RNAInvFoldConfig()


class RNAInverseFoldingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        custom_invfold_config: RNAInvFoldConfig = default_config,
    ):
        super().__init__()

        self.path = path
        self.args = custom_invfold_config

        self.embed_dim = self.args.embed_dim
        self.batch_size = 1  # self.args.batch_size ## batch_size=1 for inference
        self.num_workers = self.args.num_workers
        self.pin_memory = False  # args.pin_memory
        self.persistent_workers = True  # args.persistent_workers
        self.init_masking_rate = self.args.init_masking_rate

    def setup(self, stage: Optional[str] = None):
        ## Note: for only sampling/inference, do not load the train set (which wastes time).
        self.stage = stage
        if stage == "test":
            self.args.test_only = True
        else:
            raise NotImplementedError()

        # self.train_dataset  = InverseFoldingDataset(self.path, split='train', batch_size=self.batch_size, embed_dim=self.embed_dim,
        #                                             init_masking_rate=self.init_masking_rate)
        # print('[INFO]\tTrain dataset generated!')
        # self.val_dataset    = InverseFoldingDataset(self.path, split='val', batch_size=self.batch_size, embed_dim=self.embed_dim,
        #                                             init_masking_rate=self.init_masking_rate)
        # print('[INFO]\tValidation dataset generated!')
        self.test_dataset = InverseFoldingDataset(
            self.path,
            split="test",
            batch_size=self.batch_size,
            embed_dim=self.embed_dim,
            init_masking_rate=self.init_masking_rate,
        )
        print("[INFO]\tTest dataset generated!")

        ### NOTE: for inference only, we set self.train_dataset and self.val_dataset placeholders
        self.train_dataset = self.val_dataset = self.test_dataset

        if len(self.train_dataset) == len(self.test_dataset):
            print(
                "\n\n[WARNING] Running overfitting test for debugging purposes only.\n"
            )

        print("[INFO]\tRunning dataloader sanity check...")
        a = self.train_dataset.__getitem__(0)
        b = self.train_dataset.__getitem__(len(self.train_dataset) - 1)
        assert (a[2].shape) == (b[2].shape)

        print("self.train_dataset.max_seq_len", self.train_dataset.max_seq_len)

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
            batch_size=1,  # self.batch_size, ## TODO: setting it to 1 for now for easy sampling. later I will do batch sampling.
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # self.batch_size, ## TODO: setting it to 1 for now for easy sampling. later I will do batch sampling.
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
