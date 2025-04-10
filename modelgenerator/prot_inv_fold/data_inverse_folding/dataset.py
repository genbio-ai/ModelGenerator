import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch.distributions.categorical import Categorical

from typing import Union
from pathlib import Path


def getPositionEncoding(seq_len, d, n=10000):

    print(
        "[WARNING] (to self) You are using original sinusoidal positinal encoding. Use ROPE instead!  :) "
    )

    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


def replace_characters_at_indices(s, indices, replacement_char):
    s_list = list(s)
    for index in indices:
        if 0 <= index < len(s_list):
            s_list[index] = replacement_char
    return "".join(s_list)


def compute_entropy(probs=None, logits=None):
    if probs is None:
        probs = F.softmax(logits, dim=-1)  # B x N x D
    log_probs = torch.log(probs)  # B x N x D
    entropy = -torch.sum(probs * log_probs, dim=-1)  # B x N
    return entropy


def compute_probab(logits):
    probs = F.softmax(logits, dim=-1).max(-1).values  # B x N
    return probs


class ProteinInverseFoldingDataset(Dataset):
    def __init__(
        self,
        split,
        parent_datamodule,
        **kwargs,
    ):
        self.split = split
        self.set_hparams_custom(parent_datamodule)
        # self.save_hyperparameters()

        super().__init__(**kwargs)

        if self.test_only:
            self.split = split = (
                "test"  ## NOTE: when testing only, load only test split for all dataloaders. for speed up.
            )
        if self.overfitting_run:
            self.split = split = (
                "val"  ## NOTE: For overfitting run, we load the validation split for all splits since it's a relatively small file. For faster dataloading to RAM. Note we are not changing the self.split.
            )

        # self.chain_set_map = chain_set_map      ## NOTE: keys: 'seq', 'coords', 'num_chains', 'name', 'CATH'

        if split == "val":
            self.all_pdb_ids = self.dataset_splits["validation"]
        else:
            self.all_pdb_ids = self.dataset_splits[split]
        self.dataset_length = len(self.all_pdb_ids)
        self.allowed_samples = list(range(self.dataset_length))
        self.len_allowed_samples = self.dataset_length
        self.max_seq_len = (
            500 + 1
        )  ## TODO: hardcoded based on CATH-4.2 dataset. need to change later.

        self.positional_encoding = getPositionEncoding(
            seq_len=self.max_seq_len, d=self.embed_dim * 2, n=10_000
        )
        N = self.batch_size

        self.interval_partitions = [
            [(i - 1) / N, i / N] for i in range(1, N + 1)
        ]  ## NOTE: Ref: MDLM, Appendix C.3 Low discrepancy sampler.
        self.interval_partitions[0][
            0
        ] = 1e-3  ## NOTE: Intentionally avoiding 0. Replacing 0 with a small value (1e-3).

        self.weighted_loss = False

    def set_hparams_custom(self, parent_datamodule):
        self.test_only = parent_datamodule.test_only
        self.chain_set_map = parent_datamodule.chain_set_map
        self.dataset_splits = parent_datamodule.dataset_splits
        self.batch_size = parent_datamodule.batch_size
        self.embed_dim = parent_datamodule.embed_dim
        self.overfitting_run = parent_datamodule.overfitting_run
        self.overfit_on_sample_count = parent_datamodule.overfit_on_sample_count

    def __len__(self):
        if self.overfitting_run:
            return self.overfit_on_sample_count

        if self.split == "train":
            return self.dataset_length // 8  # // 4

        return self.dataset_length

    def get_posterior_weight(self, interval_partition):
        ##@ Simplified for log-linear noising schedule. Ref: MDLM paper, noise schedulers.
        if self.weighted_loss:
            t = random.uniform(
                interval_partition[0], interval_partition[1]
            )  ## NOTE: Sampling diffusion time-step t from (a, b). Ref: Ref: MDLM, Appendix C.3 Low discrepancy sampler.
            assert 0 < t < 1
        else:
            t = 1
        posterior_weight = 1 / t  # / 1e3
        masking_rate = t

        return posterior_weight, masking_rate

    def __getitem__(self, idx):
        interval_idx = (
            idx % self.batch_size
        )  ## NOTE: expecting indices are sequential, to ensure LDS. the randum shuffle should be is turned on at both DDP and also the dataloader level. we will randomize here.
        interval_partition = self.interval_partitions[
            interval_idx
        ]  ## NOTE: Ref: MDLM, Appendix C.3 Low discrepancy sampler (LDS).
        if (self.split == "train") or self.overfitting_run:
            ##@ Replace idx with a random index from the "allowed_samples"
            if (self.split == "train") and self.overfitting_run:
                idx = np.random.randint(low=0, high=self.overfit_on_sample_count)
            else:
                idx = np.random.randint(low=0, high=self.len_allowed_samples)
            idx = self.allowed_samples[idx]

        entry = self.chain_set_map[
            self.all_pdb_ids[idx]
        ]  ## NOTE: entry-- a dictionary with keys: 'seq', 'coords', 'num_chains', 'name', 'CATH'
        coords = [entry["coords"][key] for key in entry["coords"]]
        coords = np.asarray(coords, dtype=np.float32)
        coords = np.transpose(coords, (1, 0, 2))
        valid_positions = np.where(~np.isnan(coords.sum((1, 2))))
        coords = coords[valid_positions]
        coord_mask = ~np.isnan(coords.sum((1, 2)))
        coords = np.nan_to_num(coords)
        entry_name = entry["name"]
        seq_input = seq_target = entry["seq"]
        seq_input = seq_target = "".join([entry["seq"][i] for i in valid_positions[0]])
        seq_length = len(seq_input)
        assert (
            seq_length <= self.max_seq_len - 1
        ), f"Should be {seq_length} <= {self.max_seq_len} - 1"
        positional_encoding = self.positional_encoding  ## NOTE: tensor

        if self.split == "train":  # and (not self.overfitting_run):# and False:
            posterior_weight, masking_rate = self.get_posterior_weight(
                interval_partition
            )
        else:
            # raise NotImplementedError
            masking_rate = 1  # self.init_masking_rate                ## NOTE: for inference, start with a fully masked
            posterior_weight = 1 / masking_rate  # / 1e3
        num_mask_tokens = max(
            1, int(seq_length * masking_rate)
        )  ## NOTE: at least one token has to be masked

        ##
        perm = torch.randperm(seq_length)
        input_mask_indices = perm[:num_mask_tokens]
        seq_input = replace_characters_at_indices(
            s=seq_input, indices=input_mask_indices, replacement_char="[MASK]"
        )  ## NOTE: string, masked sequence

        pad_mask = np.zeros((self.max_seq_len,))  ## NOTE: tensor
        pad_mask[seq_length + 1 :] = (
            1  ## TODO: we are taking all tokens, including the special tokens.
        )

        input_mask = torch.zeros((self.max_seq_len,))  ## NOTE: tensor
        input_mask.scatter_(0, input_mask_indices, 1)

        target_mask = torch.zeros((self.max_seq_len,))  ## NOTE: tensor
        if self.split in {"train"}:
            target_mask = (
                input_mask.clone()
            )  ## NOTE: during training, we just want to compute loss and accuracy for the "masked tokens" only (ref: MDLM).
        else:
            # target_mask[:seq_length] = 1                             ## NOTE: during inference/testing, we want to compute loss and accuracy on the whole sequence.
            target_mask[:seq_length] = torch.from_numpy(coord_mask)

        input_mask_indices = "\t".join(
            [str(x.item()) for x in input_mask_indices]
        )  ## NOTE: String. To avoid any automated tensor preprocessing by pytorch-lightning

        seq_length = torch.tensor(
            [seq_length]
        )  ## NOTE: tensor. returning seq_length as a pytorch tensor. we will need it later for sanity-checking.
        posterior_weight = torch.Tensor([posterior_weight])
        masking_rate = torch.Tensor([masking_rate])

        ##@ padding tensors to maximum length
        tmp = coords.copy()
        coords = np.zeros((self.max_seq_len, 4, 3))
        coords[:seq_length] = tmp  ## NOTE: tensor

        tmp = coord_mask.copy()
        coord_mask = np.zeros((self.max_seq_len,))
        coord_mask[:seq_length] = tmp

        return (
            seq_input,
            seq_target,
            input_mask_indices,
            positional_encoding,
            input_mask,
            target_mask,
            pad_mask,
            seq_length,
            posterior_weight,
            masking_rate,
            interval_partition,
            interval_idx,
            entry_name,
            coords,
            coord_mask,
        )
