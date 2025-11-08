import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch.distributions.categorical import Categorical
import os

from typing import Union
from pathlib import Path

"""
### [CLS] [BOS]  A  A  A [EOS] [SEP] [PAD] [PAD] [PAD] [PAD]
###   2     11   5  5  5   12    3
vocab_a2n_lm = {
    '[PAD]': 0, 
    '[MASK]': 1,
    '[CLS]': 2,
    '[SEP]': 3,
    '[UNK]': 4, ## there should be no unknown tokens. we must address that in the preprocessing steps. 
    'A': 5,
    'G': 6,
    'C': 7,
    'T': 8, 
    'U': 9,
    'N': 10, ## for inverse_folding we are ignoring non-standard nucleotides.
    '[BOS]': 11,
    '[EOS]': 12,
}
vocab_n2a_lm = {n:a for a,n in vocab_a2n_lm.items()}

vocab_grnade2lm = {
    0:5,    #A
    1:6,    #G
    2:7,    #C
    3:9,    #U
    4:1,    #[MASK]
}
vocab_lm2grnade = {n:a for a,n in vocab_grnade2lm.items()}
"""

# List of possible RNA nucleotides
vocab_a2n_grnade = {
    "A": 0,
    "G": 1,
    "C": 2,
    "U": 3,
    "[MASK]": 4,  # placeholder for missing/unknown nucleotides
    # '_': 4,  # placeholder for missing/unknown nucleotides
}
vocab_n2a_grnade = {n: a for a, n in vocab_a2n_grnade.items()}
mapping_tensor_grnade2lm = torch.tensor([5, 6, 7, 9, 1])
mapping_tensor_lm2grnade = torch.tensor(
    [-1000, 4, -1000, -1000, -1000, 0, 1, 2, -1000, 3]
)


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


class InverseFoldingDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        batch_size,
        embed_dim,
        init_masking_rate=0.15,
        # num_diffusion_steps_T=10,
    ):
        super().__init__()
        self.split = split

        logits_all, estimated_seq_by_gRNAde_all = torch.load(
            os.path.join(data_root, f"logits_samples__{split}.pt")
        )
        lm_labels_all, structure_encodings_all = torch.load(
            os.path.join(data_root, f"encodings__{split}.pt")
        )  ## load labels

        self.string_all = [
            "".join([vocab_n2a_grnade[t] for t in seq]) for seq in lm_labels_all
        ]
        self.lm_labels_all = lm_labels_all
        # self.estimated_seq_by_gRNAde_all = [[''.join([vocab_n2a_grnade[t] for t in sample]) for sample in seq_samples] for seq_samples in estimated_seq_by_gRNAde_all]
        self.estimated_seq_by_gRNAde_all = [
            "".join([vocab_n2a_grnade[t.argmax()] for t in seq.mean(0)])
            for seq in logits_all
        ]
        self.logits_all = [torch.from_numpy(l) for l in logits_all]
        self.probab_all = [compute_probab(logits=l) for l in self.logits_all]
        self.entropy_all = [compute_entropy(logits=l) for l in self.logits_all]
        self.structure_encodings_all = structure_encodings_all

        self.sequence_lengths = [len(seq) for seq in self.lm_labels_all]
        self.max_seq_len = np.max(self.sequence_lengths) + 4
        if self.split in {"train"}:
            ALLOWED_MAX_SEQ_LEN = 200  # self.max_seq_len # next: 200.  # 500
            ##@ (During training only) not taking any sequence longer than ALLOWED_MAX_SEQ_LEN, since Das et al. test sequences are always smaller than 200 tokens.
            self.max_seq_len = min(self.max_seq_len, ALLOWED_MAX_SEQ_LEN + 4)
            self.allowed_samples = [
                i
                for i in range(len(self.sequence_lengths))
                if self.sequence_lengths[i] <= self.max_seq_len - 4
            ]
            self.len_allowed_samples = len(self.allowed_samples)
            print(
                f"\n[INFO] In {self.split} set: total allowed sequences = {len(self.allowed_samples)}.\tMaximum sequence length = {self.max_seq_len - 4}.\n"
            )

        self.positional_encoding = getPositionEncoding(
            seq_len=self.max_seq_len, d=embed_dim * 2, n=10_000
        )

        self.dataset_length = len(self.logits_all)
        # self.gRNAde_sampling_size = len(self.logits_all[0])

        # if self.split == 'train':
        #     assert init_masking_rate > 0
        self.init_masking_rate = (
            init_masking_rate  ## NOTE: Update as needed. Not used during training.
        )

        self.conditional_sampling = True  ## NOTE: Set to True if we want to condition sampling on the prediction confidence of gRNAde. In that case, low confidence predicted tokens will be masked.

        # self.num_diffusion_steps_T = 1_000          ## NOTE: Virtually infinity. TODO: Hardcoded. Create "constants.py" file and define there.
        self.num_diffusion_steps_T = 50  ## NOTE: Virtually infinity. TODO: Hardcoded. Create "constants.py" file and define there.
        self.batch_size = N = batch_size
        # self.interval_partitions = [ [(i - 1) / N , i / N] for i in range(1, int(N+1) ]   ## NOTE: Ref: MDLM, Appendix C.3 Low discrepancy sampler.
        self.interval_partitions = [
            [(i - 1) / N, i / N] for i in range(1, N + 1)
        ]  ## NOTE: Ref: MDLM, Appendix C.3 Low discrepancy sampler.
        self.interval_partitions[0][
            0
        ] = 1e-3  ## NOTE: Intentionally avoiding 0. Replacing 0 with a small value (1e-3).
        # print('self.interval_partitions:', self.interval_partitions)

        self.weighted_loss = False

    def __len__(self):
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
        interval_idx = idx % self.batch_size
        interval_partition = self.interval_partitions[
            interval_idx
        ]  ## NOTE: Ref: MDLM, Appendix C.3 Low discrepancy sampler.
        if self.split == "train":
            ##@ Replace idx with a random index from the "allowed_samples"
            idx = np.random.randint(low=0, high=self.len_allowed_samples)
            idx = self.allowed_samples[idx]
        seq_length = self.sequence_lengths[idx]
        assert seq_length <= self.max_seq_len - 4

        # sample_index = np.random.randint(low=0, high=self.gRNAde_sampling_size)
        # num_mask_tokens = np.random.randint(low=1, high=int((seq_length+1) * self.mask_ratio))    ## NOTE: at least one token will be masked, and at most the whole sequence.

        if self.split == "train":  # and False:
            posterior_weight, masking_rate = self.get_posterior_weight(
                interval_partition
            )
        else:
            masking_rate = self.init_masking_rate
            posterior_weight = 1 / masking_rate  # / 1e3
        num_mask_tokens = max(
            1, int(seq_length * masking_rate)
        )  ## NOTE: for inference, start with a predefined masking

        ##
        USE_gRNAde_OUTPUT_FOR_TRAINING = True
        if self.split == "train" and (not USE_gRNAde_OUTPUT_FOR_TRAINING):
            raise
            seq_input = self.string_all[idx]  ## NOTE: string, true sequence
        else:
            seq_input = self.estimated_seq_by_gRNAde_all[
                idx
            ]  ## NOTE: string, estimated sequence by gRNAde

        if self.split == "train" and (not USE_gRNAde_OUTPUT_FOR_TRAINING):
            raise
            # m = Categorical(logits=torch.ones((seq_length,)))               ## NOTE: uniform sampler
            # input_mask_indices = m.sample(sample_shape=(num_mask_tokens,))  ## NOTE: list
            perm = torch.randperm(seq_length)
            input_mask_indices = perm[:num_mask_tokens]
        else:  # if self.conditional_sampling and (self.split in {'test', 'val', 'train'}):
            # m = Categorical(probs=(1-self.probab_all[idx].mean(0))) ## NOTE: conditional sampler, conditioned on the confidence-level of gRNAde predictions
            input_mask_indices = (
                self.probab_all[idx].mean(0).argsort()[:num_mask_tokens]
            )

        seq_target = self.string_all[idx]  ## NOTE: string, true sequence
        target_ids = self.lm_labels_all[idx]  ## NOTE: tensor
        structure_encoding = self.structure_encodings_all[
            idx
        ]  ## NOTE: tensors. (n_nodes, d_s), (n_nodes, d_v, 3)
        structure_encoding, structure_encoding_vector = structure_encoding

        logits = self.logits_all[idx].mean(0)  ## NOTE: tensor
        positional_encoding = self.positional_encoding  ## NOTE: tensor
        pad_mask = np.zeros((self.max_seq_len,))  ## NOTE: tensor
        pad_mask[seq_length + 4 :] = (
            1  ## TODO: we are taking all tokens, including the special tokens.
        )

        seq_input = replace_characters_at_indices(
            s=seq_input, indices=input_mask_indices, replacement_char="[MASK]"
        )  ## NOTE: string

        input_mask = torch.zeros((self.max_seq_len,))  ## NOTE: tensor
        input_mask.scatter_(0, input_mask_indices, 1)

        target_mask = torch.zeros((self.max_seq_len,))  ## NOTE: tensor
        if self.split in {"train"} and False:
            raise
            target_mask = (
                input_mask.clone()
            )  ## NOTE: during training, we just want to compute loss and accuracy for the "masked tokens" only (ref: MDLM).
        else:
            target_mask[:seq_length] = (
                1  ## NOTE: during inference/testing, we want to compute loss and accuracy on the whole sequence.
            )

        input_mask_indices = "\t".join(
            [str(x.item()) for x in input_mask_indices]
        )  ## NOTE: String. To avoid any tensor preprocessing by pytorch-lightning

        ## padd tensors to maximum length
        tmp = structure_encoding.copy()
        structure_encoding = np.zeros((self.max_seq_len, structure_encoding.shape[1]))
        structure_encoding[:seq_length] = tmp  ## NOTE: tensor

        tmp = structure_encoding_vector.copy()
        structure_encoding_vector = np.zeros(
            (self.max_seq_len, structure_encoding_vector.shape[1], 3)
        )
        structure_encoding_vector[:seq_length] = tmp  ## NOTE: tensor

        tmp = logits.clone()
        logits = torch.zeros((self.max_seq_len, logits.shape[1]))
        logits[:seq_length] = tmp  ## NOTE: tensor

        tmp = target_ids.copy()
        target_ids = np.zeros((self.max_seq_len,))
        target_ids[:seq_length] = tmp  ## NOTE: tensor

        seq_length = torch.tensor(
            [seq_length]
        )  ## NOTE: return seq_length as a pytorch tensor. we will need it later for sanity-checking.
        posterior_weight = torch.Tensor([posterior_weight])
        masking_rate = torch.Tensor([masking_rate])

        if False:
            # print('\nsplit:', self.split)
            print("idx:", idx)
            print("posterior_weight", posterior_weight)
            print("masking_rate:", masking_rate)
            print("num_mask_tokens:", num_mask_tokens)
            print("seq_length:", seq_length)
            print("num_mask_tokens/seq_length:", num_mask_tokens / seq_length)
            print("t:", t)
            print("input_mask_indices:", input_mask_indices)
            print()
            exit()

        return (
            seq_input,
            seq_target,
            target_ids,
            input_mask_indices,
            logits,
            structure_encoding,
            structure_encoding_vector,
            positional_encoding,
            input_mask,
            target_mask,
            pad_mask,
            seq_length,
            posterior_weight,
            masking_rate,
            interval_partition,
            interval_idx,
        )
