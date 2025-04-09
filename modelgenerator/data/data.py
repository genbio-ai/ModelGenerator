# data.py
import anndata as ad
import os
import pandas as pd
import re
import random
import datasets
from typing import List, Literal, Optional, Tuple
from pyfaidx import Fasta
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from modelgenerator.data.base import *

string_complement_map = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "a": "t",
    "c": "g",
    "g": "c",
    "t": "a",
}


def replace_characters_at_indices(
    s: str, indices: list[int], replacement_char: str
) -> str:
    """Replace characters at given indices in a string

    Args:
        s (str): The input string
        indices (list[int]): The indices to replace
        replacement_char (str): The character to replace with

    Returns:
        str: The modified string
    """
    s_list = list(s)
    for index in indices:
        if 0 <= index < len(s_list):
            s_list[index] = replacement_char
    return "".join(s_list)


class AnyDataset(Dataset):
    """Takes as args any objects that implement __getitem__ and __len__ and wraps it as a PyTorch Dataset.

    Note:
        All datasets must have the same length

    Args:
        **kwargs: Key-value pairs of dataset names and the corresponding datasets
    """

    def __init__(self, **kwargs):
        self.datasets = []
        self.keys = []
        for key, dataset in kwargs.items():
            self.keys.append(key)
            self.datasets.append(dataset)
        self.size = len(self.datasets[0])
        for dataset in self.datasets:
            assert len(dataset) == self.size, "All datasets must have the same length"

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {key: dataset[idx] for key, dataset in zip(self.keys, self.datasets)}


class DiffusionDataset(Dataset):
    """Transforms a given dataset with 'sequences' into a dataset for discrete diffusion based on MDLM https://arxiv.org/abs/2406.07524

    Note:
        Implements a linear diffusion schedule with masking tokens as absorbing states
        Each sample includes timesteps_per_sample sequences at different noise levels
        Each sample's target sequences are under 'target_seqs', the input sequences are under 'input_seqs', and posterior weights are under 'posterior_weights'
        Retains all other keys in the input dataset

    Args:
        dataset (Dataset): The dataset to transform
        timesteps_per_sample (int): The number of timesteps per sample
        randomize_targets (bool): Whether to randomize the target sequences for each timestep (experimental efficiency boost proposed by Sazan)
    """

    def __init__(
        self,
        dataset: Dataset,
        timesteps_per_sample: int,
        randomize_targets: bool,
    ):
        self.dataset = dataset
        self.timesteps_per_sample = timesteps_per_sample
        self.randomize_targets = randomize_targets
        # Ref: MDLM, Appendix C.3 Low discrepancy sampler.
        self.interval_partitions = [
            [t / self.timesteps_per_sample, (t + 1) / self.timesteps_per_sample]
            for t in range(self.timesteps_per_sample)
        ]
        # Intentionally avoiding 0. Replacing 0 with a small value (1e-3).
        self.interval_partitions[0][0] = 1e-3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # One sample includes timesteps_per_sample sequences at different noise levels defined by the interval partitions
        # Ref: MDLM, Appendix C.3 Low discrepancy sampler (LDS).
        input_seqs = []
        target_seqs = []
        posterior_weights = []
        ret = {}
        for interval_partition in self.interval_partitions:
            # For each partition of the time interval, draw a time step uniformly and mask the input sequence accordingly
            if not self.randomize_targets:
                # By default, we gather many noised time steps for the same target sequence
                data_dict = self.dataset[idx]
            else:
                # But we can also randomize the target sequence
                data_dict = random.choice(self.dataset)
            # Extend any other keys in the dataset (e.g. labels) to each interval
            if ret == {}:
                ret = {k: [v] for k, v in data_dict.items()}
            else:
                for k, v in data_dict.items():
                    ret[k].append(v)
            # Create the masked and target sequences
            seq_target = data_dict["sequences"]
            # Sample the exact time, masking rate, and loss weight from within this interval of the time partition
            t = random.uniform(interval_partition[0], interval_partition[1])
            posterior_weight = 1 / t
            # Mask the target sequence according to the scheduled noise
            # At least one token has to be masked
            num_mask_tokens = max(1, int(len(seq_target) * t))
            perm = torch.randperm(
                len(seq_target)
            )  # todo: fix this, precludes using [MASK] in input data for targeted diffusion
            input_mask_indices = perm[:num_mask_tokens]
            # Finally, mask the target sequence
            seq_input = replace_characters_at_indices(
                s=seq_target, indices=input_mask_indices, replacement_char="[MASK]"
            )
            # Compile inputs, targets, and loss weights
            input_seqs.append(seq_input)
            target_seqs.append(seq_target)
            posterior_weights.append(posterior_weight)
        ret.update(
            {
                "sequences": input_seqs,
                "target_sequences": target_seqs,
                "posterior_weights": posterior_weights,
            }
        )
        return ret


class MLMDataset(Dataset):
    """Masked Language Modeling Dataset

    Note:
        Each sample includes a single sequence under key 'sequences' and a single target sequence under key 'target_sequences'

    Args:
        dataset (Dataset): The dataset to mask
        masking_rate (float): The masking rate
    """

    def __init__(self, dataset, masking_rate):
        self.dataset = dataset
        self.masking_rate = masking_rate

    def get_masked_sample(self, seq_target, masking_rate):
        """
        Mask a sequence with a given masking rate
        """
        num_mask_tokens = max(1, int(len(seq_target) * masking_rate))
        perm = torch.randperm(len(seq_target))
        input_mask_indices = perm[:num_mask_tokens]
        # Mask the input sequence
        seq_input = replace_characters_at_indices(
            s=seq_target, indices=input_mask_indices, replacement_char="[MASK]"
        )
        return seq_input

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        seq_target = data_dict["sequences"]
        seq_input = self.get_masked_sample(seq_target, self.masking_rate)
        data_dict.update({"sequences": seq_input, "target_sequences": seq_target})
        # prepend __empty__ to help pytorch lightning infer batch size
        return {"__empty__": 0, **data_dict}


# TODO: This class is just a use case of ColumnRetrievalDataModule.
class SequencesDataModule(DataInterface, HFDatasetLoaderMixin):
    """Data module for loading a simple dataset of sequences.

    Note:
        Each sample includes a single sequence under key 'sequences' and optionally an 'id' to track outputs.

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        id_col (str, optional): The name of the column containing the ids. Defaults to "id".
    """

    def __init__(
        self,
        *args,
        x_col: str = "sequence",
        id_col: str = "id",
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.x_col = x_col
        self.id_col = id_col

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        train_dataset, val_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        self.train_dataset = AnyDataset(
            ids=train_dataset[self.id_col], sequences=train_dataset[self.x_col]
        )
        self.val_dataset = AnyDataset(
            ids=val_dataset[self.id_col], sequences=val_dataset[self.x_col]
        )
        self.test_dataset = AnyDataset(
            ids=test_dataset[self.id_col], sequences=test_dataset[self.x_col]
        )


class DependencyMappingDataModule(SequencesDataModule):
    """Data module for doing dependency mapping via in silico mutagenesis on a dataset of sequences. Inherits from Sequences.

    Note:
        Each sample includes a single sequence under key 'sequences' and optionally an 'ids' to track outputs.

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        id_col (str, optional): The name of the column containing the ids. Defaults to "id".
        vocab_file (str, optional): The path to the file with the vocabulary to mutate.
    """

    def __init__(
        self,
        *args,
        vocab_file: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vocab_file = vocab_file

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        train_dataset, val_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        self.train_dataset = self.DependencyMappingDataset(
            AnyDataset(ids=train_dataset[self.id_col], sequences=train_dataset[self.x_col]),
            vocab_file=self.vocab_file,
        )
        self.val_dataset = self.DependencyMappingDataset(
            AnyDataset(ids=val_dataset[self.id_col], sequences=val_dataset[self.x_col]),
            vocab_file=self.vocab_file,
        )
        self.test_dataset = self.DependencyMappingDataset(
            AnyDataset(
                ids=test_dataset[self.id_col], sequences=test_dataset[self.x_col]
            ),
            vocab_file=self.vocab_file,
        )

    class DependencyMappingDataset(Dataset):
        """Dependency mapping dataset, which mutates each sequence in an input dataset at each position

        Note:
            Each sample includes a single sequence under key 'sequences'.

        Args:
            dataset (Dataset): The dataset to mask
            vocab (str): The path to the file with the vocabulary to mutate
        """

        def __init__(self, dataset, vocab_file: str):
            self.dataset = dataset
            self.vocab = open(vocab_file).read().strip().split("\n")
            # Determine dataset length
            self.sample_lens = []
            self.reverse_idx = []
            self.idx_cumsum = [0]
            for i in range(len(dataset)):
                sample_len = len(dataset[i]["sequences"]) * len(self.vocab) + 1
                self.sample_lens.append(sample_len)
                self.reverse_idx.extend([i] * sample_len)
                self.idx_cumsum.append(self.idx_cumsum[-1] + sample_len)

        def __len__(self):
            return self.idx_cumsum[-1]

        def __getitem__(self, idx):
            # Indexing (First vocab * sequences samples are mutations, last is wt)
            seq_idx = self.reverse_idx[idx]
            total_seq_samples = self.sample_lens[seq_idx]
            seq_sample_i = idx - self.idx_cumsum[seq_idx]
            data_dict = self.dataset[seq_idx]
            seq_identifier = data_dict.get("ids", str(seq_idx))
            if seq_sample_i == total_seq_samples - 1:
                # WT sequence
                data_dict.update(
                    {
                        "ids": seq_identifier,
                        "pos_i": -1,
                        "mut_i": -1,
                    }
                )
                return data_dict
            # Mutate
            mut_i = seq_sample_i % len(self.vocab)
            pos_i = seq_sample_i // len(self.vocab)
            seq = list(data_dict["sequences"])
            seq[pos_i] = self.vocab[mut_i]
            seq = "".join(seq)
            data_dict.update(
                {
                    "sequences": seq,
                    "ids": seq_identifier,
                    "pos_i": pos_i,
                    "mut_i": mut_i,
                }
            )
            return data_dict


class SequenceClassificationDataModule(DataInterface, HFDatasetLoaderMixin):
    """Data module for Hugging Face sequence classification datasets.

    Note:
        Each sample includes a single sequence under key 'sequences' and a single class label under key 'labels'

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        y_col (str | List[str], optional): The name of the column(s) containing the labels. Defaults to "label".
        class_filter (List[int] | int, optional): The class to filter. Defaults to None.
    """

    def __init__(
        self,
        *args,
        x_col: str = "sequence",
        y_col: str | List[str] = "label",
        class_filter: int | List[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.x_col = x_col
        self.y_col = y_col
        self.class_filter = None
        if class_filter is not None:
            self.class_filter = (
                class_filter if isinstance(class_filter, list) else [class_filter]
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        is_multi_label = isinstance(self.y_col, list)
        train_dataset, val_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        if not is_multi_label:
            train_dataset, val_dataset, test_dataset = self.filter_by_class(
                (train_dataset, val_dataset, test_dataset)
            )
        elif self.class_filter:
            rank_zero_warn(
                f"Class filter not supported for multi-label datasets. Ignoring class filter."
            )
        train_dataset, val_dataset, test_dataset = self.get_split_by_fold_id(
            train_dataset,
            val_dataset,
            test_dataset,
            self.cv_test_fold_id,
            self.cv_val_offset,
        )

        self.train_dataset = AnyDataset(
            sequences=train_dataset[self.x_col],
            labels=(
                train_dataset[self.y_col]
                if not is_multi_label
                else self.to_multihot(train_dataset)
            ),
        )
        self.val_dataset = AnyDataset(
            sequences=val_dataset[self.x_col],
            labels=(
                val_dataset[self.y_col]
                if not is_multi_label
                else self.to_multihot(val_dataset)
            ),
        )
        self.test_dataset = AnyDataset(
            sequences=test_dataset[self.x_col],
            labels=(
                test_dataset[self.y_col]
                if not is_multi_label
                else self.to_multihot(test_dataset)
            ),
        )

    def filter_by_class(
        self, datasets: List[datasets.Dataset]
    ) -> Tuple[datasets.Dataset]:
        if not self.class_filter:
            return datasets
        rank_zero_info(f"> Filtering to class {self.class_filter}")

        filtered_datasets = []
        for ds in datasets:
            filtered_datasets.append(
                ds.filter(lambda x: x[self.y_col] in self.class_filter)
            )
        return tuple(filtered_datasets)

    def to_multihot(self, dataset: datasets.Dataset) -> datasets.Dataset:
        if not isinstance(self.y_col, list):
            return dataset
        rank_zero_info(f"> Creating multi-hot encoding for {self.y_col}")
        return dataset.select_columns(self.y_col).to_pandas().to_numpy()


class TokenClassificationDataModule(DataInterface, HFDatasetLoaderMixin):
    """Data module for Hugging Face token classification datasets.

    Note:
        Each sample includes a single sequence under key 'sequences' and a single class sequence under key 'labels'

    Args:
        x_col (str, optional): The name of the column containing the sequences. Defaults to "sequence".
        y_col (str, optional): The name of the column containing the labels. Defaults to "label".
        max_length (int, optional): The maximum length of the sequences. Defaults to None.
        pairwise (bool, optional): Whether the labels are pairwise. Defaults to False.
    """

    def __init__(
        self,
        *args,
        x_col: str = "sequence",
        y_col: str = "label",
        max_length: Optional[int] = None,
        pairwise: bool = False,
        **kwargs,
    ):
        def collate_pad_labels(batch):
            # collate function that pads the labels in all shapes
            max_len = max([item["labels"].shape[0] for item in batch])
            padded_labels = []
            for item in batch:
                labels = item["labels"]
                padded_label = np.pad(
                    item["labels"],
                    (0, max_len - labels.shape[0]),
                    "constant",
                    constant_values=-100,
                )
                padded_labels.append(padded_label)
            return {
                "sequences": [item["sequences"] for item in batch],
                "labels": torch.tensor(np.array(padded_labels)),
            }

        super().__init__(*args, collate_fn=collate_pad_labels, **kwargs)
        self.x_col = x_col
        self.y_col = y_col
        self.max_length = max_length
        self.pairwise = pairwise

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets.

        Note:
            Assumes no default validation set. Splits into training and validation sets using a fixed random seed.

        """
        train_dataset, valid_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        train_dataset, valid_dataset, test_dataset = self.get_split_by_fold_id(
            train_dataset,
            valid_dataset,
            test_dataset,
            self.cv_test_fold_id,
            self.cv_val_offset,
        )
        train_sequences, train_labels = self.process_dataset(train_dataset)
        valid_sequences, valid_labels = self.process_dataset(valid_dataset)
        test_sequences, test_labels = self.process_dataset(test_dataset)
        self.train_dataset = AnyDataset(sequences=train_sequences, labels=train_labels)
        self.test_dataset = AnyDataset(sequences=test_sequences, labels=test_labels)
        self.val_dataset = AnyDataset(sequences=valid_sequences, labels=valid_labels)

    def process_dataset(self, dataset):
        seqs = []
        labels = []
        for line in dataset:
            seq = line[self.x_col]
            label = line[self.y_col]
            if self.pairwise:
                # convert pair indexes into an adjacency matrix
                label_mt = np.zeros((len(seq), len(seq)), dtype=np.int64)
                for p in label:
                    label_mt[int(p[0]), int(p[1])] = 1
                    label_mt[int(p[1]), int(p[0])] = 1
                if self.max_length is not None and len(seq) > self.max_length - 1:
                    seq = seq[: self.max_length - 1]
                    label_mt = label_mt[: self.max_length - 1, : self.max_length - 1]
                assert len(seq) == label_mt.shape[0]
                seqs.append(seq)
                labels.append(label_mt)
            else:
                label = np.array([int(y) for y in label])
                if self.max_length is not None and len(seq) > self.max_length - 1:
                    seq = seq[: self.max_length - 1]
                    label = label[: self.max_length - 1]
                assert len(seq) == len(label)
                seqs.append(seq)
                labels.append(label)
        return seqs, labels


class StructureTokenDataModule(DataInterface, HFDatasetLoaderMixin):
    """Test only data module for structure token predictors.

    This data module is specifically designed for handling datasets uses amino acid sequences as input
    and structure tokens as labels.

    Note:
        This module only supports testing and ignores training and validation splits.
        It assumes test split files contain sequences and optionally their structural token labels.
        If structural token labels are not provided, dummy labels are created.

    Args:
        path (str): The path to the dataset files.
        config_name (Optional[str], optional): Configuration name for dataset loading. Defaults to None.
        test_split_files (Optional[List[str]], optional): List of files for the test split. Defaults to None.
        batch_size (int, optional): The batch size for data loading. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the parent class, in which training and validation split
            settings are overridden so that only the test split is loaded.
    """

    def __init__(
        self,
        path: str,
        config_name: Optional[str] = None,
        test_split_files: Optional[List[str]] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        # this dataset only supports test split, override these kwargs
        kwargs["train_split_name"] = None
        kwargs["valid_split_name"] = None
        kwargs["valid_split_size"] = 0
        if batch_size > 1:
            rank_zero_warn(
                f"Found batch_size > 1. If you find OOM error, please set batch_size = 1."
            )

        super().__init__(
            path=path,
            config_name=config_name,
            test_split_files=test_split_files,
            batch_size=batch_size,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.val_dataset, test_dataset = (
            self.load_and_split_dataset(**self.extra_reader_kwargs)
        )
        # For prediction, labels are not necessary. You could set labels to a sequence of zeros.
        self.test_dataset = AnyDataset(
            sequences=test_dataset["aa_seq"],
            labels=(
                test_dataset["struct_seq"]
                if "struct_seq" in test_dataset.features
                else self.create_dummy_labels(test_dataset["aa_seq"])
            ),
            uid=test_dataset["idx"],
        )
        rank_zero_info(f"Loaded {len(self.test_dataset)} test samples")

    def create_dummy_labels(self, aa_seqs: List[str]) -> List[str]:
        return [str(np.zeros(len(seq), dtype=int).tolist()) for seq in aa_seqs]


class ZeroshotClassificationRetrieveDataModule(DataInterface, HFDatasetLoaderMixin):
    """Test only data module that constructs a testing dataset by reading indeces from files in test split and retrieve sequeces from a reference file.

    Note:
        Each sample includes index for a single sequence under key ["chrom","start","end","ref","mutate"] and a single class label under key 'effect'

    Args:
        method (str, ["Diff", "Distance"]): method mode to compute metrics
        index_cols ([str], optional): The list of the column name containing the index for sequence retrieval.
        y_col (str, optional): The name of the column containing the labels. Defaults to "label".
        test_split_name (str, optional): The name of the test split. Defaults to "test".
        test_split_size (float, optional): ignored
        window (int, optional): The number of token taken on either side of the mutation site. The processed sequence length is 2* window + 1
        reference_file (str, optional): The file path to the reference file for retrieving sequences
    """

    def __init__(
        self,
        *args,
        index_cols: List[str] = ["chrom", "start", "end", "ref", "mutate"],
        y_col: str = "label",
        method: Optional[Literal["Diff", "Distance"]] = None,
        window: Optional[int] = None,
        reference_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            train_split_name=None,
            valid_split_name=None,
            valid_split_size=0,
            **kwargs,
        )
        self.index_cols = index_cols
        self.y_col = y_col
        self.method = method
        self.reference_file = reference_file
        self.window = window

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.val_dataset, test_dataset = (
            self.load_and_split_dataset(**self.extra_reader_kwargs)
        )
        test_dataset: datasets.Dataset
        if os.path.exists(self.reference_file):
            reference_file = self.reference_file
        elif os.path.exists(self.path):
            reference_file = os.path.join(self.path, self.reference_file)
        else:
            raise ValueError(
                "reading sequence file from a huggingface dataset is not supported"
            )
        seq_ref = Fasta(reference_file)
        if self.method == "Diff":
            test_sequences, test_labels, test_refs, test_mutations = (
                self.process_dataset(test_dataset, seq_ref)
            )
            self.test_dataset = AnyDataset(
                labels=test_labels,
                sequences=test_sequences,
                refs=test_refs,
                mutations=test_mutations,
            )
        elif self.method == "Distance":
            test_ref_sequences, test_mutation_sequences, test_labels = (
                self.process_dataset(test_dataset, seq_ref)
            )
            self.test_dataset = AnyDataset(
                labels=test_labels,
                ref_sequences=test_ref_sequences,
                mutation_sequences=test_mutation_sequences,
            )
        else:
            ValueError(
                f"Do not support method: {self.method}, method should be Diff or Distance"
            )

    def process_dataset(self, dataset, seq_ref):
        if self.method == "Diff":
            seqs = []
            labels = []
            refs = []
            mutations = []
        else:
            ref_seqs = []
            mutation_seqs = []
            labels = []
        count = 0
        total_count = 0
        for line in dataset:
            total_count += 1
            chrom, start, end, ref, mutation = [line[key] for key in self.index_cols]
            label = line[self.y_col]
            # processed sequence length is 2 * self.window + 1
            seq = str(seq_ref[chrom][start - self.window : end + self.window]).upper()
            # replace any non-ACGTN charaters with N
            seq = re.sub(r"[^ACGTN]", "N", seq)
            # check if ref aligns with fasta seq
            if self.method == "Diff":
                # mask the mutated site
                if (seq[self.window] == ref) or (
                    seq[self.window] == string_complement_map[ref]
                ):
                    seq = replace_characters_at_indices(
                        s=seq, indices=[self.window], replacement_char="[MASK]"
                    )
                    seqs.append(seq)
                    labels.append(label)
                    refs.append(
                        ref if seq[self.window] == ref else string_complement_map[ref]
                    )
                    mutations.append(
                        mutation
                        if seq[self.window] == ref
                        else string_complement_map[mutation]
                    )
                    count += 1
                else:
                    # I don't know how to handle these 2 faulty samples, just follow what we did in original clinvar dataset
                    seq = replace_characters_at_indices(
                        s=seq, indices=[self.window], replacement_char="[MASK]"
                    )
                    seqs.append(seq)
                    labels.append(label)
                    refs.append(ref)
                    mutations.append(mutation)
            else:
                # prepare reference and mutation sequences
                if (seq[self.window] == ref) or (
                    seq[self.window] == string_complement_map[ref]
                ):
                    ref_seq = replace_characters_at_indices(
                        s=seq,
                        indices=[self.window],
                        replacement_char=(
                            ref
                            if seq[self.window] == ref
                            else string_complement_map[ref]
                        ),
                    )
                    mutation_seq = replace_characters_at_indices(
                        s=seq,
                        indices=[self.window],
                        replacement_char=(
                            mutation
                            if seq[self.window] == ref
                            else string_complement_map[mutation]
                        ),
                    )
                    ref_seqs.append(ref_seq)
                    mutation_seqs.append(mutation_seq)
                    labels.append(label)
                    count += 1
                else:
                    ref_seq = replace_characters_at_indices(
                        s=seq, indices=[self.window], replacement_char=ref
                    )
                    mutation_seq = replace_characters_at_indices(
                        s=seq, indices=[self.window], replacement_char=mutation
                    )
                    ref_seqs.append(ref_seq)
                    mutation_seqs.append(mutation_seq)
                    labels.append(label)
        rank_zero_info(
            f"successfully processed sequence: {count}, total sequence: {total_count}"
        )
        if self.method == "Diff":
            return seqs, labels, refs, mutations
        else:
            return ref_seqs, mutation_seqs, labels


class DiffusionDataModule(DataInterface, HFDatasetLoaderMixin):
    """Data module for datasets with discrete diffusion-based noising and loss weights from MDLM https://arxiv.org/abs/2406.07524.

    Notes:
        Each sample includes timesteps_per_sample sequences at different noise levels
        Each sample's target sequences are under 'target_sequences', the input sequences are under 'sequences', and posterior weights are under 'posterior_weights'

    Args:
        x_col (str, optional): The column with the data to train on, defaults to "sequence"
        timesteps_per_sample (int, optional): The number of timesteps per sample, defaults to 10
        randomize_targets (bool, optional): Whether to randomize the target sequences for each timestep (experimental efficiency boost proposed by Sazan)
        batch_size (int, optional): The batch size, defaults to 10
    """

    def __init__(
        self,
        *args,
        x_col: str = "sequence",
        timesteps_per_sample: int = 10,
        randomize_targets: bool = False,
        batch_size: int = 10,
        **kwargs,
    ):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.x_col = x_col
        self.timesteps_per_sample = timesteps_per_sample
        self.randomize_targets = randomize_targets

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading a different DiffusionDataset schedule for train, validation, and test.

        Note:
            Validation and test allow for full multi-step denoising process with a random [0,1] masking ratio for each sample
            where timesteps_per_sample=1 (i.e. one masked sequence per sample)
        """
        train_dataset, valid_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        self.train_dataset = DiffusionDataset(
            dataset=AnyDataset(sequences=train_dataset[self.x_col]),
            timesteps_per_sample=self.timesteps_per_sample,
            randomize_targets=self.randomize_targets,
        )
        # For valid and test, we denoise a single masked sample and compute MLM recovery loss
        self.val_dataset = DiffusionDataset(
            dataset=AnyDataset(sequences=valid_dataset[self.x_col]),
            timesteps_per_sample=1,
            randomize_targets=False,
        )
        self.test_dataset = DiffusionDataset(
            dataset=AnyDataset(sequences=test_dataset[self.x_col]),
            timesteps_per_sample=1,
            randomize_targets=False,
        )


class ClassDiffusionDataModule(SequenceClassificationDataModule):
    """Data module for conditional (or class-filtered) diffusion, and applying discrete diffusion noising. Inherits from SequenceClassification.

    Note:
        Each sample includes timesteps_per_sample sequences at different noise levels
        Each sample's target sequences are under 'target_seqs', the input sequences are under 'input_seqs', and posterior weights are under 'posterior_weights'

    Args:
        timesteps_per_sample (int, optional): The number of timesteps per sample, defaults to 10
        randomize_targets (bool, optional): Whether to randomize the target sequences for each timestep (experimental efficiency boost proposed by Sazan)
        batch_size (int, optional): The batch size, defaults to 10
    """

    def __init__(
        self,
        *args,
        timesteps_per_sample: int = 10,
        randomize_targets: bool = False,
        batch_size: int = 10,
        **kwargs,
    ):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.timesteps_per_sample = timesteps_per_sample
        self.randomize_targets = randomize_targets

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading the whole datasets, fltering by class, and splitting them into training, validation, and test sets.

        Note:
            Assumes no default validation set. Splits training 90/10 into training and validation sets using a fixed random seed.
            Validation and test allow for full multi-step denoising process with a random [0,1] masking ratio for each sample
            where timesteps_per_sample=1 (i.e. one masked sequence per sample)

        Args:
            stage (Optional[str], optional): training, validation, or test if these need to be setup separately. Defaults to None.
        """
        super().setup(stage)
        self.train_dataset = DiffusionDataset(
            dataset=self.train_dataset,
            timesteps_per_sample=self.timesteps_per_sample,
            randomize_targets=self.randomize_targets,
        )
        # For valid and test, we denoise a single masked sample and compute MLM recovery loss
        self.val_dataset = DiffusionDataset(
            dataset=self.val_dataset,
            timesteps_per_sample=1,
            randomize_targets=False,
        )
        self.test_dataset = DiffusionDataset(
            dataset=self.test_dataset,
            timesteps_per_sample=1,
            randomize_targets=False,
        )


# TODO: This data module should not require labels.
class MLMDataModule(SequenceClassificationDataModule):
    """Data module for continuing pretraining on a masked language modeling task. Inherits from SequenceClassificationDataModule.

    Note:
        Each sample includes a single sequence under key 'sequences' and a single target sequence under key 'target_sequences'

    Args:
        masking_rate (float, optional): The masking rate. Defaults to 0.15.
    """

    def __init__(
        self,
        *args,
        masking_rate: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masking_rate = masking_rate

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        self.train_dataset = MLMDataset(
            dataset=self.train_dataset,
            masking_rate=self.masking_rate,
        )
        self.val_dataset = MLMDataset(
            dataset=self.val_dataset,
            masking_rate=self.masking_rate,
        )
        self.test_dataset = MLMDataset(
            dataset=self.test_dataset,
            masking_rate=self.masking_rate,
        )


class SequenceRegressionDataModule(DataInterface, HFDatasetLoaderMixin):
    """Data module sequence regression datasets.

    Args:
        x_col (union[str, list], optional): The name of columns containing the sequences. Defaults to "sequence".
        y_col (union[str, list], optional): The name of columns containing the labels. Defaults to "label".
        normalize (bool, optional): Whether to normalize the labels. Defaults to True.
    """

    def __init__(
        self,
        *args,
        x_col: str = "sequence",
        y_col: str = "label",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.x_col = x_col
        self.y_col = y_col
        self.normalize = normalize
        if isinstance(self.y_col, list):
            rank_zero_info(f"> Multi-task regression for {self.y_col}")

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        train_dataset, valid_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        train_dataset, valid_dataset, test_dataset = self.get_split_by_fold_id(
            train_dataset,
            valid_dataset,
            test_dataset,
            self.cv_test_fold_id,
            self.cv_val_offset,
        )
        # train_dataset cannot be empty or None
        train_sequences = self.select_input_columns(train_dataset)
        train_labels = self.select_label_columns(train_dataset)
        valid_sequences = self.select_input_columns(valid_dataset)
        valid_labels = self.select_label_columns(valid_dataset)
        test_sequences = self.select_input_columns(test_dataset)
        test_labels = self.select_label_columns(test_dataset)
        if self.normalize:
            label_mean = np.mean(np.concatenate([train_labels, valid_labels]), axis=0)
            label_std = np.std(np.concatenate([train_labels, valid_labels]), axis=0)
            rank_zero_info(f"label: mean = {label_mean}, std = {label_std}")
            train_labels = (train_labels - label_mean) / label_std
            valid_labels = (valid_labels - label_mean) / label_std
            test_labels = (test_labels - label_mean) / label_std
        if isinstance(self.y_col, list):
            self.train_dataset = AnyDataset(
                sequences=train_sequences, labels=train_labels
            )
            self.val_dataset = AnyDataset(
                sequences=valid_sequences, labels=valid_labels
            )
            self.test_dataset = AnyDataset(sequences=test_sequences, labels=test_labels)
        else:
            self.train_dataset = AnyDataset(
                sequences=train_sequences, labels=train_labels[:, None]
            )
            self.val_dataset = AnyDataset(
                sequences=valid_sequences, labels=valid_labels[:, None]
            )
            self.test_dataset = AnyDataset(
                sequences=test_sequences, labels=test_labels[:, None]
            )

    def select_label_columns(self, dataset: datasets.Dataset) -> datasets.Dataset:
        if not isinstance(self.y_col, list):
            labels = dataset[self.y_col]
            if len(labels) > 0 and isinstance(labels[0], str):
                labels = [float(y) for y in labels]
            return np.array(labels)
        return dataset.select_columns(self.y_col).to_pandas().to_numpy()

    def select_input_columns(self, dataset: datasets.Dataset) -> datasets.Dataset:
        if not isinstance(self.x_col, list):
            return dataset[self.x_col]
        # TODO: select_columns returns a dictionary of columns, which introduces unnecessary
        # dependency on the data source column names. We should either return ndarrays as lables
        # or allow the user to rename the columns for input sequences.
        return dataset.select_columns(self.x_col)


class ColumnRetrievalDataModule(DataInterface, HFDatasetLoaderMixin):
    """Simple data module for retrieving and renaming columns from a dataset. Inherits from BaseDataModule."""

    def __init__(
        self,
        *args,
        in_cols: List[str] = [],
        out_cols: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_cols = in_cols
        self.out_cols = out_cols or in_cols

    def setup(self, stage: Optional[str] = None):
        train_dataset, valid_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        self.train_dataset = AnyDataset(
            **{
                out_col: train_dataset[in_col]
                for in_col, out_col in zip(self.in_cols, self.out_cols)
            }
        )
        self.val_dataset = AnyDataset(
            **{
                out_col: valid_dataset[in_col]
                for in_col, out_col in zip(self.in_cols, self.out_cols)
            }
        )
        self.test_dataset = AnyDataset(
            **{
                out_col: test_dataset[in_col]
                for in_col, out_col in zip(self.in_cols, self.out_cols)
            }
        )


# TODO: Almost identical to SequenceRegressionDataModule, except for label unsqueezing.
class RNAMeanRibosomeLoadDataModule(SequenceRegressionDataModule):
    def __init__(
        self,
        path: str = "genbio-ai/rna-downstream-tasks",
        config_name: str = "mean_ribosome_load",
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            config_name=config_name,
            x_col="utr",
            y_col="rl",
            train_split_name="train",
            valid_split_name="validation",
            test_split_name="test",
            normalize=normalize,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        train_dataset, valid_dataset, test_dataset = self.load_and_split_dataset(**self.extra_reader_kwargs)
        train_dataset, valid_dataset, test_dataset = self.get_split_by_fold_id(
            train_dataset, valid_dataset, test_dataset, self.cv_test_fold_id
        )
        # train_dataset cannot be empty or None
        train_sequences = train_dataset[self.x_col]
        train_labels = np.array(train_dataset[self.y_col])
        valid_sequences = valid_dataset[self.x_col]
        valid_labels = np.array(valid_dataset[self.y_col])
        test_sequences = test_dataset[self.x_col]
        test_labels = np.array(test_dataset[self.y_col])
        if self.normalize:
            label_mean = np.mean(np.concatenate([train_labels, valid_labels]))
            label_std = np.std(np.concatenate([train_labels, valid_labels]))
            train_labels = (train_labels - label_mean) / label_std
            valid_labels = (valid_labels - label_mean) / label_std
            test_labels = (test_labels - label_mean) / label_std

        self.train_dataset = AnyDataset(sequences=train_sequences, labels=train_labels)
        self.val_dataset = AnyDataset(sequences=valid_sequences, labels=valid_labels)
        self.test_dataset = AnyDataset(sequences=test_sequences, labels=test_labels)

        # def val_dataloader(self) -> DataLoader:
        #     """Get the validation data loader

        #     Returns:
        #         DataLoader: The validation data loader
        #     """
        #     return DataLoader(
        #         self.val_dataset,
        #         batch_size=self.batch_size * 2,
        #         shuffle=False,
        #         sampler=self.sampler,
        #         num_workers=self.num_workers,
        #         collate_fn=self.collate_fn,
        #         pin_memory=self.pin_memory,
        #         persistent_workers=self.persistent_workers,
        #     )

        # def test_dataloader(self) -> DataLoader:
        """Get the test data loader

        Returns:
            DataLoader: The test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class ConditionalDiffusionDataModule(SequenceRegressionDataModule):
    """Data module for conditional diffusion with a continuous condition, and applying discrete diffusion noising. Inherits from SequenceRegression.

    Note:
        Each sample includes timesteps_per_sample sequences at different noise levels
        Each sample's target sequences are under 'target_seqs', the input sequences are under 'input_seqs', and posterior weights are under 'posterior_weights'

    Args:
        timesteps_per_sample (int, optional): The number of timesteps per sample, defaults to 10
        randomize_targets (bool, optional): Whether to randomize the target sequences for each timestep (experimental efficiency boost proposed by Sazan)
        batch_size (int, optional): The batch size, defaults to 10
    """

    def __init__(
        self,
        *args,
        timesteps_per_sample: int = 10,
        randomize_targets: bool = False,
        batch_size: int = 10,
        **kwargs,
    ):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.timesteps_per_sample = timesteps_per_sample
        self.randomize_targets = randomize_targets

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading the whole datasets, fltering by class, and splitting them into training, validation, and test sets.

        Note:
            Assumes no default validation set. Splits training 90/10 into training and validation sets using a fixed random seed.
            Validation and test allow for full multi-step denoising process with a random [0,1] masking ratio for each sample
            where timesteps_per_sample=1 (i.e. one masked sequence per sample)

        Args:
            stage (Optional[str], optional): training, validation, or test if these need to be setup separately. Defaults to None.
        """
        super().setup(stage)
        self.train_dataset = DiffusionDataset(
            dataset=self.train_dataset,
            timesteps_per_sample=self.timesteps_per_sample,
            randomize_targets=self.randomize_targets,
        )
        # For valid and test, we denoise a single masked sample and compute MLM recovery loss
        self.val_dataset = DiffusionDataset(
            dataset=self.val_dataset,
            timesteps_per_sample=1,
            randomize_targets=False,
        )
        self.test_dataset = DiffusionDataset(
            dataset=self.test_dataset,
            timesteps_per_sample=1,
            randomize_targets=False,
        )


class CellClassificationDataModule(DataInterface):
    """Data module for cell classification. Inherits from BaseDataModule.

    Note:
        Each sample includes a feature vector (one of the rows in <adata.X>) and a single class label (one of the columns in <adata.obs>)

    Args:
        filter_columns (Optional[list[str]], optional): The columns of <obs> we want to use. Defaults to None, in which case all columns are used.
        rename_columns (Optional[list[str]], optional): New name of columns. Defaults to None, in which case columns are not renamed. Does nothing if filter_colums is None.
        # TODO: Add option to return a subset of genes by filtering on <var>.
    """

    def __init__(
        self,
        *args,
        filter_columns: Optional[list[str]] = None,
        rename_columns: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(self.train_split_files) != 1:
            raise NotImplementedError("Multiple files not yet supported.")
        if len(self.valid_split_files) != 1:
            raise NotImplementedError("Multiple files not yet supported.")
        if len(self.test_split_files) != 1:
            raise NotImplementedError("Multiple files not yet supported.")
        self.trainfile = self.train_split_files[0]
        self.valfile = self.valid_split_files[0]
        self.testfile = self.test_split_files[0]
        self.rename_columns = rename_columns
        self.filter_columns = filter_columns

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        adata_train = ad.read_h5ad(os.path.join(self.path, self.trainfile))
        adata_val = ad.read_h5ad(os.path.join(self.path, self.valfile))
        adata_test = ad.read_h5ad(os.path.join(self.path, self.testfile))
        # only remain useful data for classification
        if self.filter_columns is not None:
            adata_train.obs = adata_train.obs[self.filter_columns]
            adata_val.obs = adata_val.obs[self.filter_columns]
            adata_test.obs = adata_test.obs[self.filter_columns]
            # rename columns to adapt to classification task
            if self.rename_columns is not None:
                # Only rename columns if we filter first - otherwise order not guaranteed.
                adata_train.obs.columns = self.rename_columns
                adata_val.obs.columns = self.rename_columns
                adata_test.obs.columns = self.rename_columns
        D_train = {key: torch.from_numpy(adata_train.obs[key].values) for key in adata_train.obs.columns}
        # D_train['sequences'] = np.array(list(adata_train.X)) # Note: "sequences" nomenclature is due to SequenceClassification task requirements.
        # TODO: Support sparse. Hitting problems in default_collate.
        D_train["sequences"] = adata_train.X.toarray()
        D_val = {key: torch.from_numpy(adata_val.obs[key].values) for key in adata_val.obs.columns}
        # D_val['sequences'] = np.array(list(adata_val.X))
        D_val["sequences"] = adata_val.X.toarray()
        D_test = {key: torch.from_numpy(adata_test.obs[key].values) for key in adata_test.obs.columns}
        # D_test['sequences'] = np.array(list(adata_test.X))
        D_test["sequences"] = adata_test.X.toarray()

        self.train_dataset = AnyDataset(**D_train)
        self.val_dataset = AnyDataset(**D_val)
        self.test_dataset = AnyDataset(**D_test)

class ClockDataModule(DataInterface):
    """Data module for transcriptomic clock tasks. Inherits from BaseDataModule.

    Note:
        Each sample includes a feature vector (one of the rows in <adata.X>) and a single scalar corresponding to donor age (one of the columns in <adata.obs>)

    Args:
        split_column (str): The column of <obs> that defines the split assignments.
        gene_set_file (str): Path to a csv file containing gene symbols in the order expected by the model being used. 
        filter_columns (Optional[list[str]], optional): The columns of <obs> we want to use. Defaults to None, in which case all columns are used.
        rename_columns (Optional[list[str]], optional): New name of columns. Defaults to None, in which case columns are not renamed. Does nothing if filter_colums is None.
        # TODO: Add option to return a subset of genes by filtering on <var>.
    """

    def __init__(
        self,
        *args,
        split_column: str,
        gene_set_file: str,
        label_scaling: Optional[str] = 'z_scaling',
        filter_columns: Optional[list[str]] = None,
        rename_columns: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(self.train_split_files) != 1:
            raise NotImplementedError("Multiple files not yet supported.")
        if (self.valid_split_files is not None) or (self.test_split_files is not None):
            raise NotImplementedError("Data should live in a single file with splits defined by a column of <obs>.")
        self.split_column = split_column
        self.gene_set_file = gene_set_file
        self.label_scaling = label_scaling
        self.trainfile = self.train_split_files[0]
        self.rename_columns = rename_columns
        self.filter_columns = filter_columns

    def setup(self, stage: Optional[str] = None):
        """Set up the data module by loading the whole datasets and splitting them into training, validation, and test sets."""
        adata = ad.read_h5ad(os.path.join(self.path, self.trainfile))

        # align genes
        adata.var.index = adata.var['feature_name']
        model_genes = pd.read_csv(self.gene_set_file, sep='\t')['gene_name'].to_numpy()
        data_genes = adata.var.index
        common_genes = np.intersect1d(model_genes, data_genes)
        if len(common_genes) == 0:
            raise ValueError(f'Gene alignment failed, no common genes found. Are the gene names in the same format? (model: {model_genes[:5]}, data: {data_genes[:5]})')
        missing_genes = np.setdiff1d(model_genes, data_genes)
        adata_missing = ad.AnnData(np.zeros((adata.shape[0], len(missing_genes))))
        adata_missing.var.index = missing_genes
        adata_missing.obs = adata.obs
        adata_aligned = ad.concat((adata, adata_missing), axis=1, join='inner', merge='same')
        adata_aligned = adata_aligned[:, model_genes]
        rank_zero_info(f'\n***\ngene alignment results: {len(common_genes)} common genes; {len(missing_genes)} missing genes filled with zeros\n***\n')

        adata_train = adata_aligned[adata_aligned.obs[self.split_column] == 'train']
        adata_val = adata_aligned[adata_aligned.obs[self.split_column] == 'val']
        adata_test = adata_aligned[adata_aligned.obs[self.split_column] == 'test']

        # label scaling
        if self.label_scaling == 'z_scaling':
            train_labels = adata_train.obs['numeric_age'].to_numpy()
            mu = np.mean(train_labels)
            sigma = np.std(train_labels)
            if np.isnan(sigma):
                raise ValueError('z_scaling failed: std is nan')
            adata_train.obs['numeric_age'] = (adata_train.obs['numeric_age'] - mu) / sigma
            adata_val.obs['numeric_age'] = (adata_val.obs['numeric_age'] - mu) / sigma
            adata_test.obs['numeric_age'] = (adata_test.obs['numeric_age'] - mu) / sigma
        else:
            raise NotImplementedError(f'label_scaling {self.label_scaling} not recognized.')
        
        # only retain useful data for regression
        if self.filter_columns is not None:
            adata_train.obs = adata_train.obs[self.filter_columns]
            adata_val.obs = adata_val.obs[self.filter_columns]
            adata_test.obs = adata_test.obs[self.filter_columns]
            # rename columns to adapt to regression task
            if self.rename_columns is not None:
                # Only rename columns if we filter first - otherwise order not guaranteed.
                adata_train.obs.columns = self.rename_columns
                adata_val.obs.columns = self.rename_columns
                adata_test.obs.columns = self.rename_columns
        
        D_train = {key: torch.from_numpy(adata_train.obs[key].values)[:, None] for key in adata_train.obs.columns}
        D_train["sequences"] = adata_train.X.toarray()
        D_val = {key: torch.from_numpy(adata_val.obs[key].values)[:, None] for key in adata_val.obs.columns}
        D_val["sequences"] = adata_val.X.toarray()
        D_test = {key: torch.from_numpy(adata_test.obs[key].values)[:, None] for key in adata_test.obs.columns}
        D_test["sequences"] = adata_test.X.toarray()

        self.train_dataset = AnyDataset(**D_train)
        self.val_dataset = AnyDataset(**D_val)
        self.test_dataset = AnyDataset(**D_test)
