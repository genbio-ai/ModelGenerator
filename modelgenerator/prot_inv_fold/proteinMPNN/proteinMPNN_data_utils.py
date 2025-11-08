"""
Code in this file is adapted from https://github.com/BytedProtein/ByProt/blob/main/src/byprot/datamodules/datasets/data_utils.py
"""

import heapq
import itertools
import math
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator, List, Sequence, Tuple, TypeVar, Union

# import esm
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataChunk
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler


class Alphabet(object):
    def __init__(
        self, name="esm", featurizer="cath", alphabet_cfg={}, featurizer_cfg={}
    ):
        self.name = name
        self._alphabet = None

        if name == "esm":
            raise Exception("Code should not reach here.")
            self._alphabet = esm.Alphabet.from_architecture("ESM-1b")
            self.add_special_tokens = True
        elif name == "protein_mpnn":
            self._alphabet = esm.Alphabet(
                standard_toks=[
                    "A",
                    "R",
                    "N",
                    "D",
                    "C",
                    "Q",
                    "E",
                    "G",
                    "H",
                    "I",
                    "L",
                    "K",
                    "M",
                    "F",
                    "P",
                    "S",
                    "T",
                    "W",
                    "Y",
                    "V",
                ],
                prepend_toks=["<pad>", "<unk>"],
                append_toks=[],
                prepend_bos=False,
                append_eos=False,
            )
            self.add_special_tokens = False
        else:
            self._alphabet = esm.Alphabet(**alphabet_cfg)
            self.add_special_tokens = (
                self._alphabet.prepend_bos and self._alphabet.append_eos
            )

        self._featurizer = self.get_featurizer(featurizer, **featurizer_cfg)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._alphabet, name)
        except:
            raise AttributeError(f"{self.__class__} has no attribute `{name}`.")

    def __len__(self):
        return len(self._alphabet)

    def get_featurizer(self, name="cath", **kwds):
        if name == "cath":
            from .cath import Featurizer

            return Featurizer(
                alphabet=self,
                to_pifold_format=kwds.get("to_pifold_format", False),
                coord_nan_to_zero=kwds.get("coord_nan_to_zero", True),
            )
        elif name == "multichain":
            from .multichain import Featurizer

            return Featurizer(self, **kwds)

    @property
    def featurizer(self):
        return self._featurizer

    def featurize(self, raw_batch, **kwds):
        return self._featurizer(raw_batch, **kwds)

    def decode(self, batch_ids, return_as="str", remove_special=False):
        ret = []
        for ids in batch_ids.cpu():
            if return_as == "str":
                line = "".join([self.get_tok(id) for id in ids])
                if remove_special:
                    line = (
                        line.replace(self.get_tok(self.mask_idx), "_")
                        .replace(self.get_tok(self.eos_idx), "")
                        .replace(self.get_tok(self.cls_idx), "")
                        .replace(self.get_tok(self.padding_idx), "")
                        .replace(self.get_tok(self.unk_idx), "-")
                    )
            elif return_as == "list":
                line = [self.get_tok(id) for id in ids]
            ret.append(line)
        return ret
