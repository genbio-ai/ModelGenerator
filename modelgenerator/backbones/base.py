import collections
import dataclasses
import enum
import io
import os
import struct
import time
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple, Union

import filelock
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelgenerator.utils import GoogleKwargsDocstringInheritanceInitMeta
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from torch import Tensor
from torch.profiler import record_function


class LegacyAdapterType(enum.Enum):
    """Enum class for the types of legacy adapters provided by the backbones"""

    MASKED_LM = "MASKED_LM"
    TOKEN_CLS = "TOKEN_CLS"
    SEQ_CLS = "SEQ_CLS"


@dataclasses.dataclass
class DefaultConfig:
    """Used by tasks to inject default backbone configurations

    This class allows tasks to set deterministic default values for specific backbone arguments
    to help reduce redundant configurations.
    Only parameters with a clearly defined interface that are used by many backbones are intended
    to be modified. For this reason, `config_overwrites` is included, while `model_init_args` is
    excluded, as its values differ across backbones.

    For example, since a classification task already knows the number of classes, it can set the
    default for `num_labels` by
    `self.backbone_fn(DefaultConfig(config_overwrites={"num_labels": self.n_classes}))`.

    User can still override these default values by providing their own `config_overwrites`.
    Priority: user provided > task provided (through this class) > backbone default
    """

    config_overwrites: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SequenceBackboneOutput:
    last_hidden_state: Tensor
    hidden_states: Optional[List[Tensor]] = None
    attention_mask: Optional[Tensor] = None
    special_tokens_mask: Optional[Tensor] = None

    def __getitem__(self, item: Union[int, slice]) -> "SequenceBackboneOutput":
        """Allows indexing into the output object to get a new SequenceBackboneOutput."""
        hidden_states = None
        if self.hidden_states:
            hidden_states = [hs[item] for hs in self.hidden_states]

        attention_mask = None
        if self.attention_mask is not None:
            attention_mask = self.attention_mask[item]

        special_tokens_mask = None
        if self.special_tokens_mask is not None:
            special_tokens_mask = self.special_tokens_mask[item]

        return SequenceBackboneOutput(
            last_hidden_state=self.last_hidden_state[item],
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )

    @classmethod
    def concat(
        cls,
        outputs: List["SequenceBackboneOutput"],
    ) -> "SequenceBackboneOutput":
        """Concatenates a list of SequenceBackboneOutput objects into a single SequenceBackboneOutput."""
        max_length = max(o.last_hidden_state.size(1) for o in outputs)
        outputs_padded = []

        for o in outputs:
            padding_size = max_length - o.last_hidden_state.size(1)
            if padding_size == 0:
                outputs_padded.append(o)
                continue

            padded_last_hidden_state = F.pad(o.last_hidden_state, (0, 0, 0, padding_size))

            padded_attention_mask = None
            if o.attention_mask is not None:
                padded_attention_mask = F.pad(o.attention_mask, (0, padding_size))

            padded_special_tokens_mask = None
            if o.special_tokens_mask is not None:
                padded_special_tokens_mask = F.pad(o.special_tokens_mask, (0, padding_size))

            padded_hidden_states = None
            if o.hidden_states is not None:
                padded_hidden_states = [
                    F.pad(hs, (0, 0, 0, padding_size)) for hs in o.hidden_states
                ]

            outputs_padded.append(
                SequenceBackboneOutput(
                    last_hidden_state=padded_last_hidden_state,
                    hidden_states=padded_hidden_states,
                    attention_mask=padded_attention_mask,
                    special_tokens_mask=padded_special_tokens_mask,
                )
            )

        last_hidden_state = torch.cat([o.last_hidden_state for o in outputs_padded], dim=0)

        hidden_states = None
        if outputs[0].hidden_states is not None:
            hidden_states = [
                torch.cat(hs, dim=0) for hs in zip(*[o.hidden_states for o in outputs_padded])
            ]

        attention_mask = None
        if outputs[0].attention_mask is not None:
            attention_mask = torch.cat([o.attention_mask for o in outputs_padded], dim=0)

        special_tokens_mask = None
        if outputs[0].special_tokens_mask is not None:
            special_tokens_mask = torch.cat([o.special_tokens_mask for o in outputs_padded], dim=0)

        return cls(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )

    def to_device(
        self, device: torch.device, non_blocking: bool = False
    ) -> "SequenceBackboneOutput":
        """Moves the output tensors to the specified device."""
        hidden_states = None
        if self.hidden_states:
            hidden_states = [hs.to(device, non_blocking=non_blocking) for hs in self.hidden_states]

        attention_mask = None
        if self.attention_mask is not None:
            attention_mask = self.attention_mask.to(device, non_blocking=non_blocking)

        special_tokens_mask = None
        if self.special_tokens_mask is not None:
            special_tokens_mask = self.special_tokens_mask.to(device, non_blocking=non_blocking)

        return SequenceBackboneOutput(
            last_hidden_state=self.last_hidden_state.to(device, non_blocking=non_blocking),
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )

    def detach(self) -> "SequenceBackboneOutput":
        """Detaches the output tensors from the computation graph."""
        hidden_states = None
        if self.hidden_states:
            hidden_states = [hs.detach() for hs in self.hidden_states]

        attention_mask = None
        if self.attention_mask is not None:
            attention_mask = self.attention_mask.detach()

        special_tokens_mask = None
        if self.special_tokens_mask is not None:
            special_tokens_mask = self.special_tokens_mask.detach()

        return SequenceBackboneOutput(
            last_hidden_state=self.last_hidden_state.detach(),
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )

    def to_dict(self) -> dict:
        """Converts the output to a dictionary."""
        return {
            "last_hidden_state": self.last_hidden_state,
            "hidden_states": self.hidden_states,
            "attention_mask": self.attention_mask,
            "special_tokens_mask": self.special_tokens_mask,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceBackboneOutput":
        """Creates a SequenceBackboneOutput from a dictionary."""
        return cls(
            last_hidden_state=data["last_hidden_state"],
            hidden_states=data.get("hidden_states"),
            attention_mask=data.get("attention_mask"),
            special_tokens_mask=data.get("special_tokens_mask"),
        )


class SequenceBackboneInterface(nn.Module, metaclass=GoogleKwargsDocstringInheritanceInitMeta):
    """Interface class to ensure consistent implementation of essential methods for all backbones.

    Attributes:
        fsdp_wrap_modules: List of module paths to wrap when using distributed training with FSDP.
        model_path (str): Path to the model weights. May be HF.

    Args:
        enable_cache: Whether to enable caching for the backbone model.
        file_cache_dir: Directory to store the cache files.
        overwrite_file_cache: Whether to overwrite existing cache files.
        cache_write_buffer_size: Number of items to buffer before writing to disk. Only used when `enable_cache=True`.
        cache_storage_backend: The storage backend to use for caching when `enable_cache=True`, either 'lmdb' or 'indexed'.
    """

    # import paths of modules to wrap when using FSDP
    fsdp_wrap_modules: List[str] = []
    model_path: str = ""

    def __init__(
        self,
        enable_cache: bool = False,
        file_cache_dir: str = None,
        overwrite_file_cache: bool = False,
        cache_write_buffer_size: int = 1000,
        cache_storage_backend: Literal["lmdb", "indexed"] = "indexed",
    ) -> None:
        super().__init__()
        self.file_cache_dir = file_cache_dir or os.path.join(os.getcwd(), "backbone_cache")
        self.overwrite_file_cache = overwrite_file_cache
        self.cache_write_buffer_size = cache_write_buffer_size
        self.cache_storage_backend = cache_storage_backend
        self.cache = None
        self.cache_enabled = enable_cache
        if enable_cache:
            self.enable_cache()

    def enable_cache(self):
        """Enables caching for the backbone model."""
        if self.cache is None:
            self.cache = _BackboneCache(
                self,
                file_cache_dir=self.file_cache_dir,
                overwrite_file_cache=self.overwrite_file_cache,
                write_buffer_size=self.cache_write_buffer_size,
                storage_backend=self.cache_storage_backend,
            )
        self.forward = self.cache.forward
        self.process_batch = self.cache.process_batch
        self.required_data_columns = self.cache.required_data_columns
        self.cache_enabled = True

    def disable_cache(self, clear_cache: bool = True):
        """Disables caching for the backbone model."""
        self.forward = self.cache._orig_fwd
        self.process_batch = self.cache._orig_process_batch
        self.required_data_columns = self.cache._orig_required_data_columns
        if clear_cache and self.cache is not None:
            self.cache.clear()
        self.cache_enabled = False

    def clear_cache(self):
        """Clears the internal cache of the backbone model."""
        self.cache.clear()

    def setup(self):
        """Sets up the model, all expensive operations like model loading and initialization should be done here."""
        raise NotImplementedError

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        all_hidden_states: bool = False,
        **kwargs,
    ) -> SequenceBackboneOutput:
        """Defines the forward pass for the model.

        Args:
            input_ids (Tensor): Token IDs (n, seq_len).
            attention_mask (Optional[Tensor]): Attention mask (n, seq_len).
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            SequenceBackboneOutput: Model output, including last hidden state and other relevant data.
        """
        raise NotImplementedError

    def process_batch(
        self,
        batch: dict,
        device: torch.device,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        """Processes a batch of sequences to model input format.

        Args:
            batch (List[str]): List of input sequences.
            device (torch.device): Device to move the data to.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        raise NotImplementedError

    def required_data_columns(self) -> List[str]:
        """List of required data columns for the model.

        Returns:
            List[str]: List of required data columns.
        """
        return ["sequences"]

    def get_decoder(self) -> nn.Module:
        """Returns the decoder module for the model, if applicable.

        Returns:
            nn.Module: The decoder module.
        """
        raise NotImplementedError

    def tokenize(
        self,
        sequences: List[str],
        padding: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Tokenizes input sequences into input IDs and attention masks.

        Args:
            sequences (List[str]): List of input sequences.
            padding (bool, optional): Whether to pad sequences. Defaults to True.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            dict: A dictionary containing input_ids.
        """
        raise NotImplementedError

    def decode_tokens(self, tokenized_sequences: Tensor) -> List[str]:
        """Decodes tokenized sequences back to text.

        Args:
            tokenized_sequences (Tensor): Tokenized sequences.

        Returns:
            List[str]: Decoded text sequences.
        """
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        """Gets the ID of a specific token.

        Args:
            token (str): The token to look up.

        Returns:
            int: Token ID.
        """
        raise NotImplementedError

    def get_max_context(self) -> int:
        """Gets the maximum context length of the model.

        Returns:
            int: Maximum context length.
        """
        raise NotImplementedError

    def get_embedding_size(self) -> int:
        """Gets the embedding size of the model.

        Returns:
            int: Embedding size.
        """
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Gets the vocabulary size of the model.

        Returns:
            int: Vocabulary size.
        """
        raise NotImplementedError

    def on_save_checkpoint(self, checkpoint: dict):
        """Handles checkpoint saving logic for the model.

        Args:
            checkpoint (dict): The checkpoint dictionary.
        """
        raise NotImplementedError

    def get_num_layer(self) -> int:
        """Gets the number of layers in the model.

        Returns:
            int: Number of layers.
        """
        raise NotImplementedError


class HFSequenceBackbone(SequenceBackboneInterface):
    """Base class for all backbone models

    Note:
        The required possitional arguments are reserved by downstream tasks for dependency injection and cannot
        be changed by the user.

    Args:
        legacy_adapter_type: Ignore. Reserved for use by `use_legacy_adapter` in Tasks.
        default_config: Ignore. Reserved for use by `use_legacy_adapter` in Tasks.
        config_overwrites: Optional model arguments for PretrainedConfig.
        model_init_args: Optional model arguments passed to its init method.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[dict, None],
        /,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.legacy_adapter_type = legacy_adapter_type
        self.default_config = default_config or DefaultConfig()
        self.config_overwrites = config_overwrites or {}
        self.model_init_args = model_init_args or {}
        # User provided configs always takes precedence
        self.config_overwrites = {
            **self.default_config.config_overwrites,
            **self.config_overwrites,
        }
        if self.use_legacy_adapter:
            rank_zero_info(
                "You are using a legacy adapter/head, so its configuration has to be "
                "set explicitly under backbone. This is done using "
                "`model.backbone.config_overwrites` and `model.backbone.model_init_args`."
            )

    @property
    def use_legacy_adapter(self) -> bool:
        """Whether to use a legacy adapter"""
        return self.legacy_adapter_type is not None


class _BackboneCache:
    """
    Provide caching functionality for a backbone.

    Args:
        module: The PyTorch module to wrap.
        file_cache_dir: Directory to store the cache files.
        overwrite_file_cache: Whether to overwrite existing cache files.
        write_buffer_size: Number of items to buffer before writing to disk.
        storage_backend: The storage backend to use for caching, either 'lmdb' or 'indexed'.
    """

    def __init__(
        self,
        module: nn.Module,
        file_cache_dir: str,
        overwrite_file_cache: bool = False,
        write_buffer_size: int = 1000,
        storage_backend: str = "indexed",
    ) -> None:
        super().__init__()
        self._module = module
        self._orig_fwd = module.forward
        self._orig_process_batch = module.process_batch
        self._orig_required_data_columns = module.required_data_columns
        self.profiler = _CacheProfiler()
        if storage_backend == "lmdb":
            self._store = _LMDBStore(
                path=file_cache_dir,
                write_buffer_size=write_buffer_size,
                profiler=self.profiler,
            )
        elif storage_backend == "indexed":
            self._store = _IndexedStore(
                file_cache_dir, write_buffer_size=write_buffer_size, profiler=self.profiler
            )
        if overwrite_file_cache:
            self.clear()
        self._warned = False
        self._forward_count = 0

    def forward(self, uid: list | torch.Tensor = None, **kwargs):
        """Forward pass wrapper that caches outputs based on unique identifiers (uid)."""
        if uid is None:
            raise ValueError("uid must be provided when caching is enabled.")
        if self._forward_count % 1000 == 0:
            if not all(not p.requires_grad for p in self._module.parameters()):
                raise ValueError(
                    "Caching is only supported for models with no trainable parameters."
                )
        self._forward_count += 1
        if torch.is_tensor(uid):
            uid = uid.tolist()
        uid = [str(u) for u in uid]
        output_device = next((arg.device for arg in kwargs.values() if torch.is_tensor(arg)), None)
        with self.profiler.time("cache_lookup", count=len(uid)):
            uncached_indices = [i for i, k in enumerate(uid) if k not in self._store]
        if uncached_indices:
            with self.profiler.time("cache_miss", count=len(uncached_indices)):
                inputs = self._gather_uncached_inputs(kwargs, uncached_indices)
                with torch.no_grad():
                    outputs = self._orig_fwd(**inputs).detach().to_device("cpu", non_blocking=True)
                for output_idx, uid_idx in enumerate(uncached_indices):
                    # keep batch dimension
                    self._store[uid[uid_idx]] = outputs[output_idx : output_idx + 1]
        cached_outputs = [self._store[k].to_device(output_device, non_blocking=True) for k in uid]
        if not isinstance(cached_outputs[0], SequenceBackboneOutput):
            raise NotImplementedError(f"Support for {type(cached_outputs[0])} is not implemented.")
        return SequenceBackboneOutput.concat(cached_outputs)

    def _gather_uncached_inputs(self, kwargs, uncached_indices):
        inputs = {}
        for k, v in kwargs.items():
            # Select only the uncached inputs for the forward pass
            if torch.is_tensor(v):
                inputs[k] = v[uncached_indices]
            elif isinstance(v, list):
                inputs[k] = [v[i] for i in uncached_indices]
            elif isinstance(v, np.ndarray):
                inputs[k] = v[uncached_indices]
            else:
                inputs = kwargs
                if not self._warned:
                    rank_zero_warn(
                        f"Unable to gather uncached inputs for {k} of type {type(v)}. "
                        "Full batch is used for forward pass."
                    )
                    self._warned = True
                break
        return inputs

    def process_batch(
        self,
        batch: dict,
        device: torch.device,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        """Wrapper for the original process_batch method that ensures 'uid' is included in the batch."""
        result = self._orig_process_batch(batch, device, **kwargs)
        if "uid" not in result:
            result["uid"] = batch.get("uid", None)
        if result["uid"] is None:
            raise ValueError("Batch must contain 'uid' for caching to work.")
        return result

    def required_data_columns(self) -> List[str]:
        """Wrapper for the original required_data_columns method that adds 'uid' to required columns."""
        result = self._orig_required_data_columns()
        if "uid" not in result:
            result.append("uid")
        return result

    def clear(self):
        """Clears the internal cache."""
        self._store.clear()

    def flush(self, reset_readers: bool = True):
        """Flushes the cache to disk."""
        self._store.flush(reset_readers=reset_readers)

    def reset_readers(self):
        """Resets the data and index readers in the cache."""
        self._store.reset_readers()

    @property
    def should_flush(self) -> bool:
        """Whether the cache should be flushed."""
        return self._store.should_flush

    def get_stats(self) -> dict:
        """Returns the cache statistics."""
        misses = self.profiler.timings["cache_miss"]["count"]
        lookups = self.profiler.timings["cache_lookup"]["count"]
        stats = {
            "cache_miss_rate": misses / lookups if lookups > 0 else 0,
            "cache_avg_miss_time": self.profiler.timings["cache_miss"]["avg"],
            "cache_avg_lookup_time": self.profiler.timings["cache_lookup"]["avg"],
            # Max times reflect the slowest per-batch average time, not per-sample latency
            "cache_max_miss_time": self.profiler.timings["cache_miss"]["max"],
            "cache_max_lookup_time": self.profiler.timings["cache_lookup"]["max"],
        }
        return {**stats, **self._store.get_stats()}


class _IndexedStore:
    """A simple append-only storage with offset-based indexing."""

    def __init__(
        self,
        path: str,
        write_buffer_size: int = 1000,
        profiler: Optional["_CacheProfiler"] = None,
    ):
        self.path = path
        self.write_buffer_size = write_buffer_size
        self.write_buffer = {}
        self.profiler = profiler or _CacheProfiler()

        self._binfile = os.path.join(path, "data.bin")
        self._idxfile = os.path.join(path, "data.idx")
        self._lockfile = os.path.join(path, "data.lock")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if not os.path.exists(self._binfile):
            with open(self._binfile, "wb"):
                pass
        self._idx_env = lmdb.open(self._idxfile, map_size=2**31)
        self._data_reader = None
        self._idx_reader = None
        self.reset_readers()

    def reset_readers(self):
        """Reset all readers."""
        if self._data_reader:
            self._data_reader.close()
        self._data_reader = open(self._binfile, "rb")
        if self._idx_reader:
            del self._idx_reader
        self._idx_reader = self._idx_env.begin(write=False)

    def __getitem__(self, key: str) -> SequenceBackboneOutput:
        """Retrieve and decode an item from the store by its key."""
        if key in self.write_buffer:
            return self.write_buffer[key]
        value = self._idx_reader.get(key.encode("utf-8"))
        if value is None:
            raise KeyError(f"Key {key} not found in cache storage.")
        offset, length = struct.unpack(">QQ", value)

        self._data_reader.seek(offset)
        with self.profiler.time("cache_read", count=1):
            raw_bytes = self._data_reader.read(length)
        with self.profiler.time("cache_decode", count=1):
            data = torch.load(io.BytesIO(raw_bytes), weights_only=True)
        return SequenceBackboneOutput.from_dict(data)

    def __setitem__(self, key: str, value: SequenceBackboneOutput):
        """Store an item in the store with the given key."""
        self.write_buffer[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if key in self.write_buffer:
            return True
        return self._idx_reader.get(key.encode("utf-8")) is not None

    def flush(self, reset_readers: bool = True):
        """Flush the write buffer to disk."""
        if not self.write_buffer:
            return
        with filelock.FileLock(self._lockfile):
            with self.profiler.time("cache_flush", count=len(self.write_buffer)):
                with open(self._binfile, "ab") as binfile:
                    print(f"Flushing {len(self.write_buffer)} items to cache store...")
                    with self._idx_env.begin(write=True) as txn:
                        while self.write_buffer:
                            key, value = self.write_buffer.popitem()
                            if txn.get(key.encode("utf-8")) is not None:
                                # individual item updates not supported by design
                                continue
                            offset = binfile.tell()
                            # Write the value to the binary file
                            torch.save(value.to_dict(), binfile)
                            length = binfile.tell() - offset
                            # Store the offset and length in the index
                            txn.put(key.encode("utf-8"), struct.pack(">QQ", offset, length))
        if reset_readers:
            self.reset_readers()

    def clear(self):
        """Clear the store by removing all items and resetting the index."""
        if os.path.exists(self._binfile):
            with filelock.FileLock(self._lockfile):
                open(self._binfile, "wb").close()
        with self._idx_env.begin(write=True) as txn:
            txn.drop(self._idx_env.open_db(), delete=False)
        self.write_buffer.clear()
        self.reset_readers()

    @property
    def should_flush(self) -> bool:
        """Whether a flush is needed."""
        return len(self.write_buffer) > self.write_buffer_size

    def get_stats(self) -> dict:
        """Returns the storage statistics."""
        stats = {
            "cache_store_avg_read_time": self.profiler.timings["cache_read"]["avg"],
            "cache_store_avg_decode_time": self.profiler.timings["cache_decode"]["avg"],
            "cache_store_avg_flush_time": self.profiler.timings["cache_flush"]["avg"],
            # Max times reflect the slowest per-batch average time, not per-sample latency
            "cache_store_max_read_time": self.profiler.timings["cache_read"]["max"],
            "cache_store_max_decode_time": self.profiler.timings["cache_decode"]["max"],
        }
        return stats


class _LMDBStore:
    """A simple LMDB storage for caching backbone outputs."""

    def __init__(
        self,
        path: str,
        map_size: int = 2**40,
        write_buffer_size: int = 1000,
        profiler: Optional["_CacheProfiler"] = None,
    ):
        self.env: lmdb.Environment = lmdb.open(path, map_size=map_size)
        self.write_buffer_size = write_buffer_size
        self.write_buffer = {}
        self.profiler = profiler or _CacheProfiler()
        self._reader = None
        self.reset_readers()

    def reset_readers(self):
        """Resets the LMDB reader."""
        if self._reader:
            del self._reader
        self._reader = self.env.begin(write=False)

    def __getitem__(self, key: str) -> SequenceBackboneOutput:
        """Retrieve an item from the LMDB store."""
        if key in self.write_buffer:
            return self.write_buffer[key]
        raw_bytes = self._reader.get(key.encode("utf-8"))
        if raw_bytes is not None:
            with self.profiler.time("cache_decode", count=1):
                data = torch.load(io.BytesIO(raw_bytes), weights_only=True)
            return SequenceBackboneOutput.from_dict(data)
        raise KeyError(f"Key {key} not found in LMDB store.")

    def __setitem__(self, key: str, value: SequenceBackboneOutput):
        """Store an item in the LMDB store."""
        self.write_buffer[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the LMDB store."""
        if key in self.write_buffer:
            return True
        with self.profiler.time("cache_read", count=1):
            return self._reader.get(key.encode("utf-8")) is not None

    def flush(self, reset_readers: bool = True):
        """Flush the write buffer to disk."""
        if not self.write_buffer:
            return
        with self.env.begin(write=True) as txn:
            with self.profiler.time("cache_flush", count=len(self.write_buffer)):
                print(f"Flushing {len(self.write_buffer)} items to cache store...")
                while self.write_buffer:
                    key, value = self.write_buffer.popitem()
                    bytes_io = io.BytesIO()
                    torch.save(value.to_dict(), bytes_io)
                    txn.put(key.encode("utf-8"), bytes_io.getvalue())

        if reset_readers:
            self.reset_readers()

    def clear(self):
        """Clear the LMDB store and all the buffers."""
        with self.env.begin(write=True) as txn:
            txn.drop(self.env.open_db(), delete=False)
        self.write_buffer.clear()
        self.reset_readers()

    @property
    def should_flush(self) -> bool:
        """Whether a flush is needed."""
        return len(self.write_buffer) >= self.write_buffer_size

    def get_stats(self) -> dict:
        """Returns the storage statistics."""
        stats = {
            # Read time may not reflect actual disk I/O due to memory mapping
            "cache_store_avg_read_time": self.profiler.timings["cache_read"]["avg"],
            "cache_store_avg_decode_time": self.profiler.timings["cache_decode"]["avg"],
            # Flush time does not include final .commit() and may be inaccurate
            "cache_store_avg_flush_time": self.profiler.timings["cache_flush"]["avg"],
            # Max times reflect the slowest per-batch average time, not per-sample latency
            "cache_store_max_read_time": self.profiler.timings["cache_read"]["max"],
            "cache_store_max_decode_time": self.profiler.timings["cache_decode"]["max"],
        }
        return stats


class _CacheProfiler:
    """A simple profiler for the cache operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the timings."""
        self.timings = collections.defaultdict(
            lambda: {"count": 0, "total": 0.0, "avg": 0.0, "max": 0.0}
        )

    def record_duration(self, name: str, duration: float, count: int = 1):
        """Records the duration of an event."""
        timing = self.timings[name]
        timing["count"] += count
        timing["total"] += duration
        timing["avg"] = timing["total"] / timing["count"]
        timing["max"] = max(timing["max"], duration / count)

    @contextmanager
    def time(self, name: str, count: int = 1):
        """Context manager to time a block of code."""
        start = time.perf_counter()
        with record_function(f"mgen::embedding::{name}"):
            yield
        elapsed = time.perf_counter() - start
        self.record_duration(name, elapsed, count=count)
