import dataclasses
import enum
from typing import Union, Optional, List, Tuple

import torch.nn as nn
from torch import Tensor
from docstring_inheritance import GoogleDocstringInheritanceInitMeta


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


class SequenceBackboneInterface(nn.Module, metaclass=GoogleDocstringInheritanceInitMeta):
    """Interface class to ensure consistent implementation of essential methods for all backbones.

    Attributes:
        fsdp_wrap_modules: List of module paths to wrap when using distributed training with FSDP.
        model_path (str): Path to the model weights. May be HF.
    """

    # import paths of modules to wrap when using FSDP
    fsdp_wrap_modules: List[str] = []
    model_path: str = ""

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        """Defines the forward pass for the model.

        Args:
            input_ids (Tensor): Token IDs (n, seq_len).
            attention_mask (Tensor): Attention mask (n, seq_len).
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, List[Tensor]]: Model output, typically the last hidden state or logits.
        """
        raise NotImplementedError

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
    ):
        super().__init__()
        self.legacy_adapter_type = legacy_adapter_type
        self.default_config = default_config or DefaultConfig()
        # TODO: move config_overwrites and default_config.config_overwrites
        # to a separate base class as they are huggingface specific
        self.config_overwrites = config_overwrites or {}
        self.model_init_args = model_init_args or {}
        # User provided configs always takes precedence
        self.config_overwrites = {
            **self.default_config.config_overwrites,
            **self.config_overwrites,
        }

    @property
    def use_legacy_adapter(self) -> bool:
        """Whether to use a legacy adapter"""
        return self.legacy_adapter_type is not None
