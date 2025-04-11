# coding=utf-8
# Copyright 2021- NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" FM4BIO model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional

logger = logging.get_logger(__name__)

FM4BIO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "": "https://huggingface.co//resolve/main/config.json",
}


class FM4BioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FM4BioModel`]. It is used to instantiate a
    FM4BIO model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FM4BIO
    [nvidia/fm4bio-uncased-345m](https://huggingface.co/nvidia/fm4bio-uncased-345m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 29056):
            Vocabulary size of the FM4BIO model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`FM4BioModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`FM4BioModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Examples:

    ```python
    >>> from transformers import FM4BioConfig, FM4BioModel

    >>> # Initializing a FM4BIO bert-base-uncased style configuration
    >>> configuration = FM4BioConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = FM4BioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fm4bio"

    def __init__(
        self,
        vocab_size=128,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="swiglu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=2048,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=0,
        add_linear_bias=True,
        position_embedding_type="rope",
        normalization_type="RMSNorm",
        apply_residual_connection_post_layernorm=False,
        use_cache=True,
        rotary_percent=1.0,
        seq_len_interpolation_factor=None,
        moe=False,
        num_experts=0,
        experts_per_token=0,
        use_lm_head=True,
        tie_word_embeddings=True,
        output_vocab_size: int = None,  # when set, the output vocab size is different from the input vocab size
        gradient_checkpointing=False,   # Gradient checkpoint for memory saving
        str_embedding_in: Optional[int] = None, # Structure embedding input dimension
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.add_linear_bias = add_linear_bias
        assert normalization_type in [
            "RMSNorm",
            "LayerNorm",
        ], "normalization_type must be 'RMSNorm' or 'LayerNorm'"
        self.normalization_type = normalization_type
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.rotary_percent = rotary_percent
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.moe = moe
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.use_lm_head = use_lm_head
        self.tie_word_embeddings = tie_word_embeddings
        self.output_vocab_size = output_vocab_size
        self.gradient_checkpointing = gradient_checkpointing 
        self.str_embedding_in = str_embedding_in