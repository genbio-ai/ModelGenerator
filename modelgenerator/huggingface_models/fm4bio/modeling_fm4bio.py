# coding=utf-8
# Copyright 2024 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch MegatronBERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import sys
from functools import partial

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, HuberLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_fm4bio import FM4BioConfig
from collections import namedtuple


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FM4BioConfig"
_CHECKPOINT_FOR_DOC = ""

FM4BIO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "",
    # See all FM4Bio models at https://huggingface.co/models?filter=fm4bio
]

if sys.platform != "darwin":
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)
DeepNormCoefficients = namedtuple("DeepNormCoefficients", ["alpha", "beta"])

def get_checkpoint_fn():
    checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    return checkpoint

class FM4BioEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        if config.position_embedding_type not in ("rope", "rope_2d"):
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        
        if isinstance(config.str_embedding_in, str):
            if config.str_embedding_in.isdigit():
                self.str_embedding_in = int(config.str_embedding_in)
            else:
                self.str_embedding_in = None
        else:
            self.str_embedding_in = config.str_embedding_in
        
        if self.str_embedding_in is not None and self.str_embedding_in > 0:
            self.str_embeddings = nn.Linear(self.str_embedding_in, config.hidden_size, bias=False)

        # In Megatron, layer-norm is applied after the 1st dropout.
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "rope"
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        inputs_str_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.str_embedding_in is not None and self.str_embedding_in > 0:
            if inputs_str_embeds is None:
                print(f"Warning: str_embedding_in={self.str_embedding_in}, but inputs_str_embeds is None")
            else:
                # inputs_str_embeds: [B, L, D1]
                # inputs_embeds: [B, L, D]
                shape = f"inputs_str_embeds.shape={inputs_str_embeds.shape}, inputs_embeds.shape={inputs_embeds.shape}"
                assert inputs_str_embeds.ndim == 3, shape
                assert inputs_str_embeds.shape[0] == inputs_embeds.shape[0], shape
                assert inputs_str_embeds.shape[1] == inputs_embeds.shape[1], shape
                inputs_embeds = inputs_embeds + self.str_embeddings(inputs_str_embeds)

        # embeddings = inputs_embeds + token_type_embeddings
        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Megatron BERT moves that layer norm after the drop-out (and to each layer).
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->FM4Bio
class FM4BioSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.add_linear_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.add_linear_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.add_linear_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        rotary_pos_emb=None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # [b, hn, sq, c]
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb

            # [b, hn, sq, c] --> [sq, b, hn, c]
            query_layer = query_layer.permute(2, 0, 1, 3).contiguous()
            key_layer = key_layer.permute(2, 0, 1, 3).contiguous()
            
            if q_pos_emb.ndim == 5: ## 2d rope
                dim          = query_layer.shape[-1]
                query_layer1 = apply_rotary_pos_emb(query_layer[..., :dim//2], q_pos_emb[0])  
                query_layer2 = apply_rotary_pos_emb(query_layer[..., dim//2:], q_pos_emb[1])
                query_layer  = torch.cat([query_layer1, query_layer2], axis=-1)
                
                dim         = key_layer.shape[-1]
                key_layer1  = apply_rotary_pos_emb(key_layer[..., :dim//2], k_pos_emb[0])
                key_layer2  = apply_rotary_pos_emb(key_layer[..., dim//2:], k_pos_emb[1])
                key_layer   = torch.cat([key_layer1, key_layer2], axis=-1)
            else: ## 1d rope
                query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
                key_layer   = apply_rotary_pos_emb(key_layer, k_pos_emb)

            # [sq, b, hn, c] --> [b, hn, sq, c]
            query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
            key_layer = key_layer.permute(1, 2, 0, 3).contiguous()

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
        
        if output_attentions or head_mask is not None:
            # Don't use FA
            
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in FM4BioModel forward() function)
                attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            no_prob_mask = attention_mask < -1e-5
            attention_probs = attention_probs.masked_fill(no_prob_mask, 0.0)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)
            
            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
                
            context_layer = torch.matmul(attention_probs, value_layer)
            
        else:
            
            if attention_mask is not None:
                if attention_mask.shape[0] == 1: # Batch size = 1
                    attention_mask = None
                else:
                    attention_mask = attention_mask.clone()
                    if torch.is_floating_point(attention_mask):
                        attention_mask[attention_mask < -1e-5] = torch.finfo(attention_mask.dtype).min
                    if torch.allclose(attention_mask, torch.zeros_like(attention_mask)):
                        attention_mask = None

            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=attention_mask, dropout_p=self.dropout.p, is_causal=False, scale=None, enable_gqa=False)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Based transformers.models.bert.modeling_bert.BertSelfOutput. Moved LayerNorm to FM4BioAttention below.
class FM4BioSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.add_linear_bias
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


# Based transformers.models.bert.modeling_bert.BertAttention. Added LayerNorm.
class FM4BioAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = config.norm_cls(config.hidden_size, eps=config.layer_norm_eps)

        self.self = FM4BioSelfAttention(config)
        self.output = FM4BioSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        rotary_pos_emb=None,
        apply_residual_connection_post_layernorm: bool = False, 
        deepnorm_alpha: float = 1.0,
    ) -> Tuple[torch.Tensor]:
        # debug_point1 = hidden_states[0]
        ln_outputs = self.ln(hidden_states)
        self_outputs = self.self(
            ln_outputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rotary_pos_emb,
        )

        if apply_residual_connection_post_layernorm:
            # 3B dense
            attention_output = self.output(self_outputs[0], deepnorm_alpha * ln_outputs)
        else:
            # 16B MoE
            attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


def _config_to_kwargs(args):
    if isinstance(args.torch_dtype, str):
        torch_dtype = eval(args.torch_dtype)
    else:
        torch_dtype = args.torch_dtype
    common_kwargs = {
        "dtype": torch_dtype,
    }
    return common_kwargs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->FM4Bio
class FM4BioMLP(nn.Module):
    # def __init__(self, config: FM4BioConfig):
    #     super().__init__()
    #     assert config.hidden_act == "swiglu", "Only swiglu is supported."
    #     self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.add_linear_bias)
    #     self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.add_linear_bias)
    #     self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.add_linear_bias)
    #     self.intermediate_act_fn = ACT2FN['silu'] # swiglu use silu as part of its activation function

    # def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    #     down_proj = self.down_proj(self.intermediate_act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
    #     return down_proj

    def __init__(self, config: FM4BioConfig, device=None):
        super(FM4BioMLP, self).__init__()

        self.add_bias = config.add_linear_bias
        self.moe = config.moe
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token  # 2

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config),
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return x[0] * F.silu(x[1])

        def geglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return x[0] * F.gelu(x[1])

        if config.hidden_act == "geglu":
            self.activation_func = geglu
        elif config.hidden_act == "swiglu":
            self.activation_func = swiglu
        else:
            assert RuntimeError(f"Unsupported glu_activation: {config.hidden_act}")

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config),
        )

        if self.moe:
            assert self.num_experts > 1
            del self.dense_h_to_4h
            del self.dense_4h_to_h
            self.router = nn.Linear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            for i in range(0, self.num_experts):
                self.register_module(
                    f"dense_h_to_4h_{i}",
                    nn.Linear(
                        config.hidden_size,
                        config.intermediate_size * 2,
                        bias=self.add_bias,
                        device=device,
                        **_config_to_kwargs(config),
                    ),
                )
                self.register_module(
                    f"dense_4h_to_h_{i}",
                    nn.Linear(
                        config.intermediate_size,
                        config.hidden_size,
                        bias=self.add_bias,
                        device=device,
                        **_config_to_kwargs(config),
                    ),
                )

    def moe_forward(self, hidden_states, expert_idx):
        intermediate_parallel = getattr(self, f"dense_h_to_4h_{expert_idx}")(
            hidden_states
        )
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = getattr(self, f"dense_4h_to_h_{expert_idx}")(intermediate_parallel)
        return output

    def forward(self, hidden_states):
        if self.moe:
            # import pdb; pdb.set_trace();
            s, b, n = hidden_states.shape
            dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_states.size(2))  # [s*b h]
            self.router = self.router.float()
            route = self.router(hidden_states.float()).to(dtype)

            weights, selected_experts = torch.topk(route, self.experts_per_token)
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(
                hidden_states.dtype
            )
            output = torch.zeros_like(
                hidden_states, dtype=hidden_states.dtype, device=hidden_states.device
            )
            for expert_idx in range(self.num_experts):
                batch_idx, nth_expert = torch.where(selected_experts == expert_idx)
                if nth_expert.shape[0] == 0:
                    continue
                cur_out = self.moe_forward(hidden_states[batch_idx], expert_idx)
                output[batch_idx] += weights[batch_idx, nth_expert, None] * cur_out
            output = output.reshape(s, b, n)
        else:
            # [s, b, 4hp]
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
            # [s, b, h]
            output = self.dense_4h_to_h(intermediate_parallel)
        return output


# Based on transformers.models.bert.modeling_bert.BertOutput. Moved LayerNorm to FM4BioLayer below.
class FM4BioOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return input_tensor + hidden_states


# Based on transformers.models.bert.modeling_bert.BertLayer. Added LayerNorm.
class FM4BioLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FM4BioAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.deepnorm_alpha = (2 * config.num_hidden_layers) ** 0.5
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = FM4BioAttention(config)
        self.ln = config.norm_cls(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = FM4BioMLP(config)
        self.output = FM4BioOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        rotary_pos_emb=None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rotary_pos_emb=rotary_pos_emb,
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            deepnorm_alpha=self.deepnorm_alpha
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # debug: attention_output[0]
        ln_output = self.ln(attention_output)
        mlp_output = self.mlp(ln_output)
        if self.apply_residual_connection_post_layernorm:
            layer_output = self.output(mlp_output, self.deepnorm_alpha * ln_output)
        else:
            layer_output = self.output(mlp_output, attention_output)
        return layer_output


class RnaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        same as LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

ALL_LAYERNORM_LAYERS.append(RnaRMSNorm)


class FM4BioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [FM4BioLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # The final layer norm. We removed the 1st LN, moved LN to each hidden layer and this one
        # is simply the final LN (Transformer's BERT has it attached to each hidden layer).
        self.ln = config.norm_cls(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        rotary_pos_emb: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        # print(f"self.gradient_checkpointing={self.gradient_checkpointing}, torch.is_grad_enabled()={torch.is_grad_enabled()}")
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and torch.is_grad_enabled():
                layer_outputs = get_checkpoint_fn()(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rotary_pos_emb,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rotary_pos_emb,
                )

            # Because we moved the layer-norm at the end of the hidden layer, we have non-normali-
            # zed data here. If that's really needed, we must apply LN to match Transformer's BERT.

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Finalize the hidden states.
        hidden_states = self.ln(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->FM4Bio
class FM4BioPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.add_linear_bias
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->FM4Bio
class FM4BioPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size
        )  # in megatron, this will always have bias

        self.transform_act_fn = ACT2FN["gelu"]

        if config.normalization_type == "RMSNorm":
            self.LayerNorm = RnaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->FM4Bio
class FM4BioLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = FM4BioPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->FM4Bio
class FM4BioOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = FM4BioLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->FM4Bio
class FM4BioPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = FM4BioLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FM4BioPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FM4BioConfig
    # load_tf_weights = load_tf_weights_in_fm4bio
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "FM4BioLayer",
        "FM4BioEmbeddings",
        "FM4BioMLP",
    ]  # should not be on different machines

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RnaRMSNorm):
            module.weight.data.fill_(1.0)
            # no bias
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
# Copied from transformers.models.bert.modeling_bert.BertForPreTrainingOutput with Bert->FM4Bio
class FM4BioForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FM4BioForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


FM4BIO_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FM4BioConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FM4BIO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare FM4Bio Model transformer outputting raw hidden-states without any specific head on top.",
    FM4BIO_START_DOCSTRING,
)
class FM4BioModel(FM4BioPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        if config.normalization_type == "RMSNorm":
            self.config.norm_cls = RnaRMSNorm
        else:
            assert config.normalization_type == "LayerNorm"
            self.config.norm_cls = nn.LayerNorm
        self.embeddings = FM4BioEmbeddings(config)
        self.encoder = FM4BioEncoder(config)

        self.pooler = FM4BioPooler(config) if add_pooling_layer else None

        # rotary position embeddings
        if config.position_embedding_type == "rope":
            rotary_dim = config.hidden_size // config.num_attention_heads

            # partial rotary embeddings, which is better than full rotary
            # Wang and Komatsuzaki et al
            # https://github.com/kingoflolz/mesh-transformer-jax/
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim, config.rotary_percent)

        elif config.position_embedding_type == "rope_2d":
            rotary_dim = config.hidden_size // config.num_attention_heads // 2
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim, config.rotary_percent)

        # delete this from config so the config can be successfully saved
        del self.config.norm_cls

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        FM4BIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_str_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        extended_attention_mask = bert_extended_attention_mask(
            attention_mask
        )  # True for pad, false for non-pad
        extended_attention_mask = extended_attention_mask * torch.finfo(torch.float).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) 

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.config.position_embedding_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(input_ids.size(1))
        elif self.config.position_embedding_type == 'rope_2d':
            # input_ids: [1, 12800]
            # position_ids: [B, 2, 12800]
            rotary_pos_emb = self.rotary_pos_emb(input_ids.size(1)).squeeze(1) # [12800, 1, 1, D//H] -> [12800, 1, D//H//2]
            rotary_pos_emb = rotary_pos_emb[ position_ids ] # [12800, 1, D//H] -> [B, 2, 12800, 1, D//H//2]
            rotary_pos_emb = rotary_pos_emb.permute([1,2,0,3,4]) # [2, 12800, B, 1, D//H//2]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            inputs_str_embeds=inputs_str_embeds,
            past_key_values_length=past_key_values_length,
        ) 
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rotary_pos_emb=rotary_pos_emb,
        ) 
        
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    FM4Bio Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    FM4BIO_START_DOCSTRING,
)
class FM4BioForPreTraining(FM4BioPreTrainedModel):
    # _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config, add_binary_head=True):
        super().__init__(config)

        self.bert = FM4BioModel(config)
        self.cls = FM4BioPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        FM4BIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=FM4BioForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_str_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        next_sentence_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FM4BioForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FM4BioForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("")
        >>> model = FM4BioForPreTraining.from_pretrained("")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            inputs_str_embeds=inputs_str_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FM4BioForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """FM4Bio Model with a `language modeling` head on top.""", FM4BIO_START_DOCSTRING
)
class FM4BioForMaskedLM(FM4BioPreTrainedModel):
    # _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `FM4BioForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = FM4BioModel(config, add_pooling_layer=False)
        self.use_lm_head = config.use_lm_head
        if config.use_lm_head:
            self.cls = FM4BioOnlyMLMHead(config)
        else:
            if getattr(config, "output_vocab_size", None) is not None:
                # used when the output uses a different vocab
                # e.g., input vocab is amino acids, output vocab is structure tokens
                self.output_embed = nn.Linear(
                    config.hidden_size, config.output_vocab_size, bias=False
                )
            else:
                self.output_embed = nn.Linear(
                    config.hidden_size, config.vocab_size, bias=False
                )

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        if self.use_lm_head:
            return self.cls.predictions.decoder
        else:
            return self.output_embed

    def set_output_embeddings(self, new_embeddings):
        if self.use_lm_head:
            self.cls.predictions.decoder = new_embeddings
        else:
            raise NotImplementedError

    @add_start_docstrings_to_model_forward(
        FM4BIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_str_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            inputs_str_embeds=inputs_str_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.use_lm_head:
            prediction_scores = self.cls(sequence_output)
        else:
            prediction_scores = self.output_embed(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to 10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = (
            torch.arange(
                max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]

        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f"{prefix}inv_freq", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _rotate_half(x: Tensor) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype).to(t.device)
    sin_ = torch.sin(freqs).to(t.dtype).to(t.device)

    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = extended_attention_mask < 0.5

    return extended_attention_mask


class FM4BioForSequenceClassification(FM4BioPreTrainedModel):
    def __init__(
        self,
        config,
        arch="MLP",
        pooling="mean_pooling",
        conv_kernel_size=9,
        dropout_prob=None,
        augment_with_zeroshot=False,
        inter_hidden_size=None,
        activation_func="tanh",
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pooling = pooling
        self.augment_with_zeroshot = augment_with_zeroshot
        self.inter_hidden_size = inter_hidden_size

        self.bert = FM4BioModel(config, add_pooling_layer=False)
        self.classifier = FM4BioClassificationHead(
            config,
            arch,
            pooling,
            conv_kernel_size,
            dropout_prob,
            inter_hidden_size,
            augment_with_zeroshot,
            activation_func,
        )

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        FM4BIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # (bs, seq_len), 0 means masking
        zero_shot_fitness_predictions: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_str_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            inputs_str_embeds=inputs_str_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (bs, seq_len, hidden_size)
        logits = self.classifier(
            sequence_output, attention_mask, zero_shot_fitness_predictions
        )

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FM4BioForTokenClassification(FM4BioPreTrainedModel):
    def __init__(
        self,
        config,
        arch="MLP",
        conv_kernel_size=9,
        dropout_prob=None,
        pairwise=False,
        inter_hidden_size=128,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pairwise = pairwise

        self.bert = FM4BioModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.pairwise:
            self.inter_hidden_size = [inter_hidden_size, self.num_labels]
            self.classifier = FM4BioContactHead(config, self.inter_hidden_size)
        else:
            self.classifier = FM4BioClassificationHead(
                config,
                arch=arch,
                pooling=None,
                conv_kernel_size=conv_kernel_size,
                dropout_prob=dropout_prob,
                inter_hidden_size=inter_hidden_size,
            )

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        FM4BIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_str_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            inputs_str_embeds=inputs_str_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (bs, seq_len, hidden_size)

        # remove padding and [eos]
        mbs = input_ids.shape[0]
        seq_len = attention_mask.sum(1)  # (bs,)
        seq_len = seq_len - 1  # (bs,)
        assert mbs == 1, "currently only support mbs=1"
        sequence_output = sequence_output[:, : seq_len[0]]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
            else:
                loss_fct = CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FM4BioClassificationHead(nn.Module):
    """Head for classification tasks and regression tasks."""

    def __init__(
        self,
        config,
        arch="MLP",
        pooling="mean_pooling",
        conv_kernel_size=9,
        dropout_prob=None,
        inter_hidden_size=None,
        augment_with_zeroshot=False,
        activation_func="tanh",
    ):
        super().__init__()
        self.arch = arch
        self.pooling = pooling
        self.conv_kernal_size = conv_kernel_size
        self.augment_with_zeroshot = augment_with_zeroshot

        if dropout_prob is not None:
            self.dropout_prob = dropout_prob
        else:
            self.dropout_prob = config.hidden_dropout_prob

        if self.arch == "MLP" and inter_hidden_size is None:
            self.inter_hidden_size = config.hidden_size // 2
        else:
            self.inter_hidden_size = inter_hidden_size

        if activation_func == "tanh":
            self.activation_func = nn.Tanh()
        else:
            self.activation_func = nn.ReLU()

        if self.augment_with_zeroshot:
            input_hidden_size = config.hidden_size + 1
        else:
            input_hidden_size = config.hidden_size

        assert self.pooling in ["mean_pooling", None]
        if self.arch == "CNN":
            self.conv = nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=conv_kernel_size,
                padding="same",
            )
            self.dropout = nn.Dropout(self.dropout_prob)
            self.out_proj = nn.Linear(input_hidden_size, config.num_labels)
        elif self.arch == "MLP":
            self.ffn = nn.Linear(input_hidden_size, self.inter_hidden_size)
            self.dropout = nn.Dropout(self.dropout_prob)
            self.out_proj = nn.Linear(self.inter_hidden_size, config.num_labels)
        else:
            raise NotImplementedError

    def forward(
        self, hidden_states, attention_mask=None, zero_shot_fitness_predictions=None
    ):
        """
        Args:
            hidden_states: (bs, seq_len, hidden_size)
            attention_mask: (bs, seq_len), 0 means masking
        """
        x = hidden_states
        if self.arch == "CNN":
            # Refer to ProteinNPT
            x = self.dropout(x)
            x = x.permute(0, 2, 1)  # (bs, hidden_size, seq_len)
            x = self.conv(x)  # （bs, hidden_size, seq_len1)
            x = self.dropout(x)
            x = self.activation_func(x)
            x = x.permute(0, 2, 1)  # (bs, seq_len, hidden_size)
            # mean pooling
            if self.pooling == "mean_pooling":
                x = x.mean(dim=-2)  # (bs, hidden_size)
                if self.augment_with_zeroshot:
                    x = self._get_zero_shot_aug_feats(
                        x, zero_shot_fitness_predictions
                    )  # (bs, hidden_size+1)
            x = self.out_proj(x)

        elif self.arch == "MLP":
            if self.pooling == "mean_pooling":
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(x.size()).float()
                )
                x = torch.sum(x * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9
                )
                if self.augment_with_zeroshot:
                    x = self._get_zero_shot_aug_feats(
                        x, zero_shot_fitness_predictions
                    )  # (bs, hidden_size+1)
            x = self.dropout(x)
            x = self.ffn(x)
            x = self.activation_func(x)
            x = self.dropout(x)
            x = self.out_proj(x)
        return x

    def _get_zero_shot_aug_feats(self, x, zero_shot_fitness_predictions):
        """
        Add zero_shot_prediction to the beginning of x as the first feats
            x: torch.tensor, of shape (bs, hidden_size)
            zero_shot_fitness_predictions: torch.tensor, of shape (bs,) or (bs, 1)
        """
        assert zero_shot_fitness_predictions is not None
        if len(zero_shot_fitness_predictions.shape) == 1:
            zero_shot_fitness_predictions = zero_shot_fitness_predictions.unsqueeze(
                -1
            ).to(x.dtype)
        x = torch.cat((zero_shot_fitness_predictions, x), 1)  # (bs, hidden_size+1)
        return x


class FM4BioContactHead(nn.Module):
    """Head for contact prediction."""

    def __init__(self, config, inter_hidden_size=[128, 2]):
        super().__init__()
        self.ffn_0 = nn.Linear(config.hidden_size * 2, inter_hidden_size[0])
        self.ffn_1 = nn.Linear(inter_hidden_size[0], inter_hidden_size[1])

    def outer_concat(self, x):
        batch_size, seq_len, features = x.shape

        # Permute to [batch_size, features, seq_len]
        x = x.permute(0, 2, 1)

        # Introduce new dimensions for broadcasting
        x_1 = x[:, None, :, :, None]  # [batch_size, 1, features, seq_len, 1]
        x_2 = x[:, None, :, None, :]  # [batch_size, 1, features, 1, seq_len]

        # Repeat along new dimensions
        x_1 = x_1.repeat(
            1, 1, 1, 1, seq_len
        )  # [batch_size, 1, features, seq_len, seq_len]
        x_2 = x_2.repeat(
            1, 1, 1, seq_len, 1
        )  # [batch_size, 1, features, seq_len, seq_len]

        # Concatenate along the second dimension
        x = torch.cat((x_1, x_2), dim=1)  # [batch_size, 2, features, seq_len, seq_len]

        # Get lower triangular indices
        I, J = torch.tril_indices(seq_len, seq_len, -1)

        # Symmetrize
        x[:, :, :, I, J] = x[:, :, :, J, I]

        # Permute to desired shape and make contiguous
        x = x.permute(
            0, 3, 4, 2, 1
        ).contiguous()  # [batch_size, seq_len, seq_len, features, 2]

        # Reshape to combine the last two dimensions
        x = x.view(
            batch_size, seq_len, seq_len, features * 2
        )  # [batch_size, seq_len, seq_len, features * 2]

        return x

    def forward(self, hidden_states):
        # remove [sep] token at the end
        # x = hidden_states[:, :-1]     #(bs, seq_len, hidden_size)
        x = self.outer_concat(hidden_states)
        x = self.ffn_0(x)
        x = nn.ReLU()(x)
        x = self.ffn_1(x)
        return x
