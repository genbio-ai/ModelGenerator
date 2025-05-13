import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple, Callable, Union
from modelgenerator.huggingface_models.rnabert.modeling_rnabert import (
    apply_rotary_pos_emb,
)
from modelgenerator.huggingface_models.rnabert.configuration_rnabert import (
    RNABertConfig,
)
from modelgenerator.adapters.base import SequenceAdapter, TokenAdapter, FusionAdapter
from modelgenerator.adapters.adapters import LinearCLSAdapter, MLPAdapter


# Adapted from ModelGenerator/modelgenerator/huggingface_models/rnabert/modeling_rnabert.py/RNABertSelfAttention
class MultiHeadAttention(nn.Module):
    """
    Args:
        context_input_size (int): the input hidden size for the context (value and key)
    """

    def __init__(self, config, context_input_size: int, position_embedding_type=None):
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
            context_input_size, self.all_head_size, bias=config.add_linear_bias
        )
        self.value = nn.Linear(
            context_input_size, self.all_head_size, bias=config.add_linear_bias
        )
        # for residual connection
        self.dense = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.add_linear_bias
        )

        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
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
        """
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
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

            query_layer = apply_rotary_pos_emb(
                query_layer, q_pos_emb
            )  # debug query_layer[:,0]
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)

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

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RNABertModel forward() function)
            # attention_scores = attention_scores + attention_mask.to(
            #     attention_scores.dtype
            # )
            attention_mask = attention_mask > 0.5
            attention_scores = torch.where(attention_mask, attention_scores, -1e30)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # no_prob_mask = attention_mask < -1e-5
        # attention_probs = attention_probs.masked_fill(no_prob_mask, 0.0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return {
            "embeddings": self.dense(context_layer) + hidden_states,
            "query_heads": self.transpose_for_scores(mixed_query_layer),
            "value_heads": self.transpose_for_scores(self.value(encoder_hidden_states)),
            "key_heads": self.transpose_for_scores(self.key(encoder_hidden_states)),
            "attention_probs": attention_probs,
            "attention_scores": attention_scores,
            "context_layer": context_layer,
        }


class CrossAttentionFusion(nn.Module):
    """Fuse one or two other modalities into the current model

    Args:
        hidden_size (int): number of input features for major backbone
        context_input_size_1 (int): number of input features for first context backbone
        context_input_size_2 (int, optinal): number of input features for second context backbone
        num_attention_heads (int): Number of cross attention heads. Defaults to 16.
    """

    def __init__(
        self,
        hidden_size: int,
        context_input_size_1: int,
        context_input_size_2: int = None,
        num_attention_heads: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_input_size_1 = context_input_size_1
        self.context_input_size_2 = context_input_size_2
        self.cross_attention_layer_01 = MultiHeadAttention(
            config=RNABertConfig(
                num_attention_heads=num_attention_heads,
                attention_head_size=self.hidden_size // num_attention_heads,
                hidden_size=self.hidden_size,
                attention_probs_dropout_prob=0,
                max_position_embeddings=0,
            ),
            context_input_size=self.context_input_size_1,
        )
        if self.context_input_size_2 is not None:
            self.cross_attention_layer_02 = MultiHeadAttention(
                config=RNABertConfig(
                    num_attention_heads=num_attention_heads,
                    attention_head_size=self.hidden_size // num_attention_heads,
                    hidden_size=self.hidden_size,
                    attention_probs_dropout_prob=0,
                    max_position_embeddings=0,
                ),
                context_input_size=self.context_input_size_2,
            )
        self.output_hidden_size = hidden_size  # the fused embedding hidden size

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context_hidden_states_1: Tensor,
        context_attention_mask_1: Tensor,
        context_hidden_states_2: Tensor = None,
        context_attention_mask_2: Tensor = None,
    ) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, hidden_size)
            attention_mask (torch.Tensor): of shape (n, seq_len)
            context_hidden_states_1 (torch.Tensor): of shape (n, seq_len1, context_input_size_1)
            context_attention_mask_1 (torch.Tensor): of shape (n, seq_len1)
            context_hidden_states_2 (torch.Tensor): of shape (n, seq_len2, context_input_size_2)
            context_attention_mask_2 (torch.Tensor): of shape (n, seq_len2)

        Returns:
            torch.Tensor: fused embeddings of shape (n, seq_len, hidden_size)
        """
        if self.context_input_size_2 is not None and context_hidden_states_2 is None:
            raise ValueError(f"{context_hidden_states_2} is required.")

        encoder_attention_mask_01 = self.get_encoder_attention_mask(
            attention_mask, context_attention_mask_1
        )
        if context_hidden_states_2 is not None:
            encoder_attention_mask_02 = self.get_encoder_attention_mask(
                attention_mask, context_attention_mask_2
            )
        else:
            encoder_attention_mask_02 = None

        fused_embedding = self.cross_attention_layer_01.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=context_hidden_states_1,
            encoder_attention_mask=encoder_attention_mask_01,
        )["embeddings"]

        if self.context_input_size_2 is not None:
            fused_embedding = self.cross_attention_layer_02.forward(
                hidden_states=fused_embedding,
                encoder_hidden_states=context_hidden_states_2,
                encoder_attention_mask=encoder_attention_mask_02,
            )["embeddings"]

        return fused_embedding

    def get_encoder_attention_mask(self, query_attention_mask, key_attention_mask):
        """
        Args:
            query_attention_mask: attention mask for the query input, of shape (bs, seq_len1)
            key_attention_mask:  attention mask for the key input, of shape (bs, seq_len2)
        Return:
            encoder_attention_mask (tensor): of shape (bs, 1, seq_len1, seq_len2)
                - 1 for tokens that are not masked,
                - 0 for tokens that are masked.
        """
        # (bs, seq_len1, seq_len2), masking out the padding
        encoder_attention_mask = key_attention_mask.unsqueeze(
            1
        ) * query_attention_mask.unsqueeze(-1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(
            1
        )  # (bs, 1, seq_len1, seq_len2)
        return encoder_attention_mask


class ConcatFusion(nn.Module):
    """Concat output embeddings for 2-3 backbones
    
    Args:
        hidden_size (int): number of input features for first backbone
        hidden_size_1 (int): number of input features for second backbone
        hidden_size_2 (int, optinal): number of input features for third backbone
        project_size (int): the projected embedding size. Defaults to 512.
        pooling (str): Sequence pooling method. Defaults to mean_pooling
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_size_1: int,
        hidden_size_2: int = None,
        project_size: int = 512,
        pooling: str = "mean_pooling",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.project = nn.Linear(hidden_size, project_size)
        self.project_1 = nn.Linear(hidden_size_1, project_size)
        self.output_hidden_size = project_size * 2
        if self.hidden_size_2 is not None:
            self.project_2 = nn.Linear(hidden_size_2, project_size)
            self.output_hidden_size = project_size * 3
        self.pooling = pooling
        if self.pooling not in ["mean_pooling", "cls_pooling"]:
            raise NotImplementedError

    def forward(self, 
                hidden_states: Tensor, 
                attention_mask: Tensor,
                hidden_states_1: Tensor, 
                attention_mask_1: Tensor,
                hidden_states_2: Tensor = None,
                attention_mask_2: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, hidden_size)
            attention_mask (torch.Tensor): of shape (n, seq_len)
            hidden_states_1 (torch.Tensor): of shape (n, seq_len1, hidden_size_1)
            attention_mask_1 (torch.Tensor): of shape (n, seq_len1)
            hidden_states_2 (torch.Tensor): of shape (n, seq_len2, hidden_size_2)
            attention_mask_2 (torch.Tensor): of shape (n, seq_len2)

        Returns:
            torch.Tensor: fused embeddings of shape (n, seq_len, hidden_size)
        """
        if self.hidden_size_2 is not None and hidden_states_2 is None:
             raise ValueError(f"{hidden_states_2} is required.")
        # seq pooling    
        pooled_embeddings = self.seq_pooling(hidden_states, attention_mask)
        pooled_embeddings_1 = self.seq_pooling(hidden_states_1, attention_mask_1)
        # project to the same dimension
        pooled_embeddings = self.project(pooled_embeddings)
        pooled_embeddings_1 = self.project_1(pooled_embeddings_1)
        if hidden_states_2 is None:
            fused_embedding = torch.cat((pooled_embeddings, pooled_embeddings_1), dim=1)
        else:
            pooled_embeddings_2 = self.seq_pooling(hidden_states_2, attention_mask_2)
            pooled_embeddings_2 = self.project_2(pooled_embeddings_2)
            fused_embedding = torch.cat((pooled_embeddings, pooled_embeddings_1, pooled_embeddings_2), dim=1)
        return fused_embedding
    
    def seq_pooling(self, hidden_states, attention_mask):
        if self.pooling == "mean_pooling":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            embeddings = torch.sum(
                hidden_states * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling == "cls_pooling":
            embeddings = hidden_states[:, 0]
        else:
            raise NotImplementedError
        return embeddings



class MMFusionSeqAdapter(nn.Module, FusionAdapter):
    """Multimodal embeddings fusion with SequenceAdapter.

    Note: 
        Accepts 2-3 sequence embeddings as input and fuses them into a multimodal embedding for the adapter.

    Args:
        out_features: Number of output features.
        input_size: number of input features for the first modality.
        input_size_1: number of input features for the second modality.
        input_size_2: number of input features for the third modality.
        fusion: The callable that returns a fusion module.
        adapter: The callable that returns an adapter.
    """

    def __init__(
        self,
        out_features: int,
        input_size: int,
        input_size_1: int,
        input_size_2: int = None,
        fusion: Callable[[int, int, int], CrossAttentionFusion] = CrossAttentionFusion,
        adapter: Callable[[int, int], SequenceAdapter] = LinearCLSAdapter,
    ):
        super().__init__()
        self.fusion_module = fusion(input_size, input_size_1, input_size_2)
        self.adapter = adapter(self.fusion_module.output_hidden_size, out_features)

    def forward(self, 
                hidden_states: Tensor,  
                attention_mask: Tensor,
                hidden_states_1: Tensor, 
                attention_mask_1: Tensor,
                hidden_states_2: Tensor = None, 
                attention_mask_2: Tensor = None
                ) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, input_size)
            attention_mask (torch.Tensor): of shape (n, seq_len)
            hidden_states_1 (torch.Tensor): of shape (n, seq_len1, input_size_1)
            attention_mask_1 (torch.Tensor): of shape (n, seq_len1)
            hidden_states_2 (torch.Tensor, optional): of shape (n, seq_len2, input_size_2)
            attention_mask_2 (torch.Tensor, optional): of shape (n, seq_len2)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        fused_embeddings = self.fusion_module(hidden_states, attention_mask,
                                              hidden_states_1, attention_mask_1,
                                              hidden_states_2, attention_mask_2)
        preds = self.adapter(fused_embeddings, attention_mask)
        return preds

class MMFusionTokenAdapter(nn.Module, FusionAdapter):
    """Multimodal embeddings fusion with TokenAdapter. Fuses embeddings into a single token embedding.

    Note: 
        Accepts 2-3 sequence embeddings as input and fuse them into a multimodal embedding for the adapter

    Args:
        out_features: Number of output features.
        input_size: number of input features for the first modality.
        input_size_1: number of input features for the second modality.
        input_size_2: number of input features for the third modality.
        fusion: The callable that returns a fusion module.
        adapter: The callable that returns an adapter.
    """

    def __init__(
        self,
        out_features: int,
        input_size: int,
        input_size_1: int,
        input_size_2: int = None,
        fusion: Callable[[int, int, int], ConcatFusion] = ConcatFusion,
        adapter: Callable[[int, int], TokenAdapter] = MLPAdapter,
    ):
        super().__init__()
        self.fusion_module = fusion(input_size, input_size_1, input_size_2)
        self.adapter = adapter(self.fusion_module.output_hidden_size, out_features)

    def forward(self, 
                hidden_states: Tensor,  
                attention_mask: Tensor,
                hidden_states_1: Tensor, 
                attention_mask_1: Tensor,
                hidden_states_2: Tensor = None, 
                attention_mask_2: Tensor = None
                ) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, input_size)
            attention_mask (torch.Tensor): of shape (n, seq_len)
            hidden_states_1 (torch.Tensor): of shape (n, seq_len1, input_size_1)
            attention_mask_1 (torch.Tensor): of shape (n, seq_len1)
            hidden_states_2 (torch.Tensor, optional): of shape (n, seq_len2, input_size_2)
            attention_mask_2 (torch.Tensor, optional): of shape (n, seq_len2)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        fused_embeddings = self.fusion_module(hidden_states, attention_mask,
                                              hidden_states_1, attention_mask_1,
                                              hidden_states_2, attention_mask_2)
        preds = self.adapter(fused_embeddings)
        return preds
    