import os
from typing import Union, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info

from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

from transformers.utils import cached_file

from modelgenerator.backbones.base import *

from modelgenerator.data import gather_data

import numpy as np


class GenBioBERT(HFSequenceBackbone):
    """GenBioBERT model

    Note:
        Models using this interface include `aido_dna_7b`, `aido_dna_300m`, `dna_dummy`, `aido_dna_debug`,
        `aido_rna_1b600m`, `aido_rna_1b600m_cds`, `aido_rna_1m_mars`, `aido_rna_25m_mars`, `aido_rna_300m_mars`,
        `aido_rna_650m`, `aido_rna_650m_cds`.

        FSDP auto_wrap_policy is `modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy`

    Args:
        from_scratch: Whether to create the model from scratch.
        max_length: Maximum sequence length.
        use_peft: Whether to use LoRA PEFT.
        frozen: Whether to freeze encoder.
        save_peft_only: Whether to save only the PEFT weights.
        lora_r: LoRA r parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target modules.
    """

    fsdp_wrap_modules = ["modelgenerator.huggingface_models.rnabert.modeling_rnabert.RNABertLayer"]

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = ["query", "value"],
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

    def setup(self):
        # Delays hf model imports to avoid model name conflicts
        from modelgenerator.huggingface_models.rnabert import (
            RNABertConfig,
            RNABertTokenizer,
            RNABertModel,
            RNABertForMaskedLM,
            RNABertForTokenClassification,
            RNABertForSequenceClassification,
        )

        repo_base_dir = Path(__file__).resolve().parent.parent.parent
        vocab_file = os.path.join(
            repo_base_dir, "modelgenerator/huggingface_models/rnabert/vocab.txt"
        )
        self.tokenizer = RNABertTokenizer(vocab_file, version="v2")  # add [CLS] ... [SEP]
        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = RNABertForSequenceClassification
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = RNABertForTokenClassification
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = RNABertForMaskedLM
        else:
            model_class = RNABertModel

        if self.from_scratch:
            config = RNABertConfig()
        else:
            config = RNABertConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if self.from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )

        if not self.use_legacy_adapter:
            self.encoder = model
        else:
            self.encoder = model.bert
        self.decoder = None
        if model_class == RNABertForMaskedLM:
            self.decoder = model.cls
        elif model_class == RNABertForTokenClassification:
            self.decoder = model.classifier
        elif model_class == RNABertForSequenceClassification:
            self.decoder = model.classifier

        if self.use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                modules_to_save=[],
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if self.frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def process_batch(self, batch, device, add_special_tokens=True, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (List[str]): List of input sequences.
            device (torch.device): Device to move the data to.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        seq_tokenized = self.tokenize(
            batch["sequences"], padding=True, add_special_tokens=add_special_tokens, **kwargs
        )
        for k, v in seq_tokenized.items():
            if v is not None:
                if torch.is_tensor(v):
                    seq_tokenized[k] = v.to(dtype=torch.long, device=device)
                else:
                    seq_tokenized[k] = torch.tensor(v, dtype=torch.long, device=device)
        return seq_tokenized

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        all_hidden_states: bool = False,
        special_tokens_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return SequenceBackboneOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if all_hidden_states else None,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids, attention_mask, special_tokens_mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            padding=padding,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=self.max_length is not None,
            return_special_tokens_mask=True,
        )
        output_keys = ["input_ids", "attention_mask", "special_tokens_mask"]
        return {k: seq_tokenized[k] for k in output_keys}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the pre-trained model

        Returns:
            int: Vocabulary size
        """
        return self.encoder.config.vocab_size

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(self.encoder, adapter_name=adapter_name)
            prefixed_dict = {f"{prefix}.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith(f"{prefix}.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class GenBioFM(HFSequenceBackbone):
    """GenBioFM model

    Note:
        Models using this interface include `aido_protein_16b`, `aido_protein_16b_v1`, `aido_protein2structoken_16b`, `aido_protein_debug`.

        FSDP auto_wrap_policy is `modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy`

    Args:
        from_scratch: Whether to create the model from scratch.
        max_length: Maximum sequence length.
        use_peft: Whether to use LoRA PEFT.
        frozen: Whether to freeze encoder.
        save_peft_only: Whether to save only the PEFT weights.
        lora_r: LoRA r parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target modules.
        lora_modules_to_save: LoRA modules to save.
        lora_use_rslora: Whether to use RSLora.
        enable_rag (bool, optional): Whether to enable RAG which requires `msa` and `str_emb` in data batches.
    """

    fsdp_wrap_modules = [
        "modelgenerator.huggingface_models.fm4bio.modeling_fm4bio.FM4BioLayer",
        "modelgenerator.huggingface_models.fm4bio.modeling_fm4bio.FM4BioMLP",
        "modelgenerator.huggingface_models.fm4bio.modeling_fm4bio.FM4BioEmbeddings",
    ]

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = [
            "query",
            "value",
            "key",
            "dense",
            "router",
        ],
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        enable_rag: bool = False,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_modules_to_save = lora_modules_to_save
        self.lora_use_rslora = lora_use_rslora
        self.enable_rag = enable_rag

    def setup(self):
        from modelgenerator.huggingface_models.fm4bio import (
            FM4BioConfig,
            FM4BioTokenizer,
            FM4BioModel,
            FM4BioForMaskedLM,
            FM4BioForTokenClassification,
            FM4BioForSequenceClassification,
        )

        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = FM4BioForSequenceClassification
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = FM4BioForTokenClassification
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = FM4BioForMaskedLM
        else:
            model_class = FM4BioModel

        repo_base_dir = Path(__file__).resolve().parent.parent.parent
        vocab_file = os.path.join(
            repo_base_dir, "modelgenerator/huggingface_models/fm4bio/vocab_protein.txt"
        )
        self.tokenizer = FM4BioTokenizer(vocab_file, version="v1")

        if self.from_scratch:
            config = FM4BioConfig()
        else:
            config = FM4BioConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if self.from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )

        if not self.use_legacy_adapter:
            self.encoder = model
        else:
            self.encoder = model.bert
        self.decoder = None
        if model_class == FM4BioForMaskedLM:
            try:
                self.decoder = model.output_embed
            except AttributeError:
                self.decoder = model.cls
        elif model_class == FM4BioForTokenClassification:
            self.decoder = model.classifier
        elif model_class == FM4BioForSequenceClassification:
            self.decoder = model.classifier

        if self.use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=self.lora_target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                use_rslora=self.lora_use_rslora,
                inference_mode=False,
                modules_to_save=self.lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:  # use linear probing, freeze all parameters
            if self.frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def required_data_columns(self) -> list[str]:
        """Required data columns for the model

        Returns:
            list[str]: List of required data columns
        """
        columns = super().required_data_columns()
        if self.enable_rag:
            columns += ["msa", "str_emb"]
        return columns

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        all_hidden_states: bool = False,
        position_ids: Optional[Tensor] = None,
        special_tokens_mask: Optional[Tensor] = None,
        query_tokens_mask: Optional[Tensor] = None,
        input_str_embeds: Optional[Tensor] = None,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.
            position_ids (torch.Tensor, optional): Position IDs (n, seq_len). Defaults to None.
            query_tokens_mask (torch.Tensor, optional): Query tokens mask (n, seq_len). Defaults to None.
            input_str_embeds (torch.Tensor, optional): Struct embeddings (n, seq_len, embed_dim). Defaults to None.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            position_ids=position_ids,
            inputs_str_embeds=input_str_embeds,
        )

        if query_tokens_mask is not None:
            # [B, L] -> [L]
            q_mask = (query_tokens_mask.sum(0) > 0).to(input_ids.device)
            # list of [B, L, D] -> [B, l, D]
            outputs.hidden_states = tuple([state[:, q_mask] for state in outputs.hidden_states])
            # [B, L, D] -> [B, l, D]
            outputs.last_hidden_state = outputs.last_hidden_state[:, q_mask]
            attention_mask = attention_mask[:, q_mask]
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask[:, q_mask]

        return SequenceBackboneOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if all_hidden_states else None,
            special_tokens_mask=special_tokens_mask,
            attention_mask=attention_mask,
        )

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, add_special_tokens=True, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.
            device (torch.device): Device to move the data to.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        sequences = batch["sequences"]
        msa = batch.get("msa")
        str_embs = batch.get("str_emb")
        if msa is None or str_embs is None:
            seq_tokenized = self.tokenize(
                sequences,
                padding=True,
                add_special_tokens=add_special_tokens,
            )
            for k, v in seq_tokenized.items():
                if v is not None:
                    if torch.is_tensor(v):
                        seq_tokenized[k] = v.to(dtype=torch.long, device=device)
                    else:
                        seq_tokenized[k] = torch.tensor(v, dtype=torch.long, device=device)
            return seq_tokenized
        if torch.is_tensor(str_embs):
            str_embs = str_embs.cpu().numpy()
        elif isinstance(str_embs, list):
            str_embs = np.array(str_embs)

        assert self.encoder.config.position_embedding_type == "rope_2d"
        # sequences: str
        # msa: List[str]
        # str_emb: np.ndarray
        new_sequences = []
        position_ids = []
        str_embs_new = []
        query_tokens_mask = []

        for i, seq in enumerate(sequences):
            msa = msa[i]
            str_emb = str_embs[i]

            len_seq = len(seq)
            new_seq = list(seq)
            num_seq = 1
            for msa_seq in msa:
                assert len(msa_seq) == len_seq, f"len(msa_seq)={len(msa_seq)}, len_seq={len_seq}"
                new_seq += list(msa_seq)
                num_seq += 1
            new_seq = np.array(new_seq)
            gap_mask = new_seq != "-"
            new_seq = "".join(new_seq[gap_mask])
            new_seq = new_seq[: self.max_length]
            new_sequences.append(new_seq)

            # 2D RoPE encoding
            pos_encoding = np.stack(
                [
                    np.tile(np.arange(len_seq), num_seq),
                    np.repeat(np.arange(num_seq), len_seq),
                ]
            )
            pos_encoding = pos_encoding[:, gap_mask]
            pos_encoding = pos_encoding[:, : self.max_length]
            position_ids.append(pos_encoding)

            assert str_emb.shape[0] == len_seq, f"str_emb.shape={str_emb.shape}, len_seq={len_seq}"
            str_embs_new.append(str_emb)

            q_mask = np.zeros(len(new_seq))
            q_mask[: min(len_seq, self.max_length)] = 1
            query_tokens_mask.append(q_mask)

        seq_tokenized = self.tokenize(
            new_sequences,
            padding=True,
            add_special_tokens=add_special_tokens,
        )
        input_ids = seq_tokenized["input_ids"]

        # 1. Make attention_mask and special_mask same length with query sequence

        # Final padding

        final_L = len(input_ids[0])
        position_ids = [
            (
                pos_enc[:, :final_L].tolist()
                if final_L < pos_enc.shape[1]
                else np.pad(pos_enc, [(0, 0), (0, final_L - pos_enc.shape[1])]).tolist()
            )
            for pos_enc in position_ids
        ]
        query_tokens_mask = [
            (
                q_mask[:final_L].tolist()
                if final_L < q_mask.shape[0]
                else np.pad(q_mask, [(0, final_L - q_mask.shape[0])]).tolist()
            )
            for q_mask in query_tokens_mask
        ]
        str_embs_new = [
            (
                str_emb[:final_L].tolist()
                if final_L < str_emb.shape[0]
                else np.pad(str_emb, [(0, final_L - str_emb.shape[0]), (0, 0)]).tolist()
            )
            for str_emb in str_embs_new
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                seq_tokenized["attention_mask"], dtype=torch.long, device=device
            ),
            "special_tokens_mask": torch.tensor(
                seq_tokenized["special_tokens_mask"], dtype=torch.long, device=device
            ),
            "query_tokens_mask": torch.tensor(query_tokens_mask, dtype=torch.long, device=device),
            "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
            "input_str_embeds": torch.tensor(str_embs_new, dtype=torch.long, device=device),
        }

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences
            padding (bool, optional): Whether to pad sequences. Defaults to True.

        Returns:
            dict: contains input_ids, attention_mask, special_tokens_mask

        """

        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=padding,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=True,
        )
        output_keys = ["input_ids", "attention_mask", "special_tokens_mask"]
        return {k: seq_tokenized[k] for k in output_keys}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the pre-trained model

        Returns:
            int: Vocabulary size
        """
        return self.encoder.config.vocab_size

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(self.encoder, adapter_name=adapter_name)
            prefixed_dict = {f"{prefix}.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith(f"{prefix}.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class GenBioCellFoundation(HFSequenceBackbone):
    """GenBioCellFoundation model

    Note:
        Models using this interface include `aido_cell_100m`, `aido_cell_10m`, and `aido_cell_3m`.

        FSDP auto_wrap_policy is `modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy`

    Args:
        from_scratch: Whether to create the model from scratch.
        max_length: Maximum sequence length.
        use_peft: Whether to use LoRA PEFT.
        frozen: Whether to freeze encoder.
        save_peft_only: Whether to save only the PEFT weights.
        lora_r: LoRA r parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target modules.
        lora_modules_to_save: LoRA modules to save.
        lora_use_rslora: Whether to use RSLora.
    """

    fsdp_wrap_modules = [
        "modelgenerator.huggingface_models.cellfoundation.modeling_cellfoundation.CellFoundationLayer"
    ]

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],  # Should not need this.
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = [
            "query",
            "value",
            "key",
            "dense",
            "router",
        ],
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        # Note: Legacy adapters are for older sequence models.
        if legacy_adapter_type is not None:
            # raise NotImplementedError(
            #     "Legacy adapters are not implemented for CellFoundation."
            # )
            legacy_adapter_type = None
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_modules_to_save = lora_modules_to_save
        self.lora_use_rslora = lora_use_rslora

    def setup(self):
        from modelgenerator.huggingface_models.cellfoundation import (
            CellFoundationConfig,
            CellFoundationModel,
        )

        model_class = CellFoundationModel
        peft_task_type = TaskType.FEATURE_EXTRACTION

        if self.from_scratch:
            config = CellFoundationConfig()
        else:
            config = CellFoundationConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if self.from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        if self.training:
            model = model.train()
        self.encoder = model
        self.decoder = None

        if self.use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=self.lora_target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                use_rslora=self.lora_use_rslora,
                inference_mode=False,
                modules_to_save=self.lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if self.frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        all_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """
        # Converting from torch.long; should be counts.
        X = input_ids.to(dtype=torch.float32)

        # https://github.com/fm4bio/scFoundation-repro/blob/9f706d807b68ec7b7f2df735d2a96fb4b1b67a0c/annotation/cell_annotation.py#L126-L141:
        rawcountsidx = torch.maximum(
            torch.log10(X.sum(dim=1, keepdim=True)), torch.tensor(5, device=X.device)
        )
        inputcountidx = torch.maximum(
            torch.log10(X.sum(dim=1, keepdim=True)), torch.tensor(5, device=X.device)
        )
        X = torch.log1p(X / X.sum(dim=1, keepdim=True) * 10000)
        X = torch.cat(
            (
                X,
                rawcountsidx.to(X.device),
                inputcountidx.to(X.device),
            ),
            axis=1,
        )
        X[X > 20] = 20
        X = X.to(torch.bfloat16)

        outputs = self.encoder(
            input_ids=X,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Note: Trimming off embeddings corresponding to read depth inputs.
        hidden_states = (x[:, :-2, :] for x in outputs.hidden_states)
        return SequenceBackboneOutput(
            last_hidden_state=outputs.last_hidden_state[:, :-2, :],
            hidden_states=hidden_states if all_hidden_states else None,
            special_tokens_mask=None,
            attention_mask=attention_mask,
        )

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.
            device (torch.device): Device to move the data to.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        input_ids = batch["sequences"]
        if torch.is_tensor(input_ids):
            input_ids = input_ids.to(dtype=torch.float32, device=device)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.float32, device=device)
        return {"input_ids": input_ids}

    def tokenize(
        self,
        sequences: list[str],
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Note:
            This is a dummy tokenizer since the CellFoundation models consume gene expression.

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids
        """
        return {"input_ids": sequences}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(self.encoder, adapter_name=adapter_name)
            prefixed_dict = {f"{prefix}.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith(f"{prefix}.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class GenBioCellSpatialFoundation(HFSequenceBackbone):
    """GenBioCellSpatialFoundation model

    Note:
        Models using this interface include `aido_tissue_60m` and `aido_tissue_3m`.

        FSDP auto_wrap_policy is `modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy`

    Args:
        from_scratch: Whether to create the model from scratch.
        max_length: Maximum sequence length.
        use_peft: Whether to use LoRA PEFT.
        frozen: Whether to freeze encoder.
        save_peft_only: Whether to save only the PEFT weights.
        lora_r: LoRA r parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target modules.
        lora_modules_to_save: LoRA modules to save.
        lora_use_rslora: Whether to use RSLora.
        rope2d_use_xy: Whether to use 2D rope encoding.
        sep_value: Separator value for the model.
    """

    fsdp_wrap_modules = [
        "modelgenerator.huggingface_models.cellspatialfoundation.modeling_cellspatialfoundation.CellSpatialFoundationLayer"
    ]

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],  # Should not need this.
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = [
            "query",
            "value",
            "key",
            "dense",
            "router",
        ],
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        rope2d_use_xy: bool = False,
        sep_value: int = -10000,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        # Note: Legacy adapters are for older sequence models.
        if legacy_adapter_type is not None:
            raise NotImplementedError("Legacy adapters are not implemented for CellFoundation.")
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_modules_to_save = lora_modules_to_save
        self.lora_use_rslora = lora_use_rslora
        self.rope2d_use_xy = rope2d_use_xy
        self.sep_value = sep_value

    def setup(self):
        from modelgenerator.huggingface_models.cellspatialfoundation import (
            CellSpatialFoundationConfig,
            CellSpatialFoundationModel,
        )

        model_class = CellSpatialFoundationModel
        peft_task_type = TaskType.FEATURE_EXTRACTION

        if self.from_scratch:
            config = CellSpatialFoundationConfig()
        else:
            config = CellSpatialFoundationConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if self.from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        self.encoder = model
        self.decoder = None

        self.use_peft = self.use_peft
        self.frozen = self.frozen
        self.save_peft_only = self.save_peft_only
        if self.use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=self.lora_target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                use_rslora=self.lora_use_rslora,
                inference_mode=False,
                modules_to_save=self.lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if self.frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

        self.config = config

    def forward(
        self,
        input_ids: Tensor,
        all_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """
        (
            X,
            encoder_data,
            encoder_data_labels,
            encoder_attention_mask,
            encoder_rope_id,
            sep_idx_1,
            sep_idx_2,
            cell_num,
        ) = self._process_input(input_ids)
        outputs = self.encoder(
            input_ids=encoder_data.to(torch.bfloat16),
            attention_mask=encoder_attention_mask,
            output_hidden_states=True,
            position_ids=encoder_rope_id.long(),  # 2D pos id
        )

        if all_hidden_states:
            hidden_states = (x[:, :-2, :] for x in outputs.hidden_states)

        if cell_num > 1:
            # multiple cells, pool over 1st cell then return embedding of 1st cell
            # need get dim from outputs.last_hidden_state when encoder_cell_id==0 and others fill with pad dim, and set pad position=0 in encoder_attention_mask
            if self.rope2d_use_xy:
                cell_id_full = (
                    torch.tensor(range(sep_idx_2 - sep_idx_1 - 1), device=encoder_data.device)
                    .view(1, -1)
                    .repeat(X.shape[0], 1)
                    .repeat_interleave(self.config.max_position_embeddings + 2, dim=1)
                )
                cell_id_encoder, _ = gather_data(
                    cell_id_full, encoder_data_labels, self.config.pad_token_id
                )
                first_cell_max_len = (cell_id_encoder == 0).sum(dim=1).max()
                first_cell_mask = ~(
                    cell_id_encoder[:, :first_cell_max_len] == 0
                )  # true=not 1st cell
            else:
                first_cell_max_len = (encoder_rope_id[:, 1, :] == 0).sum(dim=1).max()
                first_cell_mask = ~(
                    encoder_rope_id[:, 1, :first_cell_max_len] == 0
                )  # true=not 1st cell

            first_cell_emb = torch.zeros(
                (
                    outputs.last_hidden_state.shape[0],
                    first_cell_max_len,
                    outputs.last_hidden_state.shape[2],
                ),
                device=encoder_data.device,
            )
            first_cell_mask_full = torch.unsqueeze(first_cell_mask, -1).repeat(
                1, 1, outputs.last_hidden_state.size(2)
            )

            first_cell_max_len_emb = outputs.last_hidden_state[:, 0:first_cell_max_len, :]

            first_cell_emb = torch.where(
                first_cell_mask_full == False, first_cell_max_len_emb, first_cell_emb
            )

            return SequenceBackboneOutput(
                last_hidden_state=first_cell_emb,
                hidden_states=hidden_states if all_hidden_states else None,
                attention_mask=(~(first_cell_mask)).long(),
            )

        else:
            # single-cell, pool over all genes
            return SequenceBackboneOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=hidden_states if all_hidden_states else None,
            )

    def _process_input(self, input_ids):
        sep_idx_1, sep_idx_2 = None, None
        if self.rope2d_use_xy:
            sep_value_idx = (input_ids == self.sep_value).nonzero(as_tuple=True)
            sep_idx_1 = sep_value_idx[1][0]
            sep_idx_2 = sep_value_idx[1][1]

            coordinate_x = (
                input_ids[:, sep_idx_1 + 1 : sep_idx_2]
                .clone()
                .repeat_interleave(self.config.max_position_embeddings + 2, dim=1)
            )
            coordinate_y = (
                input_ids[:, sep_idx_2 + 1 :]
                .clone()
                .repeat_interleave(self.config.max_position_embeddings + 2, dim=1)
            )
            input_ids = input_ids[:, :sep_idx_1].clone()

        assert (
            input_ids.shape[1] >= self.config.max_position_embeddings
            and input_ids.shape[1] % self.config.max_position_embeddings == 0
        ), (
            "input_ids.shape[1] has to be a multiple of max_position_embeddings, "
            f"got {input_ids.shape[1]} and {self.config.max_position_embeddings}"
        )
        if input_ids.shape[1] == self.config.max_position_embeddings:
            X = torch.as_tensor(
                input_ids, dtype=torch.bfloat16
            )  # Converting from torch.long; should be counts.

            # https://github.com/fm4bio/scFoundation-repro/blob/9f706d807b68ec7b7f2df735d2a96fb4b1b67a0c/annotation/cell_annotation.py#L126-L141:
            rawcountsidx = max(torch.log10(X.sum()), 5)
            inputcountidx = max(torch.log10(X.sum()), 5)
            X = torch.log1p(X / X.sum() * 10000).to(torch.float)
            X = torch.cat(
                (
                    X,
                    torch.tensor([rawcountsidx, inputcountidx]).repeat(X.shape[0], 1).to(X.device),
                ),
                axis=1,
            ).float()
            X[X > 20] = 20
            cell_num = 1
        elif input_ids.shape[1] / self.config.max_position_embeddings > 1:
            X_ls = []
            cell_num = int(input_ids.shape[1] / self.config.max_position_embeddings)
            for cell_idx in range(cell_num):
                X = torch.as_tensor(
                    input_ids[
                        :,
                        (self.config.max_position_embeddings * cell_idx) : (
                            self.config.max_position_embeddings * cell_idx
                            + self.config.max_position_embeddings
                        ),
                    ],
                    dtype=torch.bfloat16,
                )  # Converting from torch.long; should be counts.

                # https://github.com/fm4bio/scFoundation-repro/blob/9f706d807b68ec7b7f2df735d2a96fb4b1b67a0c/annotation/cell_annotation.py#L126-L141:
                rawcountsidx = max(torch.log10(X.sum()), 5)
                inputcountidx = max(torch.log10(X.sum()), 5)
                X = torch.log1p(X / X.sum() * 10000).to(torch.float)
                X = torch.cat(
                    (
                        X,
                        torch.tensor([rawcountsidx, inputcountidx])
                        .repeat(X.shape[0], 1)
                        .to(X.device),
                    ),
                    axis=1,
                ).float()
                X[X > 20] = 20
                X_ls.append(X)
            X = torch.cat(X_ls, dim=1)

        encoder_data_labels = X > 0
        encoder_data, encoder_data_padding = gather_data(
            X, encoder_data_labels, self.config.pad_token_id
        )
        encoder_attention_mask = (~encoder_data_padding).long()  # 1 for not mask, 0 for mask

        if cell_num > 1:
            if self.rope2d_use_xy:
                data_gene_ids = coordinate_x  # x as 1D
            else:
                data_gene_ids = torch.arange(
                    self.config.max_position_embeddings + 2, device=X.device
                ).repeat(X.shape[0], cell_num)
        else:
            if self.rope2d_use_xy:
                data_gene_ids = coordinate_x
            else:
                data_gene_ids = torch.arange(X.shape[1], device=X.device).repeat(X.shape[0], 1)
        encoder_position_gene_ids, _ = gather_data(
            data_gene_ids, encoder_data_labels, self.config.pad_token_id
        )
        if self.rope2d_use_xy:
            encoder_position_gene_ids[encoder_position_gene_ids == self.config.pad_token_id] = (
                0  # set x of pad token as 0, due to cal range
            )
        else:
            encoder_position_gene_ids[encoder_position_gene_ids == self.config.pad_token_id] = (
                self.config.max_position_embeddings + 2
            )

        encoder_rope_id = torch.zeros(
            (
                encoder_data.shape[0],
                2,
                encoder_data.shape[1],
            ),
            device=encoder_data.device,
        )
        encoder_rope_id[:, 0, :] = encoder_position_gene_ids
        if cell_num > 1:
            if self.rope2d_use_xy:
                encoder_cell_id, _ = gather_data(
                    coordinate_y, encoder_data_labels, 0
                )  # set y of pad token as 0, due to cal range
            else:
                cell_id = (
                    torch.arange(cell_num, device=encoder_data.device)
                    .view(-1, 1)
                    .repeat(1, self.config.max_position_embeddings + 2)
                    .view(1, -1)
                    .repeat(encoder_position_gene_ids.shape[0], 1)
                )
                encoder_cell_id, _ = gather_data(
                    cell_id,
                    encoder_data_labels,
                    self.config.max_position_embeddings + 2,
                )  # for pos id, pad_token_id = 19266
            encoder_rope_id[:, 1, :] = encoder_cell_id
        else:
            encoder_rope_id[:, 1, :] = 0  # assume single input cell

        return (
            X,
            encoder_data,
            encoder_data_labels,
            encoder_attention_mask,
            encoder_rope_id,
            sep_idx_1,
            sep_idx_2,
            cell_num,
        )

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.
            device (torch.device): Device to move the data to.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        # TODO: Move data processing steps in forward to this function
        input_ids = batch["sequences"]
        if torch.is_tensor(input_ids):
            input_ids = input_ids.to(device=device)
        else:
            input_ids = torch.tensor(input_ids, device=device)
        return {"input_ids": input_ids}

    def tokenize(
        self,
        sequences: list[str],
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids, attention_mask
        """

        processed_input = self._process_input(sequences)
        return {"input_ids": processed_input[1], "attention_mask": processed_input[3]}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(self.encoder, adapter_name=adapter_name)
            prefixed_dict = {f"backbone.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith(f"{prefix}.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class Onehot(HFSequenceBackbone):
    """Tokenizer-only model for one-hot encoding. Useful for baseline model testing (CNNs, linear, etc.)

    Note:
        Models using this interface include `dna_onehot` and `protein_onehot`.

        Does not contain any parameters, and cannot be used without an adapter.

    Attributes:
        vocab_file (str): Path to the vocabulary file.

    Args:
        vocab_file (str, optional): Path to the vocabulary file. Defaults to
            "modelgenerator/huggingface_models/rnabert/vocab.txt".
        max_length (Optional[int], optional): Maximum sequence length.
    """

    fsdp_wrap_modules = ["modelgenerator.huggingface_models.rnabert.modeling_rnabert.RNABertLayer"]

    vocab_file: str = os.path.join(
        Path(__file__).resolve().parent.parent.parent,
        "modelgenerator/huggingface_models/rnabert/vocab.txt",
    )

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        vocab_file: Optional[str] = None,
        max_length: Optional[int] = 512,
    ):
        from modelgenerator.huggingface_models.fm4bio import FM4BioTokenizer

        super().__init__(legacy_adapter_type, default_config)
        self.max_length = max_length
        if vocab_file is not None:
            self.vocab_file = vocab_file
        self.tokenizer = FM4BioTokenizer(self.vocab_file, version="v1")

    def setup(self):
        pass

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        """
        Returns one-hot encoding of input_ids.

        Args:
            input_ids (Tensor): Token IDs

        Returns:
            Tensor: One-hot encoding of input_ids
        """
        one_hot = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.tokenizer.vocab_size,
        ).to(input_ids.device)
        one_hot.scatter_(2, input_ids.unsqueeze(2), 1)
        return SequenceBackboneOutput(
            last_hidden_state=one_hot,
        )

    def get_decoder(self) -> nn.Module:
        """Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return _Identity()

    def process_batch(
        self, batch: dict, device: torch.device, add_special_tokens: bool = True, **kwargs
    ):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.
            device (torch.device): Device to move the data to.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        seq_tokenized = self.tokenize(
            batch["sequences"], padding=True, add_special_tokens=add_special_tokens, **kwargs
        )
        for k, v in seq_tokenized.items():
            if v is not None:
                if torch.is_tensor(v):
                    seq_tokenized[k] = v.to(dtype=torch.long, device=device)
                else:
                    seq_tokenized[k] = torch.tensor(v, dtype=torch.long, device=device)
        return seq_tokenized

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids, attention_mask, special_tokens_mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=True,
            padding=padding,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=True,
        )
        output_keys = ["input_ids", "attention_mask", "special_tokens_mask"]
        return {k: seq_tokenized[k] for k in output_keys}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the arbitrary max context specified by the user

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        return

    def get_num_layer(self) -> int:
        return


class _Identity(nn.Identity):
    def forward(self, input: torch.Tensor, *args, **kwargs):
        return input


class Huggingface(HFSequenceBackbone):
    """A generic huggingface wrapper allows for using any huggingface model as backbone.

    Note:
        **Warning:** This is an experimental feature, don't expect it to work with all models.
        Downstream task support is also extremely limited to the standard huggingface heads.
        Its usage often involves manual configuration of the model's head through `config_overwrites`.

    Args:
        model_path: Path to the huggingface model.
        modules_for_model_registration: List of python modules to register the model.
        max_length: Maximum sequence length.
        use_peft: Whether to use LoRA PEFT.
        save_peft_only: Whether to save only the PEFT weights.
        lora_r: LoRA r parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target modules.
        lora_modules_to_save: LoRA modules to save.
        lora_use_rslora: Whether to use RSLora.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        model_path: str | os.PathLike,
        modules_for_model_registration: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        if legacy_adapter_type is None:
            raise ValueError("Huggingface models can only be used with legacy adapters.")
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.model_path = model_path
        self.modules_for_model_registration = modules_for_model_registration or []
        self.max_length = max_length
        self.use_peft = use_peft
        self.save_peft_only = save_peft_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_modules_to_save = lora_modules_to_save
        self.lora_use_rslora = lora_use_rslora

    def setup(self):
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForTokenClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        import importlib

        for module in self.modules_for_model_registration:
            importlib.import_module(module)
        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = AutoModelForSequenceClassification
            peft_task_type = TaskType.SEQ_CLS
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = AutoModelForTokenClassification
            peft_task_type = TaskType.TOKEN_CLS
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = AutoModelForMaskedLM
            peft_task_type = TaskType.FEATURE_EXTRACTION
        elif self.legacy_adapter_type is None:
            model_class = AutoModel
            peft_task_type = TaskType.FEATURE_EXTRACTION
        else:
            raise ValueError(
                f"There is no standard huggingface head for the task type: {self.legacy_adapter_type}. "
                "Please create a backbone for your huggingfce model."
            )
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        def nested_set_config(config, config_overwrites):
            for k, v in config_overwrites.items():
                if isinstance(v, dict):
                    nested_set_config(getattr(config, k), v)
                else:
                    setattr(config, k, v)

        nested_set_config(config, self.config_overwrites)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, config=config, trust_remote_code=True
        )
        self.model, self.loading_info = model_class.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            output_loading_info=True,
            **self.model_init_args,
        )
        if self.use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=self.lora_target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                use_rslora=self.lora_use_rslora,
                inference_mode=False,
                modules_to_save=self.lora_modules_to_save,
            )
            self.model = get_peft_model(self.model, peft_config)
            rank_zero_only(self.model.print_trainable_parameters)()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        special_tokens_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Returns the final logits.

        Args:
            input_ids (Tensor): Token IDs
            attention_mask (Tensor): Attention mask

        Returns:
            Tensor: Logits
        """
        hf_output = self.model(input_ids, attention_mask=attention_mask)
        return SequenceBackboneOutput(
            last_hidden_state=hf_output.logits,
            hidden_states=hf_output.hidden_states,
            special_tokens_mask=special_tokens_mask,
            attention_mask=attention_mask,
        )

    def get_decoder(self) -> nn.Module:
        """Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return _Identity()

    def process_batch(
        self, batch: dict, device: torch.device, add_special_tokens: bool = True, **kwargs
    ):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.
            device (torch.device): Device to move the data to.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        seq_tokenized = self.tokenize(
            batch["sequences"], padding=True, add_special_tokens=add_special_tokens, **kwargs
        )
        for k, v in seq_tokenized.items():
            if v is not None:
                if torch.is_tensor(v):
                    seq_tokenized[k] = v.to(dtype=torch.long, device=device)
                else:
                    seq_tokenized[k] = torch.tensor(v, dtype=torch.long, device=device)
        return seq_tokenized

    def tokenize(self, sequences: list[str], **kwargs) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids, attention_mask, special_tokens_mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        output_keys = ["input_ids", "attention_mask", "special_tokens_mask"]
        return {k: seq_tokenized[k] for k in output_keys}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the arbitrary max context specified by the user

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the embedding size

        Returns:
            int: Embedding size
        """
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if not self.use_peft or not self.save_peft_only:
            return
        adapter_name = "default"
        peft_dict = get_peft_model_state_dict(self.model, adapter_name=adapter_name)
        prefixed_dict = {f"{prefix}.model.{k}": v for k, v in peft_dict.items()}
        head_keys = tuple(self.loading_info["missing_keys"])
        for k in list(checkpoint["state_dict"].keys()):
            if k.endswith(head_keys):
                # keep all newly added weights
                continue
            if k.replace(f".{adapter_name}", "") not in prefixed_dict and k not in prefixed_dict:
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)


class Enformer(HFSequenceBackbone):
    """Wraps [Enformer](https://github.com/lucidrains/enformer-pytorch) as a ModelGenerator backbone

    Note:
        Does not support LoRA

    Args:
        from_scratch: Whether to create the model from scratch.
        max_length: Maximum sequence length.
        frozen: Whether to freeze model.
        delete_crop_layer: Whether to delete cropping layer.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = 196_608,
        frozen: bool = False,
        delete_crop_layer: bool = False,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        if legacy_adapter_type is not None:
            legacy_adapter_type = None

        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.frozen = frozen
        self.delete_crop_layer = delete_crop_layer

    def setup(self):
        from modelgenerator.huggingface_models.enformer_pytorch import (
            Enformer,
            str_to_one_hot,
            EnformerConfig,
        )

        if self.from_scratch:
            config = EnformerConfig()
        else:
            config = EnformerConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)
        if self.from_scratch:
            model = Enformer(config=config, **self.model_init_args)
        else:
            model = Enformer.from_pretrained(self.model_path, config=config, **self.model_init_args)
        self.tokenizer = str_to_one_hot
        self.vocab_size = 6  # ACGTN., where . means padding
        self.encoder = model
        if self.use_legacy_adapter:
            self.decoder = model._heads
        else:
            self.decoder = None
        self.target_length = self.encoder.target_length
        if self.max_length is None:
            rank_zero_info("You didn't set a max_length for the data in the downstream task")
        if self.frozen:
            rank_zero_info(f"> {type(self.encoder).__name__} is frozen.")
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        all_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len, 4)
            attention_mask (torch.Tensor): Attention mask (n, target_length)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states
        """
        embeddings = self.encoder(
            input_ids.float(), return_only_embeddings=True, delete_crop_layer=self.delete_crop_layer
        )
        attention_mask = torch.ones(
            input_ids.size()[0], self.target_length, device=input_ids.device
        )  # (bs, target_length)
        return SequenceBackboneOutput(
            last_hidden_state=embeddings,
            hidden_states=[embeddings] if all_hidden_states else None,
            attention_mask=attention_mask,
        )

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        input_ids = self.tokenize(batch["sequences"])["input_ids"].to(device=device)
        return {"input_ids": input_ids}

    def tokenize(
        self,
        sequences: list[str],
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences, should be of the same length

        Returns:
            dict: contains input_ids, attention_mask
        """
        input_ids = self.tokenizer(sequences)  # onehot coding, of shape (bs, 197k, 4)
        return {"input_ids": input_ids}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        """Returns the one hot embedding of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            torch.tensor: one hot embedding
        """
        return self.tokenizer(token)

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.dim * 2

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the pre-trained model

        Returns:
            int: Vocabulary size
        """
        return self.vocab_size

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if not self.frozen:
            return
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith(f"{prefix}.encoder."):
                continue
            checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        # return self.encoder.config.depth
        # Hardcoding at 1 until other layers are exposed
        return 1


class Borzoi(HFSequenceBackbone):
    """Wraps [Borzoi](https://github.com/johahi/borzoi-pytorch) as a ModelGenerator backbone

    Note:
        Does not support LoRA

    Args:
        from_scratch: Whether to create the model from scratch.
        max_length: Maximum sequence length.
        frozen: Whether to freeze model.
        delete_crop_layer: Whether to skip cropping layer.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = 524_288,
        frozen: bool = False,
        delete_crop_layer: bool = False,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.frozen = frozen
        self.delete_crop_layer = delete_crop_layer

    def setup(self):
        from modelgenerator.huggingface_models.borzoi_pytorch import Borzoi
        from modelgenerator.huggingface_models.borzoi_pytorch.config_borzoi import BorzoiConfig
        from modelgenerator.huggingface_models.enformer_pytorch import str_to_one_hot

        if self.from_scratch:
            config = BorzoiConfig()
        else:
            config = BorzoiConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)
        if self.from_scratch:
            model = Borzoi(config=config, **self.model_init_args)
        else:
            model = Borzoi.from_pretrained(self.model_path, config=config, **self.model_init_args)
        self.tokenizer = str_to_one_hot
        self.vocab_size = 6  # ACGTN., where . means padding
        self.encoder = model
        if self.use_legacy_adapter:
            self.decoder = model.human_head
        else:
            self.decoder = None

        self.target_length = self.encoder.crop.target_length
        if self.max_length is None:
            rank_zero_info("You didn't set a max_length for the data in the downstream task")
        if self.frozen:
            rank_zero_info(f"> {type(self.encoder).__name__} is frozen.")
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        all_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, 4, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask (n, target_length)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states, hidden state (n, target_length, d)
        """
        embeddings = self.encoder(
            input_ids.float(), return_only_embeddings=True, delete_crop_layer=self.delete_crop_layer
        )
        attention_mask = torch.ones(
            input_ids.size()[0], self.target_length, device=input_ids.device
        )  # (bs, target_length)
        return SequenceBackboneOutput(
            last_hidden_state=embeddings,
            hidden_states=[embeddings] if all_hidden_states else None,
            attention_mask=attention_mask,
        )

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        input_ids = self.tokenize(batch["sequences"])["input_ids"].to(device=device)
        return {"input_ids": input_ids}

    def tokenize(
        self,
        sequences: list[str],
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences, should be of the same length

        Returns:
            dict: contains input_ids, attention_mask
        """
        input_ids = self.tokenizer(sequences).transpose(
            -1, -2
        )  # onehot coding, of shape (bs, 4, L), Borzoi
        return {"input_ids": input_ids}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        """Returns the one hot embedding of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            torch.tensor: one hot embedding
        """
        return self.tokenizer(token)

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return 1920

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the pre-trained model

        Returns:
            int: Vocabulary size
        """
        return self.vocab_size

    def on_save_checkpoint(self, checkpoint: dict):
        if not self.frozen:
            return
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith("backbone.encoder."):
                continue
            if self.frozen:
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        # return self.encoder.config.depth
        # Hardcoding at 1 until other layers are exposed
        return 1


class ESM(HFSequenceBackbone):
    """A wrapper for using ESM series model as backbone.

    Args:
        max_length: Maximum sequence length.
        use_peft: Whether to use LoRA PEFT.
        frozen: Whether to freeze encoder.
        save_peft_only: Whether to save only the PEFT weights.
        lora_r: LoRA r parameter.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout.
        lora_target_modules: LoRA target modules.
        lora_modules_to_save: LoRA modules to save.
        lora_use_rslora: Whether to use RSLora.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.max_length = max_length
        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_modules_to_save = lora_modules_to_save
        self.lora_use_rslora = lora_use_rslora

    def setup(self):
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForTokenClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = AutoModelForSequenceClassification
            peft_task_type = TaskType.SEQ_CLS
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = AutoModelForTokenClassification
            peft_task_type = TaskType.TOKEN_CLS
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = AutoModelForMaskedLM
            peft_task_type = TaskType.FEATURE_EXTRACTION
        elif self.legacy_adapter_type is None:
            model_class = AutoModel
            peft_task_type = TaskType.FEATURE_EXTRACTION
        else:
            raise ValueError(
                f"There is no standard huggingface head for the task type: {self.legacy_adapter_type}. "
                "Please create a backbone for your huggingface model."
            )
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        def nested_set_config(config, config_overwrites):
            for k, v in config_overwrites.items():
                if isinstance(v, dict):
                    nested_set_config(getattr(config, k), v)
                else:
                    setattr(config, k, v)

        nested_set_config(config, self.config_overwrites)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, config=config, trust_remote_code=True
        )
        model, self.loading_info = model_class.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            output_loading_info=True,
            **self.model_init_args,
        )
        if not self.use_legacy_adapter:
            self.encoder = model
        else:
            self.encoder = model.esm
        self.decoder = None
        if model_class == AutoModelForMaskedLM:
            self.decoder = model.lm_head
        elif model_class == AutoModelForTokenClassification:
            self.decoder = model.classifier
        elif model_class == AutoModelForSequenceClassification:
            self.decoder = model.classifier
        if self.use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=self.lora_target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                use_rslora=self.lora_use_rslora,
                inference_mode=False,
                modules_to_save=self.lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if self.frozen:
                rank_zero_info(f"> {type(self.encoder).__name__} is frozen.")
                for _, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        special_tokens_mask: Optional[Tensor] = None,
        all_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return SequenceBackboneOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if all_hidden_states else None,
            special_tokens_mask=special_tokens_mask,
            attention_mask=attention_mask,
        )

    def get_decoder(self) -> nn.Module:
        """
        Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.
            device (torch.device): Device to move the data to.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        seq_tokenized = self.tokenize(batch["sequences"])
        for k, v in seq_tokenized.items():
            if v is not None:
                if torch.is_tensor(v):
                    seq_tokenized[k] = v.to(dtype=torch.long, device=device)
                else:
                    seq_tokenized[k] = torch.tensor(v, dtype=torch.long, device=device)
        return seq_tokenized

    def tokenize(self, sequences: list[str], **kwargs) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids, attention_mask, special_tokens_mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        output_keys = ["input_ids", "attention_mask", "special_tokens_mask"]
        return {k: seq_tokenized[k] for k in output_keys}

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the arbitrary max context specified by the user

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the embedding size

        Returns:
            int: Embedding size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def on_save_checkpoint(self, checkpoint: dict, prefix: str = "backbone"):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(self.encoder, adapter_name=adapter_name)
            prefixed_dict = {f"{prefix}.encoder.{k}": v for k, v in peft_dict.items()}

        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith(f"{prefix}.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)


# TODO: This is not a Huggingface model, should inherit from a different class
class SCFoundation(HFSequenceBackbone):
    """Wraps SCFoundation model in ModelGenerator backbone with multiple gene embedding modes

    Note:
        Does not support LoRA.

    Args:
        num_genes: Number of genes in the model context.
        frozen: Whether to freeze model.
        output_type: Type of output embedding ('cell', 'gene', 'gene_batch', 'gene_expression').
        pool_type: Pooling type for cell embedding ('all', 'max').
        input_type: Input data type ('singlecell', 'bulk').
        pre_normalized: Whether input is pre-normalized ('T', 'F', 'A').
        train_last_n_layers: Number of layers to train in the encoder.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        num_genes: Optional[int] = 19264,
        frozen: bool = False,
        output_type: str = "cell",
        pool_type: str = "all",
        input_type: str = "singlecell",
        pre_normalized: str = "F",
        train_last_n_layers: int = 0,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
            **kwargs,
        )
        self.num_genes = num_genes
        self.frozen = frozen
        self.output_type = output_type
        self.pool_type = pool_type
        self.input_type = input_type
        self.pre_normalized = pre_normalized
        self.train_last_n_layers = train_last_n_layers

    def setup(self):
        from modelgenerator.huggingface_models.scfoundation.load_scfoundation import (
            load_model_frommmf,
            getEncoerDecoderData,
            gatherData,
        )

        self.gatherData = gatherData
        self.getEncoerDecoderData = getEncoerDecoderData

        # Load model
        if self.output_type == "cell":
            key = "cell"
        elif self.output_type in ["gene", "gene_batch", "gene_expression"]:
            key = "gene"
        else:
            raise ValueError("Invalid output_type")

        model_file = "models.ckpt"
        local_model_path = cached_file(self.model_path, model_file)
        model, self.model_config = load_model_frommmf(local_model_path, key)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.decoder = model.decoder if self.use_legacy_adapter else None
        if self.frozen:
            rank_zero_info(f"> {type(self.model).__name__} is frozen.")
            for _, param in self.model.named_parameters():
                param.requires_grad = False

            if self.train_last_n_layers > 0:
                num_layers = len(self.encoder.transformer_encoder)
                start_idx = max(0, num_layers - self.train_last_n_layers)

                # unfreeze train_last_n_layers
                for i in range(start_idx, num_layers):
                    rank_zero_info(f"> Unfreezing layer {i}")
                    for param in self.encoder.transformer_encoder[i].parameters():
                        param.requires_grad = True

    def _preprocess_input(self, input_data: Tensor) -> Tensor:
        """Preprocess input data based on input type and normalization settings"""
        if self.input_type == "bulk":
            if self.pre_normalized == "T":
                total_count = input_data.sum(dim=1, keepdim=True)
            elif self.pre_normalized == "F":
                total_count = torch.log10(input_data.sum(dim=1, keepdim=True))
            else:
                raise ValueError("pre_normalized must be T or F for bulk input")

            return torch.cat([input_data, total_count.repeat(1, 2)], dim=1)

        elif self.input_type == "singlecell":
            if self.pre_normalized == "F":
                input_data = torch.log1p(input_data / input_data.sum(dim=1, keepdim=True) * 1e4)
                total_count = input_data.sum(dim=1, keepdim=True)
            elif self.pre_normalized == "T":
                total_count = input_data.sum(dim=1, keepdim=True)
            elif self.pre_normalized == "A":
                total_count = input_data[:, -1:]
                input_data = input_data[:, :-1]
            else:
                raise ValueError("pre_normalized must be T, F or A for single cell input")

            return torch.cat(
                [input_data, torch.log10(total_count), torch.log10(total_count)], dim=1
            )

    def forward(
        self,
        input_ids: Tensor,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass with multiple embedding modes

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask
            all_hidden_states: Whether to return all hidden states

        Returns:
            Embeddings based on output_type setting
        """
        x = input_ids
        value_labels = x > 0
        x, x_padding = self.gatherData(x, value_labels, self.model_config["pad_token_id"])

        data_gene_ids = torch.arange(self.num_genes + 2, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = self.gatherData(
            data_gene_ids, value_labels, self.model_config["pad_token_id"]
        )
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb
        embeddings = self.encoder(x, x_padding)
        return SequenceBackboneOutput(
            last_hidden_state=embeddings,
        )

    def get_decoder(self) -> Optional[nn.Module]:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        sc_seq = batch["sequences"]
        if torch.is_tensor(sc_seq):
            sc_seq = sc_seq.to(dtype=torch.bfloat16, device=device)
        else:
            sc_seq = torch.tensor(sc_seq, dtype=torch.bfloat16, device=device)
        return {"input_ids": self._preprocess_input(sc_seq).to(device=device, dtype=torch.bfloat16)}

    def tokenize(
        self,
        sequences: list[str],
        **kwargs,
    ) -> dict:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            dict: contains input_ids
        """
        return {"input_ids": sequences}

    def get_max_context(self) -> int:
        """Gets maximum context length"""
        return self.max_length

    def get_embedding_size(self) -> int:
        """Gets embedding size"""
        return self.model_config["encoder"]["hidden_dim"]

    def get_num_layer(self) -> int:
        """Gets number of layers"""
        return len(self.encoder.transformer_encoder)

    def on_save_checkpoint(self, checkpoint: dict):
        """Handles checkpoint saving"""
        if not self.frozen:
            return
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith("backbone.model."):
                continue
            if self.frozen:
                checkpoint["state_dict"].pop(k)


class Geneformer(HFSequenceBackbone):
    """Geneformer model for single-cell transcriptomics inference

    Note:
        Does not support LoRA.

    Args:
        from_scratch: Whether to initialize from random weights.
        max_length: Maximum input sequence length.
        emb_layer: Layer to extract embeddings from.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: int = 4096,
        emb_layer: int = -2,
        config_overwrites: Optional[dict] = None,
        model_init_args: Optional[dict] = None,
    ):
        # initialize base class with adapters
        super().__init__(
            legacy_adapter_type,
            default_config,
            config_overwrites=config_overwrites,
            model_init_args=model_init_args,
        )
        self.from_scratch = from_scratch
        self.max_length = max_length
        self.emb_layer = emb_layer

    def setup(self):
        from modelgenerator.huggingface_models.geneformer import TranscriptomeTokenizer
        from transformers import (
            BertConfig,
            BertModel,
            BertForMaskedLM,
        )

        if self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = BertForMaskedLM
        else:
            model_class = BertModel
            self.model_init_args = {"add_pooling_layer": False, **self.model_init_args}

        if self.from_scratch:
            config = BertConfig()
        else:
            config = BertConfig.from_pretrained(self.model_path, trust_remote_code=True)

        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        # instantiate model
        if self.from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path,
                config=config,
                trust_remote_code=True,
                **self.model_init_args,
            )

        # set encoder and decoder according to legacy adapter usage
        if self.use_legacy_adapter:
            self.encoder = model.bert
            self.decoder = model.cls
        else:
            self.encoder = model
            self.decoder = None

        # Tokenizer and other attributes
        self.tokenizer = TranscriptomeTokenizer(
            model_input_size=self.max_length,
            special_token=True,
            collapse_gene_ids=True,
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Return encoder outputs (last hidden state)."""

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        return SequenceBackboneOutput(
            last_hidden_state=outputs.hidden_states[self.emb_layer],
        )

    def get_decoder(self) -> Optional[nn.Module]:
        """Returns the pre-trained decoder (if using legacy adapter)."""
        return self.decoder

    def process_batch(self, batch: dict, device: torch.device, **kwargs):
        """Processes a batch of sequences to model input format.

        Args:
            batch (dict): A dictionary containing input sequences.

        Returns:
            Dict: A dictionary containing required args for forward pass.
        """
        seq_tokenized = self.tokenize(batch["sequences"])
        seq_tokenized["input_ids"] = seq_tokenized["input_ids"].to(device=device)
        seq_tokenized["attention_mask"] = torch.tensor(
            seq_tokenized["attention_mask"], dtype=torch.long, device=device
        )
        return seq_tokenized

    def tokenize(self, input_data, **kwargs):
        """Tokenise without reordering so embeddings align with inputs."""
        import modelgenerator.huggingface_models.geneformer.perturber_utils as pu

        if "ensemble_id" in kwargs:
            ensembl_ids = kwargs["ensemble_id"]
            result = self.tokenizer.process_input_dict(input_data, ensembl_ids)
        else:
            result = self.tokenizer.process_input_dict(input_data)

        input_ids = [torch.tensor(seq, dtype=torch.long) for seq in result["input_ids"]]

        model_input_size = self.max_length
        pad_token_id = 0
        max_len = max(len(seq) for seq in input_ids)

        padded_input_ids = pu.pad_tensor_list(input_ids, max_len, pad_token_id, model_input_size)

        original_lens = [len(seq) for seq in input_ids]
        attention_mask = [
            [1] * original_len + [0] * (max_len - original_len)
            if original_len <= max_len
            else [1] * max_len
            for original_len in original_lens
        ]

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
        }

    def get_embedding_size(self):
        return self.encoder.config.hidden_size

    def get_max_context(self):
        return self.encoder.config.max_position_embeddings


class SCimilarity(HFSequenceBackbone):
    """Wraps SCimilarity model in ModelGenerator backbone with optional legacy adapter support.

    Args:
        legacy_adapter_type: Type of legacy adapter or None
        default_config: Default configuration object
        model_path: Path to pretrained SCimilarity model files
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        num_genes: int = 28231,
        **kwargs,
    ):
        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.num_genes = num_genes

    def setup(self):
        # Cache model files
        self._local_encoder_path = cached_file(self.model_path, "model_v1.1/encoder.ckpt")
        if self.use_legacy_adapter:
            self._local_decoder_path = cached_file(self.model_path, "model_v1.1/decoder.ckpt")

        # Load architecture, weights, and set up modules
        self._load_model()

        # Disable decoder if not using legacy adapter
        if not self.use_legacy_adapter:
            self.decoder = None

    def _load_model(self):
        import json
        from modelgenerator.huggingface_models.scimilarity.nn_models import Encoder, Decoder

        # Load layer configuration
        layer_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "huggingface_models",
            "scimilarity",
            "model_v1.1",
            "layer_sizes.json",
        )
        with open(layer_file, "r") as f:
            layer_sizes = json.load(f)

        # Determine latent and hidden dimensions
        layers = [
            (name, layer_sizes[name])
            for name in sorted(layer_sizes)
            if "weight" in name and len(layer_sizes[name]) > 1
        ]
        self.latent_dim = layers[-1][1][0]
        hidden_dims = [dims[1][0] for dims in layers[:-1]]

        # Build encoder
        self.encoder = Encoder(
            n_genes=self.num_genes,
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dims,
        )
        self.encoder.load_state(self._local_encoder_path)

        # Build decoder for legacy adapter
        if self.use_legacy_adapter:
            self.decoder = Decoder(
                n_genes=self.num_genes,
                latent_dim=self.latent_dim,
                hidden_dim=list(reversed(hidden_dims)),
            )
            ckpt = torch.load(self._local_decoder_path)
            state = ckpt.get("state_dict", ckpt)
            self.decoder.load_state_dict(state)

        # Set input/output dimensions
        self.input_dim = self.output_dim = self.num_genes

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        all_hidden_states: bool = False,
        **kwargs,
    ):
        """Compute embeddings for pre-aligned gene expression data"""
        x = input_ids
        embeddings = self.encoder(x)

        return SequenceBackboneOutput(
            last_hidden_state=embeddings,
        )

    def get_decoder(self) -> Optional[nn.Module]:
        """Return the legacy decoder if in adapter mode"""
        return self.decoder if self.use_legacy_adapter else None

    def tokenize(
        self,
        sequences: List[str],
        padding: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        # dummy tokenizer for scimilarity
        return {"input_ids": sequences}

    def process_batch(self, batch: dict, device: torch.device, **kwargs) -> dict:
        data = batch.get("sequences")
        tensor = (data if torch.is_tensor(data) else torch.tensor(data, dtype=torch.bfloat16)).to(
            device
        )
        # Normalize counts
        counts = tensor.div(tensor.sum(dim=1, keepdim=True) + 1e-8) * 1e4
        counts = torch.log1p(counts)
        return {"input_ids": counts}

    def get_embedding_size(self) -> int:
        return self.latent_dim

    def get_num_layer(self) -> int:
        # Count linear layers as encoder depth
        return len([l for l in self.encoder.network if isinstance(l, nn.Linear)])
