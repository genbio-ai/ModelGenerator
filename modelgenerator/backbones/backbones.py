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


class GenBioBERT(HFSequenceBackbone):
    """GenBioBERT model

    Note:
        Models using this interface include `aido_dna_7b`, `aido_dna_300m`, `dna_dummy`, `aido_dna_debug`,
        `aido_rna_1b600m`, `aido_rna_1b600m_cds`, `aido_rna_1m_mars`, `aido_rna_25m_mars`, `aido_rna_300m_mars`,
        `aido_rna_650m`, `aido_rna_650m_cds`.

        FSDP auto_wrap_policy is `modelgenerator.distributed.fsdp.wrap.AutoWrapPolicy`

    Args:
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 32.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value"].
    """

    fsdp_wrap_modules = [
        "modelgenerator.huggingface_models.rnabert.modeling_rnabert.RNABertLayer"
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
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = ["query", "value"],
        **kwargs,
    ):
        # Delays hf model imports to avoid model name conflicts
        from modelgenerator.huggingface_models.rnabert import (
            RNABertConfig,
            RNABertTokenizer,
            RNABertModel,
            RNABertForMaskedLM,
            RNABertForTokenClassification,
            RNABertForSequenceClassification,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        repo_base_dir = Path(__file__).resolve().parent.parent.parent
        vocab_file = os.path.join(
            repo_base_dir, "modelgenerator/huggingface_models/rnabert/vocab.txt"
        )
        self.max_length = max_length
        self.tokenizer = RNABertTokenizer(
            vocab_file, version="v2"
        )  # add [CLS] ... [SEP]
        if self.use_legacy_adapter:
            rank_zero_info(
                "You are using a legacy adapter/head, so its configuration has to be "
                "set explicitly under backbone. This is done using "
                "`model.backbone.config_overwrites` and `model.backbone.model_init_args`."
            )
        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = RNABertForSequenceClassification
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = RNABertForTokenClassification
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = RNABertForMaskedLM
        else:
            model_class = RNABertModel

        if from_scratch:
            config = RNABertConfig()
        else:
            config = RNABertConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
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

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.max_length = max_length
        if max_length is None:
            rank_zero_info(
                "You didn't set a max_length for the data in the downstream task"
            )
        if use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                modules_to_save=[],
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
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
        if all_hidden_states:
            return outputs.hidden_states
        return outputs.last_hidden_state

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
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            padding=padding,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=self.max_length is not None,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_mask = seq_tokenized["special_tokens_mask"]
        return input_ids, attention_mask, special_mask

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
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
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
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value", "key", "dense", "router"].
        lora_modules_to_save (Optional[List[str]], optional): LoRA modules to save. Defaults to None.
        lora_use_rslora (bool, optional): Whether to use RSLora. Defaults to False.
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
        **kwargs,
    ):
        from modelgenerator.huggingface_models.fm4bio import (
            FM4BioConfig,
            FM4BioTokenizer,
            FM4BioModel,
            FM4BioForMaskedLM,
            FM4BioForTokenClassification,
            FM4BioForSequenceClassification,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.max_length = max_length
        if self.use_legacy_adapter:
            rank_zero_info(
                "You are using a legacy adapter/head, so its configuration has to be "
                "set explicitly under backbone. This is done using "
                "`model.backbone.config_overwrites` and `model.backbone.model_init_args`."
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

        if from_scratch:
            config = FM4BioConfig()
        else:
            config = FM4BioConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
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

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        if use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:  # use linear probing, freeze all parameters
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if all_hidden_states:
            return outputs.hidden_states
        return outputs.last_hidden_state

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=padding,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=True,
        )

        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_mask = seq_tokenized["special_tokens_mask"]
        return input_ids, attention_mask, special_mask

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
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
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
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value", "key", "dense", "router"].
        lora_modules_to_save (Optional[List[str]], optional): LoRA modules to save. Defaults to None.
        lora_use_rslora (bool, optional): Whether to use RSLora. Defaults to False.
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
        **kwargs,
    ):
        from modelgenerator.huggingface_models.cellfoundation import (
            CellFoundationConfig,
            CellFoundationModel,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.max_length = max_length
        # Note: Legacy adapters are for older sequence models.
        if legacy_adapter_type is not None:
            # raise NotImplementedError(
            #     "Legacy adapters are not implemented for CellFoundation."
            # )
            legacy_adapter_type = None
        model_class = CellFoundationModel
        peft_task_type = TaskType.FEATURE_EXTRACTION

        if from_scratch:
            config = CellFoundationConfig()
        else:
            config = CellFoundationConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        if self.training:
            model = model.train()
        self.encoder = model
        self.decoder = None

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        if use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """
        X = torch.tensor(
            input_ids, dtype=torch.float32
        )  # Converting from torch.long; should be counts.

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
        if all_hidden_states:
            return (x[:, :-2, :] for x in outputs.hidden_states)
        return outputs.last_hidden_state[:, :-2, :]

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
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Note:
            This is a dummy tokenizer since the CellFoundation models consume gene expression.

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
        """
        return sequences, None, None

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
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
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
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value", "key", "dense", "router"].
        lora_modules_to_save (Optional[List[str]], optional): LoRA modules to save. Defaults to None.
        lora_use_rslora (bool, optional): Whether to use RSLora. Defaults to False.
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
        **kwargs,
    ):
        from modelgenerator.huggingface_models.cellspatialfoundation import (
            CellSpatialFoundationConfig,
            CellSpatialFoundationModel,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.max_length = max_length
        # Note: Legacy adapters are for older sequence models.
        if legacy_adapter_type is not None:
            raise NotImplementedError(
                "Legacy adapters are not implemented for CellFoundation."
            )
        model_class = CellSpatialFoundationModel
        peft_task_type = TaskType.FEATURE_EXTRACTION

        if from_scratch:
            config = CellSpatialFoundationConfig()
        else:
            config = CellSpatialFoundationConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        self.encoder = model
        self.decoder = None

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        if use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

        self.config = config
        self.rope2d_use_xy = rope2d_use_xy
        self.sep_value = sep_value

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
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
            return (x[:, :-2, :] for x in outputs.hidden_states)

        if cell_num > 1:
            # multiple cells, pool over 1st cell then return embedding of 1st cell
            # need get dim from outputs.last_hidden_state when encoder_cell_id==0 and others fill with pad dim, and set pad postion =0 in encoder_attention_mask
            if self.rope2d_use_xy:
                cell_id_full = (
                    torch.tensor(
                        range(sep_idx_2 - sep_idx_1 - 1), device=encoder_data.device
                    )
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

            first_cell_max_len_emb = outputs.last_hidden_state[
                :, 0:first_cell_max_len, :
            ]

            first_cell_emb = torch.where(
                first_cell_mask_full == False, first_cell_max_len_emb, first_cell_emb
            )

            return first_cell_emb

        else:
            # single-cell, pool over all genes
            return outputs.last_hidden_state

    def _process_input(self, input_ids):

        sep_idx_1, sep_idx_2 = None, None
        if self.rope2d_use_xy:
            sep_value_idx = (input_ids == self.sep_value).nonzero(as_tuple=True)
            sep_idx_1 = sep_value_idx[1][0]
            sep_idx_2 = sep_value_idx[1][1]

            coordinate_x = (
                input_ids[:, sep_idx_1 + 1: sep_idx_2]
                .clone()
                .repeat_interleave(self.config.max_position_embeddings + 2, dim=1)
            )
            coordinate_y = (
                input_ids[:, sep_idx_2 + 1:]
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
            X = torch.tensor(
                input_ids, dtype=torch.bfloat16
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
            cell_num = 1
        elif input_ids.shape[1] / self.config.max_position_embeddings > 1:
            X_ls = []
            cell_num = int(input_ids.shape[1] / self.config.max_position_embeddings)
            for cell_idx in range(cell_num):
                X = torch.tensor(
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
        encoder_attention_mask = (
            ~encoder_data_padding
        ).long()  # 1 for not mask, 0 for mask

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
                data_gene_ids = torch.arange(X.shape[1], device=X.device).repeat(
                    X.shape[0], 1
                )
        encoder_position_gene_ids, _ = gather_data(
            data_gene_ids, encoder_data_labels, self.config.pad_token_id
        )
        if self.rope2d_use_xy:
            encoder_position_gene_ids[
                encoder_position_gene_ids == self.config.pad_token_id
            ] = 0  # set x of pad token as 0, due to cal range
        else:
            encoder_position_gene_ids[
                encoder_position_gene_ids == self.config.pad_token_id
            ] = (self.config.max_position_embeddings + 2)

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

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Note:
            This is a dummy tokenizer since the CellSpatialFoundation models consume gene expression.
            It returns the sequences as is but generates the attention mask that matches the
            output of the encoder.

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
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
        ) = self._process_input(sequences)

        if cell_num > 1:
            # multiple cells, pool over 1st cell then return embedding of 1st cell
            # need get dim from outputs.last_hidden_state when encoder_cell_id==0
            # and others fill with pad dim, and set pad postion = 0 in encoder_attention_mask
            if self.rope2d_use_xy:
                cell_id_full = (
                    torch.tensor(
                        range(sep_idx_2 - sep_idx_1 - 1), device=encoder_data.device
                    )
                    .view(1, -1)
                    .repeat(X.shape[0], 1)
                    .repeat_interleave(self.config.max_position_embeddings + 2, dim=1)
                )
                cell_id_encoder, _ = gather_data(
                    cell_id_full, encoder_data_labels, self.config.pad_token_id
                )
                first_cell_max_len = (cell_id_encoder == 0).sum(dim=1).max()
                first_cell_mask = cell_id_encoder[:, :first_cell_max_len] == 0
            else:
                first_cell_max_len = (encoder_rope_id[:, 1, :] == 0).sum(dim=1).max()
                first_cell_mask = encoder_rope_id[:, 1, :first_cell_max_len] == 0
            return sequences, first_cell_mask.long(), None
        return sequences, encoder_attention_mask, None

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
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
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


class Onehot(HFSequenceBackbone):
    """Tokenizer-only model for one-hot encoding. Useful for baseline model testing (CNNs, linear, etc.)

    Note:
        Models using this interface include `dna_onehot` and `protein_onehot`.

        Does not contain any parameters, and cannot be used without an adapter.

    Args:
        vocab_file (str, optional): Path to the vocabulary file. Defaults to
            "modelgenerator/huggingface_models/rnabert/vocab.txt".
        max_length (Optional[int], optional): Maximum sequence length. Defaults to 512.
    """

    fsdp_wrap_modules = [
        "modelgenerator.huggingface_models.rnabert.modeling_rnabert.RNABertLayer"
    ]

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
        **kwargs,
    ):
        from modelgenerator.huggingface_models.fm4bio import FM4BioTokenizer

        super().__init__(None, None, **kwargs)
        self.max_length = max_length
        if vocab_file is not None:
            self.vocab_file = vocab_file
        self.tokenizer = FM4BioTokenizer(self.vocab_file, version="v1")

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Returns one-hot encoding of input_ids.

        Args:
            input_ids (Tensor): Token IDs
            attention_mask (Tensor): Attention mask

        Returns:
            Tensor: One-hot encoding of input_ids
        """
        one_hot = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.tokenizer.vocab_size,
        ).to(input_ids.device)
        one_hot.scatter_(2, input_ids.unsqueeze(2), 1)
        return one_hot

    def get_decoder(self) -> nn.Module:
        """Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return _Identity()

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask.
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=True,
            padding=padding,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_mask = seq_tokenized["special_tokens_mask"]
        return input_ids, attention_mask, special_mask

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

        Warning: This is an experimental feature, don't expect it to work with all models.
        Downstream task support is also extremely limited to the standard huggingface heads.
        Its usage often involves manual configuration of the model's head through `config_overwrites`.

    Args:
        model_path (str): Path to the huggingface model
        max_length (int, optional): Maximum sequence length. Defaults to None.
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
        **kwargs,
    ):
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForTokenClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        import importlib

        if legacy_adapter_type is None:
            raise ValueError(
                "Huggingface models can only be used with legacy adapters."
            )
        modules_for_model_registration = modules_for_model_registration or []
        for module in modules_for_model_registration:
            importlib.import_module(module)
        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.model_path = model_path
        self.max_length = max_length
        self.use_peft = use_peft
        self.save_peft_only = save_peft_only
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
        if use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.model = get_peft_model(self.model, peft_config)
            rank_zero_only(self.model.print_trainable_parameters)()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Returns the final logits.

        Args:
            input_ids (Tensor): Token IDs
            attention_mask (Tensor): Attention mask

        Returns:
            Tensor: Logits
        """
        return self.model(input_ids, attention_mask).logits

    def get_decoder(self) -> nn.Module:
        """Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return _Identity()

    def tokenize(self, sequences: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_tokens_mask = seq_tokenized["special_tokens_mask"]
        return input_ids, attention_mask, special_tokens_mask

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
            if (
                k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)


class Enformer(HFSequenceBackbone):
    """Wrap Enformer https://github.com/lucidrains/enformer-pytorch in ModelGenerator backbone

    Note: Do not support LoRA

    Args:
        legacy_adapter_type (LegacyAdapterType, None): Type of legacy adapter, setting it to None disables it.
        default_config (dict, None): Default values set by downstream tasks. Defaults to None.
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 196_608.
        frozen (bool, optional): Whether to freeze model. Defaults to False.
        delete_crop_layer: Whether to delete cropping layer. Defaults to False.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = 196_608,
        frozen: bool = False,
        delete_crop_layer: bool = False,
        **kwargs,
    ):
        from modelgenerator.huggingface_models.enformer_pytorch import Enformer, str_to_one_hot, EnformerConfig

        if legacy_adapter_type is not None:
            legacy_adapter_type = None

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        if from_scratch:
            config = EnformerConfig()
        else:
            config = EnformerConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)
        if from_scratch:
            model = Enformer(config=config, **self.model_init_args)
        else:
            model = Enformer.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        self.tokenizer = str_to_one_hot
        self.vocab_size = 6  # ACGTN., where . means padding
        self.encoder = model
        if self.use_legacy_adapter:
            self.decoder = model._heads
        else:
            self.decoder = None
        self.target_length = self.encoder.target_length
        self.max_length = max_length
        if max_length is None:
            rank_zero_info(
                "You didn't set a max_length for the data in the downstream task"
            )
        self.frozen = frozen
        if frozen:
            rank_zero_info(f"> {type(self.encoder).__name__} is frozen.")
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
        self.delete_crop_layer = delete_crop_layer


    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        all_hidden_states: bool = False,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len, 4)
            attention_mask (torch.Tensor): Attention mask (n, target_length)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states
        """
        embeddings = self.encoder(input_ids.float(), return_only_embeddings=True, delete_crop_layer=self.delete_crop_layer)
        if all_hidden_states:
            return [embeddings]
        return embeddings

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
        add_special_tokens: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences, should be of the same length

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask
        """
        input_ids = self.tokenizer(sequences)  # onehot coding, of shape (bs, 197k, 4)
        attention_mask = torch.ones(
            input_ids.size()[0], self.target_length
        )  # (bs, target_length)
        special_mask = None
        return input_ids, attention_mask, special_mask

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
    """Wrap Borzoi https://github.com/johahi/borzoi-pytorch in ModelGenerator backbone

    Note: Do not support LoRA

    Args:
        legacy_adapter_type (LegacyAdapterType, None): Type of legacy adapter, setting it to None disables it.
        default_config (dict, None): Default values set by downstream tasks. Defaults to None.
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 196_608.
        frozen (bool, optional): Whether to freeze model. Defaults to False.
        delete_crop_layer: Whether to skip cropping layer. Defaults to False.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = 196_608,
        frozen: bool = False,
        delete_crop_layer: bool = False,
        **kwargs,
    ):
        from modelgenerator.huggingface_models.borzoi_pytorch import Borzoi
        from modelgenerator.huggingface_models.borzoi_pytorch.config_borzoi import BorzoiConfig
        from modelgenerator.huggingface_models.enformer_pytorch import str_to_one_hot

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        if from_scratch:
            config = BorzoiConfig()
        else:
            config = BorzoiConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)
        if from_scratch:
            model = Borzoi(config=config, **self.model_init_args)
        else:
            model = Borzoi.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        self.tokenizer = str_to_one_hot
        self.vocab_size = 6  # ACGTN., where . means padding
        self.encoder = model
        if self.use_legacy_adapter:
            self.decoder = model.human_head
        else:
            self.decoder = None

        self.target_length = self.encoder.crop.target_length
        self.max_length = max_length
        if max_length is None:
            rank_zero_info(
                "You didn't set a max_length for the data in the downstream task"
            )
        self.frozen = frozen
        if frozen:
            rank_zero_info(f"> {type(self.encoder).__name__} is frozen.")
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
        self.delete_crop_layer = delete_crop_layer

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        all_hidden_states: bool = False,
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, 4, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask (n, target_length)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states
        """
        embeddings = self.encoder(input_ids.float(), return_only_embeddings=True, delete_crop_layer=self.delete_crop_layer, )
        if all_hidden_states:
            return [embeddings]
        return embeddings

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
        add_special_tokens: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences, should be of the same length

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask
        """
        input_ids = self.tokenizer(sequences).transpose(-1,-2)  # onehot coding, of shape (bs, 4, L), Borzoi
        attention_mask = torch.ones(
            input_ids.size()[0], self.target_length
        )  # (bs, target_length)
        special_mask = None
        return input_ids, attention_mask, special_mask

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
    """A wrapper allows for using ESM series model as backbone.

    Args:
        legacy_adapter_type (LegacyAdapterType, None): Type of legacy adapter, setting it to None disables it.
        default_config (dict, None): Default values set by downstream tasks. Defaults to None.
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        max_length (int, optional): Maximum sequence length. Defaults to 1024.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 32.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value"].
        lora_modules_to_save (Optional[List[str]], optional): LoRA modules to save. Defaults to None.
        lora_use_rslora (bool, optional): Whether to use RSLora. Defaults to False.
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
        **kwargs,
    ):
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForTokenClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.max_length = max_length
        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
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
        if use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if frozen:
                rank_zero_info(f"> {type(self.encoder).__name__} is frozen.")
                for _, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
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
        if all_hidden_states:
            return outputs.hidden_states
        return outputs.hidden_states[-1]

    def get_decoder(self) -> nn.Module:
        """
        Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def tokenize(self, sequences: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_tokens_mask = seq_tokenized["special_tokens_mask"]
        return input_ids, attention_mask, special_tokens_mask

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
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
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
        Models using this interface include `aido_scfoundation`

    Args:
        legacy_adapter_type: Type of legacy adapter
        default_config: Default values set by downstream tasks
        max_length: Maximum sequence length
        frozen: Whether to freeze model
        output_type: Type of output embedding ('cell', 'gene', 'gene_batch', 'gene_expression')
        pool_type: Pooling type for cell embedding ('all', 'max')
        input_type: Input data type ('singlecell', 'bulk')
        pre_normalized: Whether input is pre-normalized ('T', 'F', 'A')
        train_last_n_layers: Number of layers to train in the encoder
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
        **kwargs,
    ):

        from ..biomap_models.scfoundation.load_scfoundation import (
            load_model_frommmf,
            getEncoerDecoderData,
            gatherData,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.frozen = frozen
        self.num_genes = num_genes
        self.output_type = output_type
        self.pool_type = pool_type
        self.input_type = input_type
        self.pre_normalized = pre_normalized

        self.gatherData = gatherData
        self.getEncoerDecoderData = getEncoerDecoderData

        # Load model
        if output_type == "cell":
            key = "cell"
        elif output_type in ["gene", "gene_batch", "gene_expression"]:
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
        if frozen:
            rank_zero_info(f"> {type(self.model).__name__} is frozen.")
            for _, param in self.model.named_parameters():
                param.requires_grad = False

            if train_last_n_layers > 0:
                num_layers = len(self.encoder.transformer_encoder)
                start_idx = max(0, num_layers - train_last_n_layers)

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
                input_data = torch.log1p(
                    input_data / input_data.sum(dim=1, keepdim=True) * 1e4
                )
                total_count = input_data.sum(dim=1, keepdim=True)
            elif self.pre_normalized == "T":
                total_count = input_data.sum(dim=1, keepdim=True)
            elif self.pre_normalized == "A":
                total_count = input_data[:, -1:]
                input_data = input_data[:, :-1]
            else:
                raise ValueError(
                    "pre_normalized must be T, F or A for single cell input"
                )

            return torch.cat(
                [input_data, torch.log10(total_count), torch.log10(total_count)], dim=1
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        all_hidden_states: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass with multiple embedding modes

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask
            all_hidden_states: Whether to return all hidden states

        Returns:
            Embeddings based on output_type setting
        """
        # Preprocess input
        x = self._preprocess_input(input_ids)
        value_labels = x > 0
        x, x_padding = self.gatherData(
            x, value_labels, self.model_config["pad_token_id"]
        )

        data_gene_ids = torch.arange(self.num_genes + 2, device=x.device).repeat(
            x.shape[0], 1
        )
        position_gene_ids, _ = self.gatherData(
            data_gene_ids, value_labels, self.model_config["pad_token_id"]
        )
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb
        embeddings = self.encoder(x, x_padding)

        return embeddings

    def get_decoder(self) -> Optional[nn.Module]:
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
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Note:
            This is a dummy tokenizer from since CellFoundation.

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
        """
        return sequences, None, None

    def decode_tokens(self, tokenized_sequences: Tensor) -> List[str]:
        """Decodes tokenized sequences"""
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        """Gets token ID"""
        raise NotImplementedError

    def get_max_context(self) -> int:
        """Gets maximum context length"""
        return self.max_length

    def get_embedding_size(self) -> int:
        """Gets embedding size"""
        return self.model_config["encoder"]["hidden_dim"]

    def get_vocab_size(self) -> int:
        """Gets vocabulary size"""
        raise NotImplementedError

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
