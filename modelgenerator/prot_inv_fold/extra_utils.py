import torch
from time import time
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from torch.distributions.categorical import Categorical

import lightning.pytorch as pl

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import grad_norm

from torchmetrics.regression import R2Score
from torchmetrics import Metric
import pickle
import warnings

import argparse
from pathlib import Path

from .data_inverse_folding.datamodule import ProteinInverseFoldingDataModule
from .proteinMPNN.proteinMPNN import ProteinMPNNCMLM
from .adapter import (
    ProteinInverseFoldingPredictionHead,
    VanillaMDLMHead,
    MultiheadAttnMDLMHead,
)

from peft import LoraConfig, TaskType, get_peft_model


class MyAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, acc_sum: Tensor, batch_size: Tensor) -> None:
        self.correct += torch.tensor(acc_sum.clone()).float()
        self.total += torch.tensor(batch_size).float()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total.float()


### Args to replicate protein-mpnn-cmlm. Adapted from https://github.com/BytedProtein/ByProt/tree/main
protein_mpnn_args = {
    "encoder": {
        "d_model": 128,
        "d_node_feats": 128,
        "d_edge_feats": 128,
        "k_neighbors": 48,
        "augment_eps": 0.0,
        "n_enc_layers": 3,
        "dropout": 0.1,
        "n_vocab": 22,
        "n_dec_layers": 3,
        "random_decoding_order": True,
        "nar": True,
        "crf": False,
        "use_esm_alphabet": False,
    },
    "adapter_layer_indices": [32],
    "separate_loss": True,
    "initialize_input": True,
}


def get_lm_from_checkpoint(model_location, args):
    lm_configuration = FM4BioConfig.from_pretrained(model_location)
    lm_configuration.hidden_dropout_prob = args.drop_prob  # 0.1
    lm_configuration.attention_probs_dropout_prob = args.drop_prob  # 0.1
    lm = FM4BioForMaskedLM.from_pretrained(
        model_location,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        config=lm_configuration,
    )
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules="all-linear",  # ['query', 'value'], #'all-linear',
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            inference_mode=False,  ## TODO: what to set for inference_mode?
        )
        lm = get_peft_model(lm, lora_config)
        print(lm)
        print("[INFO]: L0RA lm.print_trainable_parameters():")
        print(lm.print_trainable_parameters())
        # exit()
    return lm


# implement LoRA
def get_random_model(args):
    model_config = FM4BioConfig(
        vocab_size=128,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=2048,
        intermediate_size=32,
        normalization_type="RMSNorm",
        add_linear_bias=True,
        position_embedding_type="rope",
        moe=True,
        num_experts=8,
        experts_per_token=2,
        use_lm_head=False,  # the BERT in megatron-deepspeed is implemented with GPT actually, so it does not have lm head
        tie_word_embeddings=False,
        hidden_dropout_prob=args.drop_prob,
        attention_probs_dropout_prob=args.drop_prob,
        return_dict=True,
    )
    lm = FM4BioForMaskedLM(
        model_config,
        # conv_kernel_size=args.conv_kernel_size,
        # pooling=args.pooling,
        # dropout_prob=args.drop_prob,
        # augment_with_zeroshot=False,
    )

    # target_modules: modules to apply the adapter to, modules_to_save: modules apart from adapter layers that are not frozen and updated during the training
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["query", "value"],  #'all-linear',
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            inference_mode=False,
        )
        lm = get_peft_model(lm, lora_config)
        print(lm)
        print("[INFO]: L0RA lm.print_trainable_parameters():")
        print(lm.print_trainable_parameters())

    return lm


def get_lm(args, model_location=None, verbose=0):
    if model_location:
        if args.lm_modality == "protein":
            lm = get_lm_from_checkpoint(model_location, args)
            vocab_file = "./vocab_protein.txt"
            tokenizer = FM4BioTokenizer(vocab_file=vocab_file, biotype="protein")
        else:
            raise NotImplementedError()
    if args.lm:
        lm = args.lm
        tokenizer = args.lm.tokenizer
    else:
        raise NotImplementedError()
        # print('[WARNING] initializing random model for debugging!')
        # if args.lm_modality == 'protein':
        #     lm = get_random_model(args)
        # else:
        #     raise NotImplementedError()

    if verbose > 0:
        lm_configuration = lm.config
        print("\n[INFO] lm_configuration:", lm_configuration, "\n")

    pad_idx = tokenizer.encode(
        "[PAD]", add_special_tokens=False, add_prefix_space=True
    )[0]
    mask_idx = tokenizer.encode(
        "[MASK]", add_special_tokens=False, add_prefix_space=True
    )[0]

    ## DEBUG only
    assert pad_idx == 0
    assert mask_idx == 28

    return lm, tokenizer, pad_idx, mask_idx


def get_pred_head(args, c_in):
    ### TODO: hardcoded. will change later.
    if args.inverse_folding:
        print("[INFO] Using InverseFoldingPredictionHead.")
        pred_head = ProteinInverseFoldingPredictionHead(
            c_in=c_in,
            node_h_dim=128,
            embed_dim=args.embed_dim,
            num_blocks=args.num_blocks,
            num_attn_heads=8,
            attn_dropout=0.1,
            add_pos_enc_seq=True,
            add_pos_enc_str=False,
        )
    elif args.multihead_attn_predhead:
        raise NotImplementedError()
        # print('[INFO] Using MultiheadAttnMDLMHead.')
        # pred_head = MultiheadAttnMDLMHead(
        #     c_in=self.lm.config.hidden_size,
        #     embed_dim=args.embed_dim,
        #     num_blocks=args.num_blocks,
        #     num_attn_heads=8,
        #     attn_dropout=0.1,
        #     add_pos_enc_seq=True,
        # )
    else:
        raise NotImplementedError()
        # print('[INFO] Using original MLM head.')
        # pred_head = None
        # print('[INFO] Using VanillaMDLMHead.')
        # self.pred_head = VanillaMDLMHead(
        #     c_in=self.lm.config.hidden_size,
        #     embed_dim=head_embed_dim,
        #     dropout=0.1,
        #     add_pos_enc_seq=True,
        # )

    return pred_head


from dataclasses import dataclass


@dataclass
class ProtInvFoldModelConfig:
    init_params: str = None  # Path to the '.pt' file containing model weights
    # pretrained_LM_location: str = None
    seed: int = None
    embed_dim: int = 512
    num_blocks: int = 3
    drop_prob: float = 0.1
    add_str_enc_logits: bool = False
    lm_logit_update_weight: float = 1.0
    inverse_folding: bool = True
    multihead_attn_predhead: bool = False
    unconditional: bool = False

    test_only: bool = False

    # diffusion
    init_masking_rate: float = 1.0
    num_denoise_steps: int = 1
    num_full_seq_mask_epoch: int = 0
    sample_seq: bool = False
    sampling_temperature: float = 0.1

    lm_modality: str = "protein"
    lm = None

    # tokenizer_version: str = "v2"

    # class OptimizerConfig:
    # class_path: str = "torch.optim.AdamW"
    # lr: float = 1e-4
    # weight_decay: float = 0.01

    # # class OverfittingConfig:
    # overfitting_run: bool = False
    # overfit_on_sample_count: int = 50

    # for diffusion verbose
    diffusion_verbose: int = 1

    # class LoRAConfig:
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # # class SchedulerConfig:
    # lr_scheduler_total_iters: int = 500
    # lr_scheduler_end_factor: float = 0.1

    # # class DataloaderConfig:
    batch_size: int = 1
    # num_workers: int = 4
    # pin_memory: bool = True


default_config = ProtInvFoldModelConfig()
