import torch
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
import warnings

import argparse
from pathlib import Path

from .data_inverse_folding.datamodule import (
    RNAInverseFoldingDataModule,
    RNAInvFoldConfig,
    default_config,
)
from .data_inverse_folding.dataset import (
    mapping_tensor_grnade2lm,
    mapping_tensor_lm2grnade,
)
from .adapter import RNAInverseFoldingAdapter

from typing import Mapping, Any

from modelgenerator.tasks import *
from modelgenerator.backbones import *
from modelgenerator.huggingface_models.rnabert import RNABertTokenizer


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


class RNAInvFold(TaskInterface):
    def __init__(
        self,
        backbone: BackboneCallable = aido_rna_1b600m,
        strict_loading: bool = True,
        # grnade_ckpt_path: str = None,
        custom_invfold_config: RNAInvFoldConfig = default_config,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__(**kwargs)

        self.args = args = custom_invfold_config
        self.args.batch_size = (
            1  # self.batch_size   # inference always with batch_size=1
        )
        # self.grnade_ckpt_path = grnade_ckpt_path
        self.backbone_fn = backbone

        self.loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.05
        )  ## NOTE: "label_smoothing" taken from gRNAde codebase
        self.accuracies = []
        self.accuracies_grnade = []
        self.true_sequences = []
        self.pred_sequences = []
        self.baseline_sequences = []
        self.acc_metric = MyAccuracy()

        self.mapping_tensor_lm2grnade = mapping_tensor_lm2grnade.clone()
        self.mapping_tensor_grnade2lm = mapping_tensor_grnade2lm.clone()

        print(self)

    def configure_model(self) -> None:
        self.lm = self.backbone_fn(None, None)
        # self.tokenizer = self.lm.tokenizer

        repo_base_dir = Path(__file__).resolve().parent.parent.parent
        vocab_file = os.path.join(
            repo_base_dir, "modelgenerator/huggingface_models/rnabert/vocab.txt"
        )
        self.tokenizer = RNABertTokenizer(vocab_file, version="v1")
        self.pad_idx = self.tokenizer.encode(
            "[PAD]", add_special_tokens=False, add_prefix_space=True
        )[0]

        self.pred_head = RNAInverseFoldingAdapter(
            c_in=self.lm.get_embedding_size(),
            node_h_dim=(128, 16),
            embed_dim=self.args.embed_dim,
            num_blocks=self.args.num_blocks,
            num_attn_heads=8,
            attn_dropout=0.1,
            add_pos_enc_seq=True,
            add_pos_enc_str=False,
        )
        self.lm = self.lm.encoder

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = dict(state_dict)
        new_state_dict = {}
        for key, value in state_dict.items():
            if "scaler." in key:
                continue
            else:
                new_state_dict[key] = value
        return super().load_state_dict(new_state_dict, strict=strict, assign=assign)

    def load_pretrained_lm_weights(self, pretrained_weights_path):
        raise NotImplementedError()
        # self.lm.load_state_dict(torch.load(pretrained_weights_path))

    def forward(self, data):
        # x = self.lm(tokens)["representation"]
        x = self.lm(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"]
        ).last_hidden_state

        ## pad_mask will only mask the padding tokens
        x[data["pad_mask"], :] = 0.0
        data["positional_encoding"][data["pad_mask"], :] = 0.0

        data["lm_representation"] = x

        pred = self.pred_head(data)

        return pred

    def _common_step(self, batch, batch_idx, log_prefix: str):
        (
            seq_input,
            seq_target,
            target_ids,
            input_mask_indices,
            logits,
            structure_encoding,
            structure_encoding_vector,
            positional_encoding,
            input_mask,
            target_mask,
            pad_mask,
            seq_length,
            posterior_weight,
            masking_rate,
            interval_partition,
            interval_idx,
        ) = batch

        posterior_weight = (
            posterior_weight / posterior_weight.sum()
        )  ## TODO: experimental, normalizing loss weights for stable training?
        posterior_weight = posterior_weight.view(-1)

        raw_input = self.tokenizer(seq_input, padding=True)  # , truncation=True)
        # raw_target = self.tokenizer(seq_target, padding=True)#, truncation=True)
        slen = seq_length.max()
        batch_size = seq_length.shape[0]

        data = {
            "max_seq_length_in_batch": slen,
            "seq_input": seq_input,
            "seq_target": seq_target,
            "input_mask_indices": input_mask_indices,
            "input_ids": torch.tensor(raw_input["input_ids"])
            .long()
            .to(structure_encoding.device),
            "attention_mask": torch.tensor(raw_input["attention_mask"])
            .long()
            .to(structure_encoding.device),
            "target_ids": target_ids[
                :, :slen
            ].long(),  # torch.tensor(raw_target['input_ids']).long().to(structure_encoding.device),
            "structure_encoding": structure_encoding[:, : slen + 4].float(),
            "structure_encoding_vector": structure_encoding_vector[
                :, : slen + 4
            ].float(),
            "positional_encoding": positional_encoding[:, : slen + 4].float(),
            "str_enc_logits": logits[:, :slen].float(),
            "input_mask": input_mask[:, :slen].float(),
            "target_mask": target_mask[:, :slen].float(),
            "pad_mask": pad_mask[:, : slen + 4].bool(),
            "seq_length": seq_length.long(),
            "need_attn_weights": True,
        }

        target_ids = data["target_ids"]
        # target_ids = target_ids[data['target_mask']]
        # data['str_enc_logits'] = data['str_enc_logits'][data['target_mask'], :]

        preds = None
        if log_prefix in {"test", "val"}:
            # raise Exception('Bug!')
            ##@ gRNAde and LM vocab mappings, both way
            self.mapping_tensor_lm2grnade = self.mapping_tensor_lm2grnade.to(
                data["input_ids"]
            )
            self.mapping_tensor_grnade2lm = self.mapping_tensor_grnade2lm.to(
                data["input_ids"]
            )

            # assert data['input_ids'].shape[0] == 1, f'Testing only accepts batch_size=1 per GPU (for now). Got input_ids with shape {data["input_ids"].shape}.'
            # assert data['input_mask'].sum() > 0, f"data['input_mask'].sum() cannot be zero. at least one token has to be masked for diffusion."
            # assert (data['input_ids'] == 1).sum() == data['input_mask'].sum()

            pred_tokens = self.mapping_tensor_lm2grnade[data["input_ids"][:, 2:-2]]
            acc_temp = ((pred_tokens == target_ids) * data["target_mask"]).sum(
                -1
            ) / data["target_mask"].sum(-1)
            if self.args.diffusion_verbose == 1:
                print(
                    batch_idx,
                    ">\tMasked acc:",
                    acc_temp.detach().cpu().item(),
                    "#total=",
                    pred_tokens.shape,
                    "#mask=",
                    (pred_tokens == 4).sum(),
                )
                print("Seq:", data["seq_target"])

            unmask_count = int(
                max(
                    1,
                    np.ceil(
                        data["input_mask"].sum().detach().cpu().item()
                        / self.args.num_denoise_steps
                    ),
                )
            )

            # print(data['input_mask'].sum(), data['input_mask'].shape, unmask_count, self.args.num_denoise_steps, data['input_mask'].sum())

            ##@ Run reverse process of diffusion
            denoise_step = 0
            # for denoise_step in range(0, self.args.num_denoise_steps):
            while True:
                ##@ Check if any masked tokens need to be unmasked. otherwise break out of loop.
                if (data["input_mask"].sum() == 0) or (
                    denoise_step == self.args.num_denoise_steps
                ):
                    # print("No more mask tokens.")
                    break

                ##@ Run inference

                # print("\n\tdenoise_step:", denoise_step, unmask_count, data['input_mask'].sum(), (data['input_ids'] == 1).sum())
                preds = self(data).detach()  # 1 x (N+4)

                ##@ Add bias to the logits if we want to sample from a joint distribution (w/ and w/o LM). Ref: LM-Design, DPLM
                if self.args.add_str_enc_logits and False:
                    assert 0 <= self.args.lm_logit_update_weight <= 1
                    lm_w = self.args.lm_logit_update_weight
                    preds[:, 2:-2, :] = (
                        lm_w * preds[:, 2:-2, :] + (1 - lm_w) * data["str_enc_logits"]
                    )

                ##@ Update probabilites: anything other than the mask tokens are set to low value, so that they won't be chosen.
                probs = F.softmax(preds, dim=-1).max(-1).values[:, 2:-2]  # 1 x N
                probs = probs * data["input_mask"]
                # print('2 probs>', probs)
                unmask_indices = probs[0].argsort()[
                    -unmask_count:
                ]  ## NOTE: These indices were predicted with the highest probabilities. So will be unmasked.

                ##@ Modifiy input for new reverse diffusion step
                if False:
                    data["input_mask"][0, unmask_indices] = 0.0
                    data["input_ids"][:, 2:-2][0, unmask_indices] = (
                        self.mapping_tensor_grnade2lm[
                            preds.argmax(-1)[:, 2:-2][0, unmask_indices]
                        ]
                    )  ## NOTE: Predicted and input tokens come from different vocabs. Make sure to map them properly.
                else:
                    ##@ Sampling and modifying input for new reverse diffusion step
                    if self.args.sample_seq:
                        probs_t = F.softmax(
                            preds[0, 2:-2] / self.args.sampling_temperature, dim=-1
                        )  ## NOTE: higher sampling_temperature ==> more random sequence.
                        pred_token_candidates = torch.multinomial(
                            probs_t, 1, replacement=True
                        )[unmask_indices, 0]
                    else:
                        pred_token_candidates = preds.argmax(-1)[:, 2:-2][
                            0, unmask_indices
                        ]

                    ##@ Use the modified input tokens as the predicted tokens. Only the masked tokens are unmasked, The original tokens are not modified.
                    data["input_ids"][:, 2:-2][0, unmask_indices] = (
                        self.mapping_tensor_grnade2lm[pred_token_candidates]
                    )  ## NOTE: Predicted and input tokens come from different vocabs. Make sure to map them properly.
                    data["input_mask"] = (
                        data["input_ids"][:, 2:-2] == 1
                    ).float()  ## TODO: [MASK]=1, hardcoded. change later.

                    pred_tokens = self.mapping_tensor_lm2grnade[
                        data["input_ids"][:, 2:-2]
                    ]
                    acc_temp = ((pred_tokens == target_ids) * data["target_mask"]).sum(
                        -1
                    ) / data["target_mask"].sum(-1)
                    if self.args.diffusion_verbose == 1:
                        print(
                            "\tdenoise_step:",
                            denoise_step,
                            "\tacc_temp:",
                            acc_temp.detach().cpu().item(),
                        )
                    denoise_step += 1

            ##@ Use the modified input tokens as the predicted tokens. Only the masked tokens are unmasked, The original tokens are not modified.
            pred_tokens = self.mapping_tensor_lm2grnade[data["input_ids"][:, 2:-2]]
            preds = preds[:, 2:-2, :]

        elif log_prefix in {"train", "test", "val"}:
            preds = self(data)
            preds = preds[:, 2:-2, :]
            pred_tokens = preds.argmax(-1).detach()
        else:
            raise Exception("[BUG] Code should never reach here!")

        loss = (
            self.loss(preds.permute(0, 2, 1).contiguous(), target_ids)
            * data["target_mask"]
        )  ## NOTE: CE loss for each token, masked by 'input_mask' or 'target_mask'.
        loss = loss.sum(-1) * posterior_weight  ## NOTE: Ref: MDLM, Equation 11
        loss = loss.sum() / data["target_mask"].sum()

        # with torch.no_grad():
        #     loss.clamp_(min=1e-3)

        # acc = (((pred_tokens == target_ids) * data['target_mask']).sum(-1) / data['target_mask'].sum(-1))
        acc = ((pred_tokens == target_ids) * data["target_mask"]).sum(-1) / data[
            "target_mask"
        ].sum(-1)
        # acc = acc * posterior_weight
        self.acc_metric.update(acc.detach().sum(), batch_size=batch_size)
        acc = acc.detach().cpu().numpy().tolist()
        self.accuracies += acc

        bases = ["A", "G", "C", "U"]
        if self.trainer.testing:
            self.true_sequences += [
                "".join(
                    [bases[x] for x in target_ids[0].detach().cpu().numpy().tolist()]
                )
            ]
            self.pred_sequences += [
                "".join(
                    [bases[x] for x in pred_tokens[0].detach().cpu().numpy().tolist()]
                )
            ]
            self.baseline_sequences += [
                "".join(
                    [
                        bases[x]
                        for x in data["str_enc_logits"]
                        .argmax(-1)[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    ]
                )
            ]

        acc_grnade = (
            (data["str_enc_logits"].argmax(-1) == target_ids) * data["target_mask"]
        ).sum(-1) / data["target_mask"].sum(-1)
        # acc_grnade = acc_grnade * posterior_weight
        acc_grnade = acc_grnade.detach().cpu().numpy().tolist()
        self.accuracies_grnade += acc_grnade

        log = {
            f"{log_prefix}/loss": loss,
        }
        self.log_dict(log, sync_dist=True)

        if log_prefix in {"train"}:
            self.log(
                f"{log_prefix}_loss",
                round(loss.item(), 3),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{log_prefix}_acc",
                round(np.mean(self.accuracies), 3),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{log_prefix}_bacc",
                round(np.mean(self.accuracies_grnade), 3),
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def _eval_step(self, batch, batch_idx, log_prefix):
        return self._common_step(batch, batch_idx, log_prefix=log_prefix)

    def _on_eval_epoch_start(self):
        # Reset metric calculator
        self.accuracies = []
        self.accuracies_grnade = []
        self.true_sequences = []
        self.pred_sequences = []
        self.baseline_sequences = []
        self.acc_metric.reset()

    def _on_eval_epoch_end(self, log_prefix: str):

        # Log and reset metric calculator
        if not self.trainer.sanity_checking:
            acc = self.acc_metric.compute()
            self.log(f"{log_prefix}/acc", acc, sync_dist=True)
            self.log(
                f"{log_prefix}_acc", round(acc.item(), 3), prog_bar=True, sync_dist=True
            )
            print(
                f"\n{log_prefix}_acc: {acc.item()} \tself.acc_metric.total {self.acc_metric.total}"
            )
            print(
                f"\n{log_prefix}_acc2: {np.mean(self.accuracies)} \tself.acc_metric.total {len(self.accuracies)}"
            )
            print(
                f"\n{log_prefix}_bacc: {np.mean(self.accuracies_grnade)} \tself.acc_metric.total {len(self.accuracies_grnade)}"
            )
            # print()

            if self.trainer.testing:
                import pickle
                import json

                os.makedirs("rnaIF_outputs", exist_ok=True)
                json.dump(
                    {
                        "true_seq": self.true_sequences,
                        "pred_seq": self.pred_sequences,
                        "baseline_seq": self.baseline_sequences,
                    },
                    open("rnaIF_outputs/designed_sequences.json", "w"),
                    indent="\t",
                )

            self.accuracies = []
            self.accuracies_grnade = []
            self.true_sequences = []
            self.pred_sequences = []
            self.baseline_sequences = []
            self.acc_metric.reset()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            trainable_param = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            self.log("TrPara", trainable_param, prog_bar=True)
        if not self.trainer.testing:
            cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return self._common_step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="val")

    def on_validation_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_validation_epoch_end(self):
        if (self.current_epoch > self.args.num_full_seq_mask_epoch) and (
            self.trainer.datamodule.train_dataset.weighted_loss is False
        ):
            print(
                '\nPrevious "training weighted_loss" = ',
                self.trainer.datamodule.train_dataset.weighted_loss,
            )
            self.trainer.datamodule.train_dataset.weighted_loss = True
            print(
                'New "training weighted_loss" = ',
                self.trainer.datamodule.train_dataset.weighted_loss,
                "\n",
            )

        return self._on_eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="test")

    def on_test_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end("test")

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.pred_head, norm_type=2)
        if "grad_2.0_norm_total" in norms:
            self.log(f"grad_norm", norms["grad_2.0_norm_total"], prog_bar=True)

    def on_fit_start(self):
        if self.trainer.testing:
            self.args.test_only = True
