#### inv folding
from modelgenerator.tasks import *
from modelgenerator.backbones import aido_protein_16b
from .extra_utils import *
from typing import Mapping, Any
import os


class ProteinInvFold(TaskInterface):
    def __init__(
        self,
        backbone: BackboneCallable = aido_protein_16b,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = LinearLR,
        batch_size: Optional[int] = None,
        use_legacy_adapter: bool = False,
        strict_loading: bool = True,
        reset_optimizer_states: bool = False,
        proteinmpnn_ckpt_path: str = None,
        custom_invfold_config: ProtInvFoldModelConfig = default_config,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__(**kwargs)

        ## internal and task-specific customization/packaging. does not conflict with the overall ModelGenerator structure.
        self.args = custom_invfold_config
        self.args.batch_size = (
            1  # self.batch_size   # inference always with batch_size=1
        )
        self.proteinmpnn_ckpt_path = proteinmpnn_ckpt_path
        self.backbone_fn = backbone

        self.loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
        self.accuracies = []
        self.true_sequences = []
        self.predictions = []
        self.accuracies_str_enc = []
        self.acc_metric = MyAccuracy()

    def configure_model(self) -> None:
        self.lm = self.backbone_fn(None, None)
        self.tokenizer = self.lm.tokenizer
        self.pred_head = get_pred_head(self.args, self.lm.get_embedding_size())
        self.lm = self.lm.encoder

        if True:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules="all-linear",
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                inference_mode=False,
            )
            self.lm = get_peft_model(self.lm, lora_config)
            print("[INFO]: L0RA lm.print_trainable_parameters():")
            print(self.lm.print_trainable_parameters())

        self.pad_idx = self.tokenizer.encode(
            "[PAD]", add_special_tokens=False, add_prefix_space=True
        )[0]
        self.MASK_TOKEN = self.mask_idx = self.tokenizer.encode(
            "[MASK]", add_special_tokens=False, add_prefix_space=True
        )[0]

        self.structure_encoder = ProteinMPNNCMLM(
            protein_mpnn_args["encoder"], self.proteinmpnn_ckpt_path
        )  # Adapted from https://github.com/BytedProtein/ByProt/tree/main

        ## DEBUG only
        assert self.pad_idx == 0
        assert self.mask_idx == 28

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = dict(state_dict)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key == "lm.base_model.model.output_embed.weight":
                continue
            else:
                new_state_dict[key.replace(".bert", "")] = value

        return super().load_state_dict(new_state_dict, strict=strict, assign=assign)

    def load_pretrained_lm_weights(self, pretrained_weights_path):
        self.lm.load_state_dict(torch.load(pretrained_weights_path))

    def forward(self, data):
        assert (data["input_ids_mpnn"] > 21).sum() == 0, str(
            (data["input_ids_mpnn"] > 21)
        )
        encoder_logits, encoder_out = self.structure_encoder(batch=data)
        data["str_enc_logits"] = encoder_logits
        data["structure_encoding"] = encoder_out["feats"]

        init_pred = data["str_enc_logits"].argmax(-1).detach()
        data["input_ids"][:, :-1] = torch.where(
            data["input_mask"].bool(), init_pred, data["input_ids"][:, :-1]
        )

        outputs = self.lm(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            output_hidden_states=True,
        )

        if self.args.inverse_folding:
            x = outputs.hidden_states[-1]
            x = x[:, :-1, :]
            x[data["pad_mask"], :] = 0.0
            data["positional_encoding"][data["pad_mask"], :] = 0.0
            data["lm_representation"] = x
            preds = self.pred_head(data)  # (batch_size X len X 44)

        else:
            raise NotImplementedError()

        return preds

    def compute_loss(self, pred_logits, data):
        loss = self.loss(
            pred_logits.permute(0, 2, 1).contiguous(), data["target_ids"][:, :-1]
        )
        loss = (
            loss * data["target_mask"]
        )  ## NOTE: CE loss for each token, masked by 'target_mask'.
        loss = loss.sum(-1) * data["posterior_weight"]  ## NOTE: Ref: MDLM, Equation 11
        loss = loss.sum() / data["target_mask"].sum()

        return loss

    def compute_and_record_accuracy(self, pred_tokens, data):
        acc = ((pred_tokens == data["target_ids"][:, :-1]) * data["target_mask"]).sum(
            -1
        ) / data["target_mask"].sum(-1)
        # acc = acc * data['posterior_weight']
        self.acc_metric.update(acc.detach().sum(), batch_size=data["batch_size"])
        acc = acc.detach().cpu().numpy().tolist()
        self.accuracies += acc

        self.true_sequences += [
            (data["target_ids"][..., :-1].float() * data["coord_mask"].float())
            .long()
            .detach()
            .cpu()
            .numpy()
        ]
        self.predictions += [pred_tokens.detach().cpu().numpy()]
        self.entry_names += [data["entry_name"]]

        acc_grnade = (
            (data["str_enc_logits"].argmax(-1).detach() == data["target_ids"][:, :-1])
            * data["target_mask"]
        ).sum(-1) / data["target_mask"].sum(-1)
        # acc_grnade = acc_grnade * data['posterior_weight']
        acc_grnade = acc_grnade.detach().cpu().numpy().tolist()
        self.accuracies_str_enc += acc_grnade

    def inference_step(self, batch, batch_idx, log_prefix: str):

        data = self._get_data_from_batch(batch=batch)
        assert (data["target_ids"] > 20).sum() == 0, str(
            (data["target_ids"] > 20).sum()
        )

        if (log_prefix in {"test", "val"}) and (self.args.num_denoise_steps > 1):
            preds, pred_tokens = self._run_reverse_diffusion(data=data)
        else:
            preds = self(data)  ## (B x N)
            pred_tokens = preds.argmax(-1).detach()

        # print(pred_tokens.shape)

        loss = self.compute_loss(pred_logits=preds, data=data)
        loss_str = self.compute_loss(pred_logits=data["str_enc_logits"], data=data)

        LM_loss_wt = 0.5
        loss = loss * LM_loss_wt + loss_str * (1 - LM_loss_wt)

        self.compute_and_record_accuracy(pred_tokens=pred_tokens, data=data)

        # with torch.no_grad():
        #     loss.clamp_(min=1e-3)

        log = {
            f"{log_prefix}/loss": loss,
        }
        # print(log)

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
                round(np.mean(self.accuracies_str_enc), 3),
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def _get_data_from_batch(self, batch):
        (
            seq_input,
            seq_target,
            input_mask_indices,
            positional_encoding,
            input_mask,
            target_mask,
            pad_mask,
            seq_length,
            posterior_weight,
            masking_rate,
            interval_partition,
            interval_idx,
            entry_name,
            coords,
            coord_mask,
        ) = batch

        posterior_weight = posterior_weight / posterior_weight.sum()
        posterior_weight = posterior_weight.view(-1)

        raw_input = self.tokenizer(seq_input, padding=True)
        raw_target = self.tokenizer(seq_target, padding=True)

        slen = seq_length.max()
        batch_size = seq_length.shape[0]

        data = {
            "max_seq_length_in_batch": slen,
            "seq_input": seq_input,
            "seq_target": seq_target,
            "input_ids": torch.tensor(raw_input["input_ids"]).long().to(coords.device),
            "attention_mask": torch.tensor(raw_input["attention_mask"])
            .long()
            .to(coords.device),
            "target_ids": torch.tensor(raw_target["input_ids"])
            .long()
            .to(coords.device),
            "positional_encoding": positional_encoding[:, :slen].float(),
            "input_mask": input_mask[:, :slen].float(),
            "target_mask": target_mask[:, :slen].float(),
            "pad_mask": pad_mask[:, :slen].bool(),
            "seq_length_all": seq_length.long(),
            "need_attn_weights": True,
            "batch_size": batch_size,
            "posterior_weight": posterior_weight,
            "coords": coords[:, :slen].float(),
            "coord_mask": coord_mask[:, :slen].bool(),
            "entry_name": entry_name,
        }
        data["input_ids_mpnn"] = torch.ones_like(data["input_ids"])
        data["target_mask"] = torch.where(
            data["target_ids"][:, :-1] > 20, 0, data["target_mask"]
        )
        data["target_mask"] = torch.where(
            data["target_ids"][:, :-1] == 0, 0, data["target_mask"]
        )
        data["target_ids"] = torch.where(data["target_ids"] > 20, 0, data["target_ids"])

        return data

    def _run_reverse_diffusion(self, data):
        assert (
            data["input_ids"].shape[0] == 1
        ), f'Testing only accepts batch_size=1 per GPU (for now). Got input_ids with shape {data["input_ids"].shape}.'
        assert (
            data["input_mask"].sum() > 0
        ), f"data['input_mask'].sum() cannot be zero. at least one token has to be masked for diffusion."
        assert (data["input_ids"] == self.MASK_TOKEN).sum() == data["input_mask"].sum()

        if self.args.diffusion_verbose == 1:
            acc_temp = (
                (data["input_ids"] == data["target_ids"])[:, :-1] * data["target_mask"]
            ).sum(-1) / data["target_mask"].sum(-1)
            print("\n\n\tinput seq accuracy:", acc_temp.detach().cpu().item())

        unmask_count = int(
            max(
                1,
                np.ceil(
                    data["input_mask"].sum().detach().cpu().item()
                    / self.args.num_denoise_steps
                ),
            )
        )

        ##@ Run reverse process of diffusion
        denoise_step = 0
        mapLM2MPNN = torch.tensor(
            [
                0,
                12,
                2,
                9,
                21,
                17,
                8,
                3,
                18,
                11,
                5,
                16,
                13,
                7,
                4,
                15,
                20,
                14,
                10,
                19,
                6,
                1,
            ]
        ).to(data["input_ids_mpnn"])
        while True:
            ##@ Run inference
            preds = self(data).detach()  # 1 x (N,)
            assert (
                preds.shape[1] == data["max_seq_length_in_batch"]
            ), f'{[preds.shape[1], data["max_seq_length_in_batch"]]}'

            ##@ Add bias to the logits if we want to sample from a joint distribution (w/ and w/o LM).
            if self.args.add_str_enc_logits:
                assert 0 <= self.args.lm_logit_update_weight <= 1
                lm_w = self.args.lm_logit_update_weight
                preds = lm_w * preds + (1 - lm_w) * data["str_enc_logits"]

            ##@ Update probabilites: anything other than the mask tokens are set to 0, so that they won't be chosen.
            probs = F.softmax(preds, dim=-1).max(-1).values  # 1 x N
            probs = probs * data["input_mask"]
            unmask_indices = probs[0].argsort()[
                -unmask_count:
            ]  ## NOTE: These indices were predicted with the highest probabilities. So will be unmasked.

            ##@ Sampling and modifying input for new reverse diffusion step
            if self.args.sample_seq:
                probs_t = F.softmax(
                    preds[0] / self.args.sampling_temperature, dim=-1
                )  ## NOTE: higher sampling_temperature ==> more random sequence.
                pred_token_candidates = torch.multinomial(probs_t, 1, replacement=True)[
                    unmask_indices, 0
                ]
            else:
                pred_token_candidates = preds.argmax(-1)[0, unmask_indices]

            ##@ Use the modified input tokens as the predicted tokens. Only the masked tokens are unmasked, The original tokens are not modified.
            data["input_ids"][:, :-1][
                0, unmask_indices
            ] = pred_token_candidates  ## NOTE: Predicted and input tokens come from different vocabs. Make sure to map them properly.

            data["input_mask"][0, unmask_indices] = 0 
            data["input_ids_mpnn"] = data["input_ids"].clone().detach()
            data["input_ids_mpnn"] = torch.where(
                data["input_ids_mpnn"] > 20, 21, data["input_ids_mpnn"]
            )
            data["input_ids_mpnn"] = mapLM2MPNN[data["input_ids_mpnn"]]

            pred_tokens = data["input_ids"][:, :-1]

            acc_temp = (
                (pred_tokens == data["target_ids"][:, :-1]) * data["target_mask"]
            ).sum(-1) / data["target_mask"].sum(-1)

            denoise_step += 1

            if self.args.diffusion_verbose == 1:
                print(
                    "\tdenoise_step:",
                    denoise_step,
                    "\tacc_temp:",
                    acc_temp.detach().cpu().item(),
                    f"\t{(data['input_ids'] == self.MASK_TOKEN).sum()}, {data['input_mask'].sum()}",
                )

            ##@ Check if any masked tokens need to be unmasked. otherwise break out of loop.
            if (data["input_mask"].sum() == 0) or (
                denoise_step >= self.args.num_denoise_steps
            ):
                break

        return preds, pred_tokens

    def _eval_step(self, batch, batch_idx, log_prefix):
        return self.inference_step(batch, batch_idx, log_prefix=log_prefix)

    def _on_eval_epoch_start(self):
        # Reset metric calculator
        self.accuracies = []
        self.true_sequences = []
        self.predictions = []
        self.entry_names = []
        self.accuracies_str_enc = []
        self.acc_metric.reset()

    def _on_eval_epoch_end(self, log_prefix: str):

        # Log and reset metric calculator
        if not self.trainer.sanity_checking:
            acc = self.acc_metric.compute()
            self.log(f"{log_prefix}/acc", np.median(self.accuracies), sync_dist=True)

            tokens = "_LAGVSERTIDPKQNFYMHWC-"
            true_strings = [
                "".join([tokens[x] for x in l[0]]) for l in self.true_sequences
            ]
            pred_strings = [
                "".join([tokens[x] for x in l[0]]) for l in self.predictions
            ]
            if self.trainer.testing or log_prefix == "test":
                os.makedirs("proteinIF_outputs", exist_ok=True)
                pickle.dump(
                    {"true_seq": self.true_sequences, "pred_seq": self.predictions},
                    open("proteinIF_outputs/designed_sequences.pkl", "wb"),
                )
            _acc = []
            _len = []
            writer_str = ""
            for i, l in enumerate(true_strings):
                non_zero = (self.true_sequences[i][0] != 0).astype(np.float32)
                rec_acc = (
                    (self.true_sequences[i][0] == self.predictions[i][0]) * non_zero
                ).sum() / non_zero.sum()
                _acc += [rec_acc]
                _len += [non_zero.sum()]
                writer_str += f">name={self.entry_names[i][0]} | L={len(l)} | Recovery={rec_acc}\ntrue:{l}\npred:{pred_strings[i]}\n\n"
            _acc = np.array(_acc)
            _len = np.array(_len)
            print("Avg acc:", np.mean(_acc))
            print("Median acc:", np.median(_acc))
            _acc = (_acc * _len).sum() / _len.sum()
            print("Weighted avg acc:", _acc)

            if self.trainer.testing or log_prefix == "test":
                open(f"proteinIF_outputs/results_acc_{_acc}.txt", "w").write(
                    writer_str
                )

            self.accuracies = []
            self.true_sequences = []
            self.predictions = []
            self.accuracies_str_enc = []
            self.entry_names = []
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

        try:
            return self.inference_step(batch, batch_idx, log_prefix="train")
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            print("Skipped batch due to OOM", flush=True)
            for p in self.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            return None

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
        if self.pred_head:
            norms = grad_norm(self.pred_head, norm_type=2)
            if "grad_2.0_norm_total" in norms:
                self.log(f"grad_norm", norms["grad_2.0_norm_total"], prog_bar=True)

    def on_fit_start(self):
        if self.trainer.testing:
            self.args.test_only = True
