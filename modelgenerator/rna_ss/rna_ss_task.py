from modelgenerator.tasks import *
from typing import Mapping, Any
from modelgenerator.rna_ss.rna_ss_utils import *
from collections import defaultdict
from modelgenerator.adapters.adapters import ResNet2DAdapter


class RNASSPairwiseTokenClassification(TaskInterface):
    """Task for fine-tuning a RNA-FM model on RNA Secondary Structure prediction.

    Args:
        backbone (BackboneCallable, optional): The callable that returns a backbone. Defaults to --model.backbone aido_dna_dummy.
        adapter (AdapterCallable, optional): The callable that returns an adapter. Defaults to LinearAdapter.
        optimizer (OptimizerCallable, optional): The optimizer to use for training. Defaults to torch.optim.AdamW.
        lr_scheduler (LRSchedulerCallable, optional): The learning rate scheduler to use for training. Defaults to None.
        batch_size (int, optional): The batch size to use for training. Defaults to None.
        use_legacy_adapter (bool, optional):
            Whether to use the adapter from the backbone. Defaults to False.
            Will be deprecated once the new adapter API is fully verified.
        strict_loading (bool, optional): Whether to strictly load the model. Defaults to True.
            Set it to False if you want to replace the adapter (e.g. for continue pretraining)
        reset_optimizer_states (bool, optional): Whether to reset the optimizer states. Defaults to False.
            Set it to True if you want to replace the adapter (e.g. for continue pretraining).
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/train_sec_struct_prediction.py

    def __init__(
        self,
        backbone_fn: BackboneCallable = aido_dna_dummy,
        adapter_fn: Optional[Callable[[int, int], SequenceAdapter]] = ResNet2DAdapter,
        tune_threshold_every_n_epoch: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.tune_threshold_every_n_epoch = tune_threshold_every_n_epoch
        self.backbone_fn = backbone_fn
        self.adapter_fn = adapter_fn
        self.backbone = None
        self.adapter = None
        self.loss = nn.BCEWithLogitsLoss()
        self._eval_step_outputs = None
        self.threshold = 0.5
        self.THRESHOLD_TUNE_METRIC = "f1"
        self.THRESHOLD_CANDIDATES = [i / 100 for i in range(1, 30, 1)]

    def configure_model(self) -> None:
        if self.backbone is not None:
            return
        if self.use_legacy_adapter:
            raise NotImplementedError()
        else:
            self.backbone = self.backbone_fn(None, None)
            self.adapter = self.adapter_fn(
                self.backbone.get_embedding_size() * 2,
            )

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: int
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data into a format that can be passed to the forward and evaluate methods.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data containing sample ids, sequences, and secondary structures
            batch_idx (int): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch containing sample ids, sequences, secondary structures (labels), input_ids, and attention_mask
        """
        tokenized_result = self.backbone.tokenize(batch["sequences"])
        input_ids = tokenized_result.pop("input_ids", None)
        attention_mask = tokenized_result.pop("attention_mask", None)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
        labels = batch["sec_structures"].half().to(self.device)
        return {
            "ss_ids": batch["ss_ids"],
            "sequences": batch["sequences"],
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **tokenized_result,
        }

    def outer_concat(self, x):
        """Taken directly from FM4BioContactHead"""
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

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask

        Returns:
            Tensor: The classifier logits
        """
        encoder_hidden = self.backbone(**collated_batch)
        if self.use_legacy_adapter:
            raise NotImplementedError()
        else:
            x = self.outer_concat(encoder_hidden)
            cls_logits = self.adapter(x[:, 1:-1, 1:-1])
        return cls_logits

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The sequence-level model predictions
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing labels
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss. Defaults to False.

        Returns:
            Tensor: The loss.
        """
        labels = collated_batch["labels"]
        seq_len = labels.shape[1]
        upper_tri_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits.device),
            diagonal=1,
        )
        loss = self.loss(
            logits[..., upper_tri_mask], labels[..., upper_tri_mask].to(logits)
        )
        if loss_only:
            return {"loss": loss}
        metrics = self.get_metrics_by_stage(stage)
        preds = logits.argmax(-1)
        for metric in metrics.values():
            metric(preds, labels)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            trainable_param = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            self.log("TrPara", trainable_param, prog_bar=True)
        return super().training_step(batch, batch_idx)

    def on_validation_start(self):
        self._reset_eval_step_outputs()

    def validation_step(self, batch, batch_idx):
        return self._eval_step(
            batch, batch_idx, thresholds=self.THRESHOLD_CANDIDATES, log_prefix="val"
        )

    def on_validation_epoch_end(self):
        if (
            self.trainer.sanity_checking
            or (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch != 0
        ):
            # If sanity checking, don't update the threshold
            return

        # Find threshold with highest validation score
        best_metric_val = -1.0
        best_threshold = 0.0
        for threshold in self.THRESHOLD_CANDIDATES:
            curr_metric_val = sum(
                self._eval_step_outputs[self.THRESHOLD_TUNE_METRIC][threshold]
            ) / len(self._eval_step_outputs[self.THRESHOLD_TUNE_METRIC][threshold])

            if curr_metric_val > best_metric_val:
                best_metric_val = curr_metric_val
                best_threshold = threshold

        self.threshold = best_threshold

        self.log(f"val_f1", round(best_metric_val, 3), prog_bar=True, sync_dist=True)
        self.log(f"thresh", self.threshold, prog_bar=True, sync_dist=True)

        self.log_dict(
            {
                f"val/{self.THRESHOLD_TUNE_METRIC}": best_metric_val,
                f"val/threshold": self.threshold,
            },
            sync_dist=True,
        )

    def on_test_start(self):
        self._reset_eval_step_outputs()

    def test_step(self, batch, batch_idx):
        return self._eval_step(
            batch, batch_idx, thresholds=[self.threshold], log_prefix="test"
        )

    def on_test_epoch_end(self):
        # Get macro average of each metric
        for key in self._eval_step_outputs:
            metric_avg_val = sum(self._eval_step_outputs[key][self.threshold]) / len(
                self._eval_step_outputs[key][self.threshold]
            )
            self.log(f"test/{key.lower()}", metric_avg_val)

    def _update_eval_step_outputs(
        self, logits, sec_struct_true, ss_ids, seqs, thresholds
    ):
        batch_size, *_ = logits.shape

        probs = torch.sigmoid(logits)

        if probs.dtype == torch.bfloat16:
            # Cast brain floating point into floating point
            probs = probs.type(torch.float16)

        probs = probs.cpu().numpy()
        sec_struct_true = sec_struct_true.cpu().numpy()

        for i in range(batch_size):
            for threshold in thresholds:
                sec_struct_pred = prob_mat_to_sec_struct(
                    probs=probs[i], seq=seqs[i], threshold=threshold
                )

                y_true = sec_struct_true[i]
                y_pred = sec_struct_pred

                self._eval_step_outputs["precision"][threshold].append(
                    ss_precision(y_true, y_pred)
                )
                self._eval_step_outputs["recall"][threshold].append(
                    ss_recall(y_true, y_pred)
                )
                self._eval_step_outputs["f1"][threshold].append(ss_f1(y_true, y_pred))

            if self.trainer.testing:
                output_dir = Path(self.trainer.default_root_dir)
                f1_score = self._eval_step_outputs["f1"][threshold][-1]

                save_to_ct(
                    output_dir / f"{ss_ids[i]}_pred_f1={f1_score}.ct",
                    sec_struct=y_pred,
                    seq=seqs[i],
                )

    def _reset_eval_step_outputs(self):
        self._eval_step_outputs = defaultdict(lambda: defaultdict(list))

    def _eval_step(self, batch, batch_idx, thresholds, log_prefix="eval"):
        collated_batch = self.transform(batch, batch_idx)
        ss_ids = collated_batch["ss_ids"]
        seqs = collated_batch["sequences"]
        sec_struct_true = collated_batch["labels"]
        logits = self(collated_batch)

        if (
            self.trainer.testing
            or (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch == 0
        ):
            self._update_eval_step_outputs(
                logits, sec_struct_true, ss_ids, seqs, thresholds
            )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = dict(state_dict)
        self.threshold = state_dict["threshold"]
        state_dict.pop(
            "threshold"
        )  # Remove 'threshold' key for possible "strict" clashes

        return super().load_state_dict(state_dict, strict, assign)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"]["threshold"] = self.threshold
        return super().on_save_checkpoint(checkpoint)
