# tasks.py
import functools
from typing import Callable, Literal, Optional, Union
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

from lightning.pytorch.utilities import grad_norm

import torchmetrics as tm

from modelgenerator.backbones import (
    DefaultConfig,
    LegacyAdapterType,
)
from modelgenerator.adapters import (
    SequenceAdapter,
    TokenAdapter,
    ConditionalGenerationAdapter,
    FusionAdapter,
    LinearAdapter,
    LinearCLSAdapter,
    ConditionalLMAdapter,
    MMFusionTokenAdapter,
)
from modelgenerator.metrics import (
    TopLAcc,
    AUROC,
    AUPRC,
    SpearmanCorrCoef,
    PearsonCorrCoef,
    MeanSquaredError,
    MeanAbsoluteError,
)
from modelgenerator.tasks.base import *


def once_only(method):
    attr_name = f"__{method.__name__}_called"

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, attr_name, False):
            return
        setattr(self, attr_name, True)
        return method(self, *args, **kwargs)

    return wrapper


class MLM(TaskInterface):
    """Task for performing masked language modeling (MLM) with a pretrained backbone.
    Can be used to train from scratch or for domain adaptation.
    Uses the [MLMDataModule](./#modelgenerator.data.MLMDataModule).
    Evaluates in terms of reconstruction accuracy on all tokens and cross-entropy loss.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        use_legacy_adapter: bool = True,
        **kwargs,
    ):
        if not use_legacy_adapter:
            raise ValueError("MLM must use the adapter from the backbone.")
        super().__init__(use_legacy_adapter=True, **kwargs)
        if self.__class__ is MLM:
            self.save_hyperparameters()
        self.backbone = backbone(LegacyAdapterType.MASKED_LM, None)
        self.adapter = None
        self.loss = nn.CrossEntropyLoss()
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                {
                    "accuracy": tm.MeanMetric(),
                }
            )
        self.metrics_to_pbar = {"accuracy"}

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        self.adapter = self.backbone.get_decoder()

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing target_sequences
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with target_ids and backbone inputs
        """
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        target_ids = None
        if batch.get("target_sequences", None) is not None:
            target_ids = self.backbone.tokenize(batch["target_sequences"])["input_ids"]
            if torch.is_tensor(target_ids):
                target_ids = target_ids.to(dtype=torch.long, device=self.device)
            else:
                target_ids = torch.tensor(target_ids, dtype=torch.long, device=self.device)
        return {
            "target_ids": target_ids,
            **processed_batch,
        }

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data from transform

        Returns:
            Tensor: The decoder logits
        """
        backbone_inputs = {k: v for k, v in collated_batch.items() if k != "target_ids"}
        encoder_hidden = self.backbone(**backbone_inputs).last_hidden_state
        decoder_logits = self.adapter(encoder_hidden)
        return decoder_logits

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions from forward against the ground truth labels.

        Args:
            logits (Tensor): The token-wise predicted logits
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing target_ids
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            Tensor: The loss.
        """
        target_ids = collated_batch["target_ids"]
        loss = self.loss(logits.permute(0, 2, 1).contiguous(), target_ids)
        if loss_only:
            return {"loss": loss}
        preds = logits.argmax(-1)
        metrics = self.get_metrics_by_stage(stage)
        self.call_or_update_metric(stage, metrics["accuracy"], (preds == target_ids).float().mean())
        return {"loss": loss}

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            return []
        return ["target_sequences"]


class Inference(MLM):
    """Task for performing inference with a pretrained backbone end-to-end, including the backbone's original adapter.

    Note:
        Must be used with [PredictionWriter](../callbacks/#modelgenerator.callbacks.PredictionWriter).
        Model outputs are stored under "predictions".

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
    """

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences (and optionally ids)
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with input_ids, attention_mask, and target_ids
        """
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        batch.update(processed_batch)
        return batch


class SequenceClassification(TaskInterface):
    """Task for fine-tuning a sequence model for classification.
    Evaluates in terms of accuracy, F1 score, Matthews correlation coefficient (MCC), and AUROC.

    Note:
        Supports binary, multiclass, and binary multi-label classification tasks.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        adapter: A SequenceAdapter for the model.
        n_classes: The number of classes in the classification task.
        multilabel: Indicate whether multiple labels can be positive, turning this into a multi-way binary classification task.
    """

    legacy_adapter_type = LegacyAdapterType.SEQ_CLS

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[Callable[[int, int], SequenceAdapter]] = LinearCLSAdapter,
        n_classes: int = 2,
        multilabel: bool = False,
        weighted_loss: bool = False,
        **kwargs,
    ):
        if n_classes < 2:
            raise ValueError(
                "n_classes must be greater than 1. Set n_classes=2 for binary classification."
            )
        super().__init__(**kwargs)
        if self.__class__ is SequenceClassification:
            self.save_hyperparameters()
        if self.use_legacy_adapter:
            self.backbone = backbone(
                self.legacy_adapter_type,
                DefaultConfig(config_overwrites={"num_labels": self.n_classes}),
            )
        else:
            self.backbone = backbone(None, None)
        self.adapter = None
        self.adapter_fn = adapter
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.weighted_loss = weighted_loss
        if self.weighted_loss:
            # Check if data module has class_weight property
            if not hasattr(type(self.data_module), "class_weight"):
                raise ValueError(
                    "Data module does not have property 'class_weight', "
                    "which is required by weighted loss."
                )
        for stage in ["train", "val", "test"]:
            if not multilabel:
                task = "binary" if n_classes == 2 else "multiclass"
                self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                    {
                        # Note: `average` need to be set explicitly for accuracy and f1
                        # see https://github.com/Lightning-AI/torchmetrics/issues/2280
                        "accuracy": tm.Accuracy(task, num_classes=n_classes, average="micro"),
                        "f1": tm.F1Score(task, num_classes=n_classes, average="macro"),
                        "mcc": tm.MatthewsCorrCoef(task, num_classes=n_classes),
                        "auroc": tm.AUROC(task, num_classes=n_classes),
                    }
                )
            else:
                self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                    {
                        "accuracy": tm.Accuracy(
                            "multilabel", num_labels=n_classes, average="macro"
                        ),
                        "f1": tm.F1Score("multilabel", num_labels=n_classes, average="macro"),
                        "mcc": tm.MatthewsCorrCoef("multilabel", num_labels=n_classes),
                        "auroc": tm.AUROC("multilabel", num_labels=n_classes, average="macro"),
                    }
                )
                if stage == "test":
                    # calculates score for each label
                    label_wise_acc = nn.ModuleDict(
                        {"accuracy_" + str(i): tm.Accuracy("binary") for i in range(n_classes)}
                    )
                    label_wise_f1 = nn.ModuleDict(
                        {"f1_" + str(i): tm.F1Score("binary") for i in range(n_classes)}
                    )
                    label_wise_mcc = nn.ModuleDict(
                        {"mcc_" + str(i): tm.MatthewsCorrCoef("binary") for i in range(n_classes)}
                    )
                    label_wise_auroc = nn.ModuleDict(
                        {"auroc_" + str(i): tm.AUROC("binary") for i in range(n_classes)}
                    )
                    self.metrics[f"{stage}_metrics"].update(label_wise_acc)
                    self.metrics[f"{stage}_metrics"].update(label_wise_f1)
                    self.metrics[f"{stage}_metrics"].update(label_wise_mcc)
                    self.metrics[f"{stage}_metrics"].update(label_wise_auroc)

        self.metrics_to_pbar = set(self.metrics["train_metrics"].keys())

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        if not self.multilabel:
            # input: (bs, C), target: (bs,)
            self.loss = nn.CrossEntropyLoss(
                weight=self.data_module.class_weight if self.weighted_loss else None
            )
        else:
            # input: (bs, C), target: (bs, C)
            self.loss = nn.BCEWithLogitsLoss(
                weight=self.data_module.class_weight if self.weighted_loss else None
            )
        if self.use_legacy_adapter:
            self.adapter = self.backbone.get_decoder()
        else:
            self.adapter = self.adapter_fn(self.backbone.get_embedding_size(), self.n_classes)

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data into a format that can be passed to the forward and evaluate methods.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data containing sequences and labels
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch containing input_ids, attention_mask, and labels
        """

        processed_batch = self.backbone.process_batch(batch, device=self.device)

        if batch.get("labels") is not None:
            if torch.is_tensor(batch["labels"]):
                labels = batch["labels"].to(self.device, dtype=torch.long)
            else:
                labels = torch.tensor(batch["labels"]).to(self.device, dtype=torch.long)
        else:
            labels = None
        return {
            "labels": labels,
            **processed_batch,
        }

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask

        Returns:
            Tensor: The classifier logits
        """

        inputs = {k: v for k, v in collated_batch.items() if k != "labels"}
        outputs = self.backbone(**inputs)
        logits = self.adapter(outputs.last_hidden_state, outputs.attention_mask)
        return logits

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
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            Tensor: The loss.
        """

        labels = collated_batch["labels"]
        if not self.multilabel:
            labels = labels.view(-1)  # (bs,)
            loss = self.loss(logits.view(-1, self.n_classes), labels)
        else:
            # Note: BCEWithLogitsLoss requires the labels to be float instead of int
            # TODO: to float should behandled in collate
            loss = self.loss(logits, labels.float())
        if loss_only:
            return {"loss": loss}
        metrics = self.get_metrics_by_stage(stage)
        if not self.multilabel:
            preds = torch.softmax(logits, dim=-1)  # probs of shape (bs, C)
            if self.n_classes == 2:
                preds = preds[:, 1]  # probs of shape (bs,)
        else:
            preds = torch.sigmoid(logits)  # probs of shape (bs, C)
        if self.multilabel and stage == "test":
            for name, metric in metrics.items():
                if len(name.split("_")) == 1:
                    self.call_or_update_metric(stage, metric, preds, labels)
                else:
                    i = int(name.split("_")[-1])
                    self.call_or_update_metric(stage, metric, preds[:, i], labels[:, i])
        else:
            for metric in metrics.values():
                self.call_or_update_metric(stage, metric, preds, labels)

        return {"loss": loss}

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            return []
        return ["labels"]


class TokenClassification(SequenceClassification):
    """Task for fine-tuning a model for token-wise classification.
    Evaluates in terms of accuracy, F1 score, Matthews correlation coefficient (MCC), and AUROC.

    Args:
        adapter: A TokenAdapter for the model.
        **kwargs: Additional keyword arguments for the parent class. `multilabel=False` is always overridden.

    Attributes:
        legacy_adapter_type (LegacyAdapterType): The LegacyAdapterType.TOKEN_CLS legacy adapter.
    """

    legacy_adapter_type = LegacyAdapterType.TOKEN_CLS

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[Callable[[int, int], TokenAdapter]] = LinearAdapter,
        n_classes: int = 2,
        **kwargs,
    ):
        # TODO: multi-label can be supported once token classification dataset
        # supports it and padding values are handled correctly
        super().__init__(backbone, adapter=adapter, n_classes=n_classes, multilabel=False, **kwargs)
        if self.__class__ is TokenClassification:
            self.save_hyperparameters()

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask

        Returns:
            Tensor: The classifier logits
        """
        inputs = {k: v for k, v in collated_batch.items() if k != "labels"}
        outputs = self.backbone(**inputs)
        collated_batch["special_tokens_mask"] = outputs.special_tokens_mask
        logits = self.adapter(outputs.last_hidden_state, outputs.attention_mask)
        return logits

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The token-level model predictions
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing labels
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            Tensor: The loss.
        """
        padded_labels = collated_batch["labels"]
        collated_batch["labels"] = padded_labels[padded_labels != -100].view(-1)
        if collated_batch["special_tokens_mask"] is not None:
            special_tokens_mask = torch.logical_not(
                torch.as_tensor(collated_batch["special_tokens_mask"])
            )
            logits = logits[special_tokens_mask].view(-1, self.n_classes)
        else:
            logits = logits.view(-1, self.n_classes)
        return super().evaluate(logits, collated_batch, stage, loss_only)


class PairwiseTokenClassification(SequenceClassification):
    """Task for fine-tuning a model for pairwise token classification.
    Evaluates in terms of accuracy, F1 score, Matthews correlation coefficient (MCC), AUROC, and top-k accuracy for k=2,5,10.

    Attributes:
        legacy_adapter_type (LegacyAdapterType): The LegacyAdapterType.TOKEN_CLS legacy adapter.

    Args:
        adapter: A TokenAdapter for the model.
        adapter_dim_multiplier (int, optional): The multiplier for the adapter dimension.
        **kwargs: Additional keyword arguments for the parent class. `n_classes=2` and `multilabel=False` are always overridden.
    """

    legacy_adapter_type = LegacyAdapterType.TOKEN_CLS

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[Callable[[int, int], TokenAdapter]] = LinearAdapter,
        adapter_dim_multiplier: int = 2,
        **kwargs,
    ):
        """
        If the we use MLPAdapter and outer_concat, then adapter_dim_multiplier should be 2;
        if we use MLPAdapterWithoutOutConcat, then adapter_dim_multiplier should be 1.
        The outer_concat operation is very memory intensive.
        """
        self.adapter_dim_multiplier = adapter_dim_multiplier
        kwargs["n_classes"] = 2
        kwargs["multilabel"] = False
        super().__init__(backbone, adapter=adapter, **kwargs)
        if self.__class__ is PairwiseTokenClassification:
            self.save_hyperparameters()
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict({"top_L_acc": TopLAcc(k=1)})
            self.metrics[f"{stage}_metrics"].update(
                {f"top_L{k}_acc": TopLAcc(k=k) for k in [2, 5, 10]}
            )
        self.metrics_to_pbar = {"top_L5_acc"}

    @once_only
    def configure_model(self) -> None:
        self.loss = nn.BCEWithLogitsLoss(
            weight=self.data_module.class_weight if self.weighted_loss else None
        )
        self.backbone.setup()
        if self.use_legacy_adapter:
            self.adapter = self.backbone.get_decoder()
        else:
            self.adapter = self.adapter_fn(
                self.backbone.get_embedding_size() * self.adapter_dim_multiplier,
                self.n_classes,
            )

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask

        Returns:
            Tensor: The classifier logits
        """
        inputs = {k: v for k, v in collated_batch.items() if k != "labels"}
        hidden_states = self.backbone(**inputs).last_hidden_state

        if self.use_legacy_adapter:
            logits = self.adapter(hidden_states)
        elif self.adapter_dim_multiplier > 1:
            x = self.outer_concat(hidden_states)
            logits = self.adapter(x)
        else:
            logits = self.adapter(hidden_states)
        return logits

    def outer_concat(self, x):
        """Taken directly from FM4BioContactHead"""
        batch_size, seq_len, features = x.shape

        # Permute to [batch_size, features, seq_len]
        x = x.permute(0, 2, 1)

        # Introduce new dimensions for broadcasting
        x_1 = x[:, None, :, :, None]  # [batch_size, 1, features, seq_len, 1]
        x_2 = x[:, None, :, None, :]  # [batch_size, 1, features, 1, seq_len]

        # Repeat along new dimensions
        x_1 = x_1.repeat(1, 1, 1, 1, seq_len)  # [batch_size, 1, features, seq_len, seq_len]
        x_2 = x_2.repeat(1, 1, 1, seq_len, 1)  # [batch_size, 1, features, seq_len, seq_len]

        # Concatenate along the second dimension
        x = torch.cat((x_1, x_2), dim=1)  # [batch_size, 2, features, seq_len, seq_len]

        # Get lower triangular indices
        I, J = torch.tril_indices(seq_len, seq_len, -1)

        # Symmetrize
        x[:, :, :, I, J] = x[:, :, :, J, I]

        # Permute to desired shape and make contiguous
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # [batch_size, seq_len, seq_len, features, 2]

        # Reshape to combine the last two dimensions
        x = x.view(
            batch_size, seq_len, seq_len, features * 2
        )  # [batch_size, seq_len, seq_len, features * 2]

        return x

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
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            Tensor: The loss.
        """
        padded_labels = collated_batch["labels"]  # (1, seq_len-1, seq_len-1)
        labels = padded_labels[padded_labels != -100]  # vector: (seq_len-1) * (seq_len-1)
        if collated_batch["special_tokens_mask"] is not None:
            special_tokens_mask = torch.logical_not(
                torch.as_tensor(collated_batch["special_tokens_mask"]).to(logits.device)
            )
            # (bs, seq_len) -> (bs, seq_len, seq_len) by batch wise outer product
            special_tokens_mask_expanded = torch.einsum(
                "bp, bq -> bpq", special_tokens_mask, special_tokens_mask
            )
            logits = logits[special_tokens_mask_expanded]  # (labels.shape[0], n_classes)

        logits = logits[..., -1]  # Use the second column which represents P(class=1)
        labels_float = (
            labels.float()
        )  # Convert labels to float for compatibility with BCEWithLogitsLoss
        loss = self.loss(logits, labels_float)
        if loss_only:
            return {"loss": loss}
        metrics = self.get_metrics_by_stage(stage)
        indices = torch.argsort(-logits)
        L = special_tokens_mask.sum().item()
        for acc in metrics.values():
            self.call_or_update_metric(stage, acc, logits, labels, indices, L)
        return {"loss": loss}


class Diffusion(TaskInterface):
    """[Masked Diffusion Language Modeling](https://arxiv.org/abs/2406.07524) training and generation on sequences.
    Evaluates in terms of reconstruction accuracy on masked tokens and cross-entropy loss.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        adapter: A TokenAdapter for the model.
        sample_seq: Whether to sample tokens during denoising, instead of always using the most likely token.
        num_denoise_steps: Granularity of the denoising process. Less steps makes fewer forward passes and denoises more aggressively.
        sampling_temperature: The temperature for sampling tokens, if sample_seq is True.
        normalize_posterior_weights: Whether to normalize posterior weights. Experimental feature to help training stability.
        verbose: Print while denoising (warning: fun to watch).
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[Callable[[int, int], TokenAdapter]] = None,
        use_legacy_adapter: bool = True,
        sample_seq: bool = False,
        num_denoise_steps: int = 4,
        sampling_temperature: float = 1.0,
        normalize_posterior_weights: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(use_legacy_adapter=use_legacy_adapter, **kwargs)
        if self.__class__ is Diffusion:
            self.save_hyperparameters()
        if self.use_legacy_adapter:
            self.backbone = backbone(LegacyAdapterType.MASKED_LM, None)
        else:
            self.backbone = backbone(None, None)
        self.adapter = None
        self.adapter_fn = adapter
        self.sample_seq = sample_seq
        self.num_denoise_steps = num_denoise_steps
        self.sampling_temperature = sampling_temperature
        self.normalize_posterior_weights = normalize_posterior_weights
        self.verbose = verbose
        self.loss = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.05
        )  ## NOTE: "label_smoothing" taken from gRNAde codebase
        self.mask_id = None
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                {
                    "accuracy": tm.MeanMetric(),
                }
            )
        self.metrics_to_pbar = {"accuracy"}

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        if self.use_legacy_adapter:
            self.adapter = self.backbone.get_decoder()
        else:
            self.adapter = self.adapter_fn(
                self.backbone.get_embedding_size(), self.backbone.get_vocab_size()
            )
        self.mask_id = self.backbone.get_token_id("[MASK]")

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences, target_sequences, and posterior_weights
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with input_ids, input_seqs, input_masks, attention_mask, target_ids, target_masks, target_seqs, and posterior_weights
        """
        # Each sample in a batch is a list of noised sequences at various noise levels. Stack them for easy training.
        batch["sequences"] = [seq for seqs in batch["sequences"] for seq in seqs]
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        input_ids = processed_batch["input_ids"]
        input_mask = torch.where(input_ids == self.mask_id, 1, 0)
        target_seqs = None
        target_ids = None
        target_mask = None
        posterior_weights = None
        if batch.get("target_sequences", None) is not None:
            target_seqs = [seq for seqs in batch["target_sequences"] for seq in seqs]
            target_ids = self.backbone.tokenize(target_seqs)["input_ids"]
            target_ids = torch.tensor(target_ids, dtype=torch.long).to(self.device)
            target_mask = torch.where(target_ids == self.mask_id, 1, 0)
        if batch.get("posterior_weights", None) is not None:
            posterior_weights = torch.tensor(
                [weight for weights in batch["posterior_weights"] for weight in weights]
            ).to(self.device)
            if self.normalize_posterior_weights:
                # Experimental! Normalizing posterior weights for stable training
                posterior_weights = posterior_weights / posterior_weights.sum()
                posterior_weights = posterior_weights.view(-1)
        return {
            "input_masks": input_mask,
            "target_ids": target_ids,
            "target_masks": target_mask,
            "target_seqs": target_seqs,
            "posterior_weights": posterior_weights,
            **processed_batch,
        }

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data with input_ids and attention_mask

        Returns:
            Tensor: The decoder logits
        """
        inputs = {
            k: v
            for k, v in collated_batch.items()
            if k not in ["target_ids", "target_masks", "posterior_weights"]
        }
        outputs = self.backbone(**inputs)
        decoder_logits = self.adapter(outputs.last_hidden_state)
        return decoder_logits

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The token-wise predicted logits
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing target_ids, posterior_weights, input_masks, and target_masks
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            Tensor: The loss on tokens with input mask and not target mask
        """
        target_ids = collated_batch["target_ids"]
        posterior_weight = collated_batch["posterior_weights"]
        # Evaluate loss and accuracy on all tokens with input mask and not target mask. Ignore any samples with no input mask.
        eval_mask = collated_batch["input_masks"] * (1 - collated_batch["target_masks"])
        eval_mask_count = eval_mask.sum(-1)
        good_samples = eval_mask_count > 0
        if not good_samples.any():
            # If we have no samples to evaluate, return a loss of 0
            return {"loss": torch.tensor(0.0).to(self.device)}
        # Avoid division by zero. These samples will have zero loss, but be ignored in the final average over good samples.
        eval_mask_count[eval_mask_count == 0] = 1
        # Get loss only on [MASK] tokens, scaled by posterior_weight / total masks
        loss = self.loss(logits.permute(0, 2, 1).contiguous(), target_ids) * eval_mask
        # If we've unmasked everything, eval_mask will be all zeros. Make sure this isn't nan when we have no loss to evaluate.
        # Loss for each timestep is a posterior-weighted sum of avg-unmasked-token-loss
        loss = posterior_weight * loss.sum(-1) / eval_mask_count
        # Total loss is an average over both timesteps and samples in a batch (stacked together in collate)
        loss = loss.sum() / good_samples.sum()
        if loss_only:
            return {"loss": loss}
        pred_tokens = logits.argmax(-1).detach()
        acc = ((pred_tokens == target_ids) * eval_mask).sum(-1) / eval_mask_count
        avg_acc = acc.sum() / good_samples.sum()
        metrics = self.get_metrics_by_stage(stage)
        self.call_or_update_metric(stage, metrics["accuracy"], avg_acc.item())
        return {"loss": loss}

    def _iterative_denoise(
        self, collated_batch: dict[str, Union[list, Tensor]]
    ) -> tuple[dict[str, Union[list, Tensor]], float]:
        """Denoises input sequences iteratively by predicting tokens at masked positions and unmasking the highest probability tokens.

        Note:
            Denoises wherever there are masks, but only evaluates loss where we have no labeled target
            With num_denoise_steps == 1, this is equivalent to the one-step inference used for training
            The loss is the unweighted average of BCE losses for each masked token across all denoising step.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing input_ids, input_masks, and target_masks

        Returns:
            tuple[dict[str, Union[list, Tensor]], float]: The denoised batch with no masks and the loss from the denoising process
        """
        # Keep track of original target masks since we use target masks to specify loss evaluation
        target_masks_init = collated_batch["target_masks"].clone()
        # Do denoise on all input masks. Only evaluate loss where input is mask and target is not
        unmask_counts = collated_batch["input_masks"].sum(-1) // self.num_denoise_steps + 1
        # Run reverse process of diffusion
        denoise_step = 0
        denoise_loss = 0
        while True:
            if collated_batch["input_masks"].sum() == 0 or denoise_step == self.num_denoise_steps:
                # Check if we're finished denoising
                break
            # Predict tokens
            logits = self(collated_batch)
            probs = F.softmax(logits / self.sampling_temperature, dim=-1)
            if self.sample_seq:
                # Make flat on batch x seq length dim for sampling
                pred_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1, replacement=True
                ).squeeze(-1)
                # Reshape back to batch x seq length
                pred_tokens = pred_tokens.view(probs.size(0), -1)
            else:
                # Take maximum probability tokens
                pred_tokens = logits.argmax(-1).squeeze(-1)
            # Unmask highest probability unmask_count tokens from the masked entries
            # Unmasked tokens remain the same
            probs = probs * collated_batch["input_masks"].unsqueeze(-1)
            input_mask_new = collated_batch["input_masks"].clone()
            for i in range(len(collated_batch["input_ids"])):
                unmask_count = min(unmask_counts[i], collated_batch["input_masks"][i].sum().item())
                top_probs = probs[i].max(-1).values
                unmask_indices = top_probs.argsort(descending=True)[:unmask_count]
                collated_batch["input_ids"][i, unmask_indices] = pred_tokens[i, unmask_indices]
                input_mask_new[i, unmask_indices] = 0
                # Specify target masks for loss evaluation
                collated_batch["target_masks"][i] = torch.ones_like(
                    collated_batch["target_masks"][i]
                )
                collated_batch["target_masks"][i, unmask_indices] = 0
            # Still never evaluate loss where we have no labeled target, even though we denoise
            collated_batch["target_masks"] = collated_batch["target_masks"] * target_masks_init
            # Get metrics on unmasked tokens and update the mask to reflect unmasking
            loss = self.evaluate(logits, collated_batch, loss_only=True)
            collated_batch["input_masks"] = input_mask_new
            # Loss averaged over tokens before averaging over samples, so sum over iters should be propto sum of token-wise loss at unmasked tokens with logits from masks at each iter
            denoise_loss += loss["loss"]
            denoise_step += 1
            if self.verbose:

                def clean(s):
                    return (
                        s.replace("[MASK]", ".")
                        .replace("[CLS]", "")
                        .replace("[SEP]", "")
                        .replace("[PAD]", "")
                        .replace(" ", "")
                    )

                pred_strings = self.backbone.decode_tokens(collated_batch["input_ids"])
                pred_strings = [clean(s) for s in pred_strings]
                print(pred_strings)
        # Reset the target mask, since we used this during denoising to specify loss evaluation
        collated_batch["target_masks"] = target_masks_init
        return collated_batch, denoise_loss

    def _val_test_step(
        self,
        batch: dict[str, Union[list, Tensor]],
        split: str,
        batch_idx: Optional[int] = None,
    ) -> Tensor:
        """Runs a validation or test step on a batch of data.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            split (str): The split to run the step on (val or test)
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The loss from the validation or test step
        """
        # TODO: this function might need to be merged into evaluate as it now
        #       has a reason to take in "stage" as an argument.
        collated_batch = self.transform(batch, batch_idx)
        input_mask_init = collated_batch["input_masks"].clone()
        collated_batch, denoise_loss = self._iterative_denoise(collated_batch)
        eval_mask = input_mask_init * (1 - collated_batch["target_masks"])
        acc = ((collated_batch["input_ids"] == collated_batch["target_ids"]) * eval_mask).sum(
            -1
        ) / eval_mask.sum(-1)
        avg_acc = acc.mean().item()
        metrics = self.get_metrics_by_stage(split)
        metrics["accuracy"](avg_acc)
        return {"loss": denoise_loss}

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            # TODO: should not require target_sequences
            return ["sequences", "target_sequences"]
        return ["sequences", "target_sequences", "posterior_weights"]

    def provided_data_columns(self, stage):
        return ["sequences"]

    def validation_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> Tensor:
        """Runs a validation step on a batch of data.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The loss from the validation step
        """
        loss = self._val_test_step(batch, "val", batch_idx)
        self.log_loss_and_metrics(loss["loss"], "val")
        return loss

    def test_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> Tensor:
        """Runs a test step on a batch of data.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The loss from the test step
        """
        loss = self._val_test_step(batch, "test", batch_idx)
        self.log_loss_and_metrics(loss["loss"], "test")
        return loss

    def predict_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> Tensor:
        """Infers predictions from a batch of data. Calls collate and forward methods in order.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The model predictions
        """
        collated_batch = self.transform(batch, batch_idx)
        collated_batch, _ = self._iterative_denoise(collated_batch)
        pred_strings = self.backbone.decode_tokens(collated_batch["input_ids"])

        def clean(s):
            return (
                s.replace("[MASK]", ".")
                .replace("[CLS]", "")
                .replace("[SEP]", "")
                .replace("[PAD]", "")
                .replace(" ", "")
            )

        collated_batch.update(
            {
                "predictions": [clean(s) for s in pred_strings],
                "sequences": [clean(s) for s in collated_batch["target_seqs"]],
            }
        )
        return collated_batch

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Runs before each optimizer step to compute the 2-norm of the gradients.

        Note:
            If using mixed precision, the gradients are already unscaled here

        Args:
            optimizer (torch.optim.Optimizer): The optimizer
            optimizer_idx (int): The index of the optimizer
        """
        norms = grad_norm(self.adapter, norm_type=2)
        if "grad_2.0_norm_total" in norms:
            self.log("grad_norm", norms["grad_2.0_norm_total"], prog_bar=True)


class ConditionalMLM(TaskInterface):
    """Task for masked language modeling with extra condition inputs.
    Evaluates in terms of reconstruction accuracy on all tokens and cross-entropy loss.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        adapter: A ConditionalGenerationAdapter for the model.
        use_legacy_adapter: Whether to use the pre-trained legacy adapter within the conditional decoder.
        condition_dim: The dimension of the condition.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[
            Callable[[int, int, int, nn.Module], ConditionalGenerationAdapter]
        ] = ConditionalLMAdapter,
        use_legacy_adapter: bool = True,
        condition_dim: int = 1,
        **kwargs,
    ):
        super().__init__(use_legacy_adapter=use_legacy_adapter, **kwargs)
        if self.__class__ is ConditionalMLM:
            self.save_hyperparameters()
        if self.use_legacy_adapter:
            self.backbone = backbone(LegacyAdapterType.MASKED_LM, None)
        else:
            self.backbone = backbone(None, None)
        self.adapter = None
        self.adapter_fn = adapter
        self.condition_dim = condition_dim
        self.loss = nn.CrossEntropyLoss()
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                {
                    "accuracy": tm.MeanMetric(),
                }
            )
        self.metrics_to_pbar = {"accuracy"}

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        self.adapter = self.adapter_fn(
            self.backbone.get_embedding_size(),
            self.condition_dim,
            self.backbone.get_vocab_size(),
            self.backbone.get_decoder() if self.use_legacy_adapter else None,
        )
        self.mask_id = self.backbone.get_token_id("[MASK]")

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences, target_sequences, and labels
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with input_ids, attention_mask, target_ids, and labels
        """
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        target_ids = None
        if batch.get("target_sequences", None) is not None:
            target_ids = self.backbone.tokenize(batch["target_sequences"])["input_ids"]
            if torch.is_tensor(target_ids):
                target_ids = target_ids.to(detype=torch.long, device=self.device)
            else:
                target_ids = torch.tensor(target_ids, dtype=torch.long, device=self.device)
        labels = batch["labels"].type(self.dtype)
        if len(batch["labels"].shape) == 1:
            labels = labels.unsqueeze(-1)
        return {
            "target_ids": target_ids,
            "labels": labels,
            **processed_batch,
        }

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids, attention_mask, and labels.

        Returns:
            Tensor: The decoder logits
        """
        inputs = {k: v for k, v in collated_batch.items() if k not in ["labels", "target_ids"]}
        outputs = self.backbone(**inputs)
        logits = self.adapter(outputs.last_hidden_state, collated_batch["labels"])
        return logits

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions from forward against the ground truth labels.

        Args:
            logits (Tensor): The token-wise predicted logits
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing target_ids
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            Tensor: The loss.
        """
        target_ids = collated_batch["target_ids"]
        loss = self.loss(logits.permute(0, 2, 1).contiguous(), target_ids)
        if loss_only:
            return {"loss": loss}
        preds = logits.argmax(-1)
        metrics = self.get_metrics_by_stage(stage)
        self.call_or_update_metric(stage, metrics["accuracy"], (preds == target_ids).float().mean())
        return {"loss": loss}

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.
        Args:
            stage (str): The stage of training (train, val, test)

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            return ["labels"]
        return ["labels", "target_sequences"]


class ConditionalDiffusion(Diffusion):
    """Task for masked diffusion language modeling with extra condition inputs.
    Evaluates in terms of reconstruction accuracy on masked tokens and cross-entropy loss.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        adapter: A ConditionalGenerationAdapter for the model.
        use_legacy_adapter: Whether to use the pre-trained legacy adapter within the conditional decoder.
        condition_dim: The dimension of the condition.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[
            Callable[[int, int, int, nn.Module], ConditionalGenerationAdapter]
        ] = ConditionalLMAdapter,
        use_legacy_adapter: bool = True,
        condition_dim: int = 1,
        **kwargs,
    ):
        super().__init__(backbone, use_legacy_adapter=use_legacy_adapter, **kwargs)
        if self.__class__ is ConditionalDiffusion:
            self.save_hyperparameters()
        self.adapter = None
        self.adapter_fn = adapter
        self.condition_dim = condition_dim

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        self.adapter = self.adapter_fn(
            self.backbone.get_embedding_size(),
            self.condition_dim,
            self.backbone.get_vocab_size(),
            self.backbone.get_decoder() if self.use_legacy_adapter else None,
        )
        self.mask_id = self.backbone.get_token_id("[MASK]")

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences, target_sequences, posterior_weights, and labels
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with input_ids, input_masks, attention_mask, target_ids, target_masks, posterior_weights, and labels
        """
        collated_batch = super().transform(batch, batch_idx)
        labels = torch.cat(batch["labels"]).type(self.dtype)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(-1)
        collated_batch.update({"labels": labels})
        return collated_batch

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask

        Returns:
            Tensor: The decoder logits
        """
        encoder_hidden = self.backbone(**collated_batch).last_hidden_state
        logits = self.adapter(encoder_hidden, collated_batch["labels"])
        return logits

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        return super().required_data_columns(stage) + ["labels"]


class SequenceRegression(TaskInterface):
    """Task for fine-tuning a sequence model for single-/multi-task regression.
    Evaluates in terms of mean absolute error, mean squared error, R2 score, Pearson correlation, and Spearman correlation.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        adapter: A SequenceAdapter for the model.
        num_outputs: The number of outputs for the regression task.
        loss_func: The loss function to use for training.
        log_grad_norm_step: The step interval for logging gradient norms.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[Callable[[int, int], SequenceAdapter]] = LinearCLSAdapter,
        num_outputs: int = 1,
        loss_func: Callable[..., torch.nn.Module] = torch.nn.MSELoss,
        log_grad_norm_step: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.__class__ is SequenceRegression:
            self.save_hyperparameters()
        self.adapter_fn = adapter
        self.num_outputs = num_outputs
        if self.use_legacy_adapter:
            self.backbone = backbone(
                LegacyAdapterType.SEQ_CLS,
                DefaultConfig(
                    config_overwrites={
                        "problem_type": "regression",
                        "num_labels": self.num_outputs,
                    }
                ),
            )
        else:
            self.backbone = backbone(None, None)
        self.adapter = None
        self.loss = loss_func()
        self.log_grad_norm_step = log_grad_norm_step
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                {
                    "pearson": PearsonCorrCoef(
                        num_outputs=num_outputs, multioutput="uniform_average"
                    ),
                    "spearman": SpearmanCorrCoef(
                        num_outputs=num_outputs, multioutput="uniform_average"
                    ),
                    "mae": MeanAbsoluteError(
                        num_outputs=num_outputs, multioutput="uniform_average"
                    ),
                    "r2": tm.R2Score(multioutput="uniform_average"),
                    "mse": MeanSquaredError(num_outputs=num_outputs, multioutput="uniform_average"),
                }
            )
            if stage == "test" and self.num_outputs > 1:
                # calculate scores for each task
                label_wise_spearman = nn.ModuleDict(
                    {
                        "spearman_" + str(i): SpearmanCorrCoef(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                label_wise_pearson = nn.ModuleDict(
                    {
                        "pearson_" + str(i): PearsonCorrCoef(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                label_wise_r2 = nn.ModuleDict(
                    {"r2_" + str(i): tm.R2Score() for i in range(self.num_outputs)}
                )
                label_wise_mse = nn.ModuleDict(
                    {
                        "mse_" + str(i): MeanSquaredError(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                label_wise_mae = nn.ModuleDict(
                    {
                        "mae_" + str(i): MeanAbsoluteError(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                self.metrics[f"{stage}_metrics"].update(label_wise_spearman)
                self.metrics[f"{stage}_metrics"].update(label_wise_pearson)
                self.metrics[f"{stage}_metrics"].update(label_wise_r2)
                self.metrics[f"{stage}_metrics"].update(label_wise_mse)
                self.metrics[f"{stage}_metrics"].update(label_wise_mae)
        self.metrics_to_pbar = set(self.metrics["train_metrics"].keys())

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        if self.use_legacy_adapter:
            self.adapter = self.backbone.get_decoder()
        else:
            self.adapter = self.adapter_fn(self.backbone.get_embedding_size(), self.num_outputs)

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data into a format that can be passed to the forward and evaluate methods.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data containing sequences and labels
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch containing sequences, input_ids, attention_mask, and labels
        """
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        labels = None
        if batch.get("labels") is not None:
            labels = batch["labels"].to(self.device, dtype=self.dtype)
        return {
            "labels": labels,
            **processed_batch,
        }

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask.

        Returns:
            Tensor: The regression predictions
        """

        inputs = {k: v for k, v in collated_batch.items() if k != "labels"}
        outputs = self.backbone(**inputs)  # (bs, seq_len, dim)
        preds = self.adapter(outputs.last_hidden_state, outputs.attention_mask)
        return preds

    def evaluate(
        self,
        preds: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The model predictions
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing labels
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            dict[str, Union[Tensor, float]]: A dictionary of metrics containing loss and mse
        """

        labels = collated_batch["labels"]
        loss = self.loss(preds, labels)
        if loss_only:
            return {"loss": loss}
        metrics = self.get_metrics_by_stage(stage)

        if self.num_outputs > 1 and stage == "test":
            for name, metric in metrics.items():
                if len(name.split("_")) == 1:
                    self.call_or_update_metric(stage, metric, preds, labels)
                else:
                    i = int(name.split("_")[-1])
                    self.call_or_update_metric(stage, metric, preds[:, i], labels[:, i])
        else:
            for metric in metrics.values():
                self.call_or_update_metric(stage, metric, preds, labels)

        return {"loss": loss}

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            return []
        return ["labels"]

    def log_grad_norm(self, optimizer):
        """
        Log the total total_norm, adaptor_param_norm and adaptor_grad_norm.

        Refer to
        https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/plugins/precision/precision.py
        https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/core/module.py
        for the calculation of the gradient norm
        """
        parameters = self.trainer.precision_plugin.main_params(optimizer)
        parameters = list(parameters)
        if len(parameters) > 0:
            assert all([p.requires_grad for p in parameters])
            if all([p.grad is not None for p in parameters]):
                total_norm = vector_norm(
                    torch.stack([vector_norm(p.grad, ord=2) for p in parameters]), ord=2
                )
                adaptor_param_norm = vector_norm(
                    torch.stack([vector_norm(p, ord=2) for p in self.adapter.parameters()]),
                    ord=2,
                )
                adaptor_grad_norm = vector_norm(
                    torch.stack([vector_norm(p.grad, ord=2) for p in self.adapter.parameters()]),
                    ord=2,
                )

                self.log("total_norm", total_norm, rank_zero_only=True)
                self.log("adaptor_param_norm", adaptor_param_norm, rank_zero_only=True)
                self.log("adaptor_grad_norm", adaptor_grad_norm, rank_zero_only=True)

    def on_before_optimizer_step(self, optimizer):
        """
        Log gradient norm of adaptor's parameters
        """
        if self.log_grad_norm_step > 0 and self.trainer.global_step % self.log_grad_norm_step == 0:
            self.log_grad_norm(optimizer)


class SequenceRegressionWithScaling(SequenceRegression):
    """Task for fine-tuning a sequence model on a regression task with scaling, where the label is scaled with dynamically adjusted mean and standard derivation.
    Evaluates in terms of mean absolute error, mean squared error, R2 score, Pearson correlation, and Spearman correlation.

    Note:
        Does not tolerate legacy adapters.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        adapter: Optional[Callable[[int, int], SequenceAdapter]] = LinearCLSAdapter,
        num_outputs: int = 1,
        loss_func: Callable[..., torch.nn.Module] = torch.nn.MSELoss,
        log_grad_norm_step: int = 0,
        **kwargs,
    ):
        super().__init__(
            backbone,
            adapter=adapter,
            num_outputs=num_outputs,
            loss_func=loss_func,
            log_grad_norm_step=log_grad_norm_step,
            **kwargs,
        )
        if self.__class__ is SequenceRegressionWithScaling:
            self.save_hyperparameters()
        self.adapter_fn = adapter
        self.num_outputs = num_outputs
        self.loss = nn.MSELoss()
        self.scaler = (
            self.StandardScaler()
        )  ## Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/train_ribosome_loading.py
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                {
                    "pearson": tm.PearsonCorrCoef(num_outputs=num_outputs),
                    "spearman": tm.SpearmanCorrCoef(num_outputs=num_outputs),
                    "mae": tm.MeanAbsoluteError(num_outputs=num_outputs),
                    "r2": tm.R2Score(),
                    "mse": tm.MeanSquaredError(num_outputs=num_outputs),
                }
            )
        self.metrics_to_pbar = set(self.metrics["train_metrics"].keys())

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask.

        Returns:
            Tensor: The regression predictions
        """
        ## Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/train_ribosome_loading.py

        inputs = {k: v for k, v in collated_batch.items() if k != "labels"}
        outputs = self.backbone(**inputs)  # (bs, seq_len, dim)
        hidden_states = outputs.last_hidden_state

        # Nullify padding token representations
        if outputs.attention_mask is not None:
            padding_mask = ~outputs.attention_mask
        else:
            padding_mask = torch.zeros(
                hidden_states.shape[:-1], dtype=torch.bool, device=hidden_states.device
            )
        hidden_states[padding_mask, :] = 0.0
        hidden_states = hidden_states[:, 1:-1, :]
        padding_mask = padding_mask[:, 1:-1]

        preds = self.adapter(hidden_states, padding_mask)
        return preds

    def evaluate(
        self,
        preds: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The model predictions
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing labels
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            dict[str, Union[Tensor, float]]: A dictionary of metrics containing loss and mse
        """
        labels = collated_batch["labels"]
        scaled_labels = self.scaler.transform(labels)  # "Scale" labels
        loss = self.loss(preds, scaled_labels)

        preds = self.scaler.inverse_transform(preds).clamp(min=0.0)  # "Unscale" predictions

        if loss_only:
            return {"loss": loss}
        metrics = self.get_metrics_by_stage(stage)
        for metric in metrics.values():
            self.call_or_update_metric(stage, metric, preds, labels)
        return {"loss": loss}

    def training_step(self, collated_batch, batch_idx):
        if batch_idx == 0:
            trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self.log("TrPara", trainable_param, prog_bar=True)
        if self.current_epoch == 0:
            return self.scaler.partial_fit(collated_batch["labels"])

        return super().training_step(collated_batch, batch_idx)

    class StandardScaler(nn.Module):
        # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/utils/scaler.py
        # Inspired by https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        def __init__(self) -> None:
            super().__init__()

            self.register_buffer("_mean", torch.tensor([0.0]))
            self.register_buffer("_std", torch.tensor([1.0]))

            self._seen_samples = []

            self._need_update = False

        def _update_mean_and_std(self) -> None:
            self._mean[0] = np.mean(self._seen_samples)
            self._std[0] = np.std(self._seen_samples)

        def partial_fit(self, x: torch.Tensor) -> None:
            self._need_update = True
            self._seen_samples.extend(x.cpu().view(-1).tolist())

            self._update_mean_and_std()

        def transform(self, x: torch.Tensor) -> torch.Tensor:
            return (x - self._mean) / self._std

        def inverse_transform(self, scaled_x: torch.Tensor) -> torch.Tensor:
            return scaled_x * self._std + self._mean


class Embed(TaskInterface):
    """Task for getting embeddings from a pretrained backbone. This task is used only for inference.

    Note:
        Must be used with [PredictionWriter](../callbacks/#modelgenerator.callbacks.PredictionWriter).
        Embeddings are stored under "predictions".

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        **kwargs: Additional keyword arguments for the parent class.
            `use_legacy_adapter=False` is always overridden.
    """

    def __init__(self, backbone: BackboneCallable, **kwargs):
        super().__init__(use_legacy_adapter=False, **kwargs)
        if self.__class__ is Embed:
            self.save_hyperparameters()
        self.backbone = backbone(None, None)
        self.adapter = None

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with sequences, input_ids, and attention_mask
        """
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        return processed_batch

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data from collate containing input_ids and attention_mask.

        Returns:
            Tensor: The decoder logits
        """
        encoder_hidden = self.backbone(**collated_batch)  # (bs, seq_len, dim)
        return encoder_hidden

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        return []


class ZeroshotPredictionDiff(TaskInterface):
    """Task for zero-shot prediction with a languange model that produces token logits.
       Computes the log-likelihood difference between probability of ref and alt at the mutated position.
       Evaluates in terms of AUROC and AUPRC.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        **kwargs: Additional keyword arguments for the parent class.
            `use_legacy_adapter=True` is always overridden.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        **kwargs,
    ):
        if self.__class__ is ZeroshotPredictionDiff:
            self.save_hyperparameters()
        super().__init__(use_legacy_adapter=True, **kwargs)
        self.backbone = backbone(LegacyAdapterType.MASKED_LM, None)
        self.adapter = None
        self.metrics["test_metrics"] = nn.ModuleDict({"AUROC": AUROC(), "AUPRC": AUPRC()})
        self.metrics_to_pbar = {"AUROC", "AUPRC"}

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        self.adapter = self.backbone.get_decoder()

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences and target_sequences
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with input_ids, attention_mask, and target_ids
        """
        processed_batch = self.backbone.process_batch(batch, device=self.device)
        labels = None
        ref_ids = None
        mutation_ids = None
        if batch.get("labels") is not None:
            labels = batch["labels"]
        if batch.get("refs") is not None:
            ref_ids = self.backbone.tokenize(batch["refs"], add_special_tokens=False)["input_ids"]
            ref_ids = torch.tensor(ref_ids, dtype=torch.long, device=self.device)
        if batch.get("mutations") is not None:
            mutation_ids = self.backbone.tokenize(batch["mutations"], add_special_tokens=False)[
                "input_ids"
            ]
            mutation_ids = torch.tensor(mutation_ids, dtype=torch.long, device=self.device)
        return {
            "ref_ids": ref_ids,
            "mutation_ids": mutation_ids,
            "labels": labels,
            **processed_batch,
        }

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data from collate

        Returns:
            Tensor: The decoder logits
        """
        inputs = {
            k: v
            for k, v in collated_batch.items()
            if k not in ["labels", "ref_ids", "mutation_ids"]
        }
        outputs = self.backbone(**inputs)
        decoder_logits = self.adapter(outputs.last_hidden_state)
        b, l, d = decoder_logits.shape
        if outputs.special_tokens_mask is not None:
            # remove special token before computing zeroshot score
            special_tokens_mask = torch.tensor(outputs.special_tokens_mask)
            decoder_logits = decoder_logits[torch.logical_not(special_tokens_mask)].view(b, -1, d)
        else:
            decoder_logits = decoder_logits.view(b, -1, d)
        return decoder_logits

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions from forward against the ground truth labels.

        Args:
            logits (Tensor): The token-wise predicted logits
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing ref_ids, mutation_ids
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss.

        """
        outputs = {}
        ref_ids = collated_batch["ref_ids"]
        mutation_ids = collated_batch["mutation_ids"]
        snp_location = logits.shape[1] // 2
        probs = F.softmax(logits, dim=2)
        snp_probs = probs[:, snp_location, :]

        ref_probs = torch.gather(snp_probs, 1, ref_ids)
        mutate_probs = torch.gather(snp_probs, 1, mutation_ids)

        log_ratios = torch.log(mutate_probs / ref_probs)
        metrics = self.get_metrics_by_stage(stage)
        for acc in metrics.values():
            self.call_or_update_metric(stage, acc, log_ratios.view(-1), collated_batch["labels"])
        outputs["score"] = log_ratios.view(-1).cpu().tolist()
        outputs["label"] = collated_batch["labels"].cpu().tolist()
        outputs["loss"] = -1
        return outputs

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            return []
        return ["labels", "refs", "mutations"]


class ZeroshotPredictionDistance(TaskInterface):
    """Task for zero-shot prediction with a model that produces embeddings.
       Computes the L1 and L2 distance between the reference and mutated sequence embeddings.
       Evaluates in terms of AUROC and AUPRC of the embedding distance.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        all_hidden_states: Whether to run the test on all available hidden layers.
        shared_ref: Whether to use a shared reference sequence to accelerate zero-shot computation. Uses a separate reference sequence for each mutated sequence by default.
        **kwargs: Additional keyword arguments for the parent class.
            `use_legacy_adapter=False` is always overridden.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        all_hidden_states: bool = False,
        shared_ref: bool = False,
        **kwargs,
    ):
        if self.__class__ is ZeroshotPredictionDistance:
            self.save_hyperparameters()
        super().__init__(use_legacy_adapter=False, **kwargs)
        self.backbone = backbone(None, None)
        self.adapter = None
        self.ref_hidden_mean = None
        self.all_hidden_states = all_hidden_states
        self.shared_ref = shared_ref

    @once_only
    def configure_model(self) -> None:
        self.backbone.setup()
        self.adapter = self.backbone.get_decoder()  # CE: Why does this require an adapter
        self.n_layers = self.backbone.get_num_layer() if self.all_hidden_states else 1
        metrics_dict = {}
        for i in range(self.n_layers):
            metrics_dict.update(
                {
                    f"L1_AUROC_layer_{i + 1}": AUROC(),
                    f"L1_AUPRC_layer_{i + 1}": AUPRC(),
                    f"L2_AUROC_layer_{i + 1}": AUROC(),
                    f"L2_AUPRC_layer_{i + 1}": AUPRC(),
                }
            )
        self.metrics["test_metrics"] = nn.ModuleDict(metrics_dict)
        self.metrics_to_pbar = set()
        for i in range(self.n_layers):
            self.metrics_to_pbar.update(
                {
                    f"L1_AUROC_layer_{i + 1}",
                    f"L1_AUPRC_layer_{i + 1}",
                    f"L2_AUROC_layer_{i + 1}",
                    f"L2_AUPRC_layer_{i + 1}",
                }
            )

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data for forward and evaluate.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader containing sequences and target_sequences
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch with ref_ids,mutation_id and labels
        """
        collated_batch = {}
        for key in batch.keys():
            if "sequences" in key:  # e.g. "mutation_sequences", "ref_sequences"
                # Note: previous version, add_special_token = False
                processed_batch = self.backbone.process_batch(
                    {"sequences": batch[key]}, device=self.device
                )
                for k, v in processed_batch.items():
                    collated_batch[key.replace("sequences", k)] = v
            else:
                collated_batch[key] = batch[key]
        return collated_batch

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data from collate

        Returns:
            Tensor: The decoder logits
        """
        inputs = {
            k.replace("mutation_", ""): v
            for k, v in collated_batch.items()
            if k.startswith("mutation_")
        }
        output = self.backbone(**inputs, all_hidden_states=self.all_hidden_states)

        if not self.all_hidden_states:
            mutation_encoder_hidden = output.last_hidden_state.unsqueeze(0)
        else:
            mutation_encoder_hidden = torch.stack(output.hidden_states)
        n, b, s, d = mutation_encoder_hidden.shape

        if self.ref_hidden_mean is not None:
            ref_encoder_hidden = self.ref_hidden_mean.unsqueeze(1).repeat(1, b, 1)
        else:
            ref_inputs = {
                k.replace("ref_", ""): v for k, v in collated_batch.items() if k.startswith("ref_")
            }
            ref_output = self.backbone(**ref_inputs, all_hidden_states=self.all_hidden_states)
            if not self.all_hidden_states:
                ref_encoder_hidden = ref_output.last_hidden_state.unsqueeze(0)
            else:
                ref_encoder_hidden = torch.stack(ref_output.hidden_states)
            if ref_output.special_tokens_mask is not None:
                ref_encoder_hidden = (
                    ref_encoder_hidden[:, torch.logical_not(ref_output.special_tokens_mask)]
                    .view(n, b, -1, d)
                    .mean(dim=-2)
                )
            else:
                ref_encoder_hidden = ref_encoder_hidden.mean(dim=-2)
            if self.shared_ref:
                self.ref_hidden_mean = ref_encoder_hidden[:, 0, :]

        # remove special token before computing zeroshot score
        if output.special_tokens_mask is not None:
            masked_hidden_list = []
            for i in range(b):
                mask = torch.logical_not(output.special_tokens_mask[i])
                masked_hidden = mutation_encoder_hidden[:, i, mask, :].mean(dim=1)
                masked_hidden_list.append(masked_hidden)
            mutation_encoder_hidden = torch.stack(masked_hidden_list, dim=1)
        else:
            mutation_encoder_hidden = mutation_encoder_hidden.mean(dim=-2)
        return torch.stack([ref_encoder_hidden, mutation_encoder_hidden])

    def evaluate(
        self,
        logits: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions from forward against the ground truth labels.

        Args:
            logits (Tensor): list of ref_hidden_states and mutation_hidden_states
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing ref_ids, mutation_ids
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss.

        """
        ref_hidden_states = logits[0]
        mutation_hidden_states = logits[1]
        prediction_dict = {
            key: []
            for key in collated_batch.keys()
            if ("mask" not in key) and ("input_id" not in key)
        }
        # add score related keys
        prediction_dict["norm_type"] = []
        prediction_dict["score"] = []
        prediction_dict["num_layer"] = []
        metrics = self.get_metrics_by_stage(stage)
        batch_size = ref_hidden_states.shape[1]
        # update metrics of each layer embedding
        for key in metrics.keys():
            index = int(key.split("_")[-1]) - 1
            norm_type = key.split("_")[0]
            score = self._compute_norm_score(
                norm_type, ref_hidden_states[index], mutation_hidden_states[index]
            )
            self.call_or_update_metric(stage, metrics[key], score, collated_batch["labels"])
        # prepare prediction score to be saved to a tsv file
        for index in range(ref_hidden_states.shape[0]):
            for norm_type in ["L1", "L2", "cosine"]:
                score = self._compute_norm_score(
                    norm_type, ref_hidden_states[index], mutation_hidden_states[index]
                )
                prediction_dict["score"].extend(score.cpu().tolist())
                prediction_dict["norm_type"].extend([norm_type] * batch_size)
                prediction_dict["num_layer"].extend([index] * batch_size)
                for key in collated_batch.keys():
                    if ("mask" not in key) and ("input_id" not in key):
                        if torch.is_tensor(collated_batch[key]):
                            prediction_dict[key].extend(collated_batch[key].cpu().tolist())
                        else:
                            prediction_dict[key].extend(collated_batch[key])
        prediction_dict.update({"loss": -1})
        outputs = prediction_dict
        return outputs

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        if stage == "predict":
            return []
        return ["labels", "ref_sequences", "mutation_sequences"]

    def provided_data_columns(self, stage):
        return ["sequences"]

    def _compute_norm_score(self, norm_type, ref_hidden_state, mutation_hidden_state):
        """Compute norm score between reference and mutation embeddings from one layer

        Args:
            norm_type (str): norm type. Options are 'L1' and 'L2'
            ref_hidden_state (Tensor): Reference sequence embeddings of one layer
            mutation_hidden_state (Tensor): Variant sequence embeddings of one layer
        Returns:
            score (Tensor): norm distance score

        """
        if norm_type == "L1":
            score = torch.abs(ref_hidden_state - mutation_hidden_state).sum(dim=1)
        elif norm_type == "L2":
            score = torch.norm(ref_hidden_state - mutation_hidden_state, p=2, dim=1)
        else:
            score = 1 - F.cosine_similarity(ref_hidden_state, mutation_hidden_state, dim=1)
        return score


class MMSequenceRegression(TaskInterface):
    """Task for fine-tuning multiple models on single-/multi-task regression.
    Evaluates in terms of mean absolute error, mean squared error, R2 score, Pearson correlation, and Spearman correlation.

    Note:
        Supports any combination of DNA, RNA and protein backbones
        Requires data column names to be prefixed by `backbone_order` list.
        For example, if `backbone_order` is ["dna_seq", "rna_seq"], the sequence columns should be named "dna_seq_sequences" and "rna_seq_sequences".
        Please use `rename_cols` in the data module to rename the columns if necessary.

    Args:
        backbone: A pretrained backbone from the modelgenerator library.
        backbone1: A second pretrained backbone from the modelgenerator library.
        backbone2: An optional third pretrained backbone from the modelgenerator library.
        backbone_order: A list of data columns in order of the backbones.
        adapter: A callable that returns a FusionAdapter.
        num_outputs: The number of outputs for the regression task.
        loss_func: A callable that returns a loss function.
    """

    def __init__(
        self,
        backbone: BackboneCallable,
        backbone1: BackboneCallable,
        backbone2: Optional[BackboneCallable] = None,
        backbone_order: list = ["dna_seq", "rna_seq"],
        adapter: Optional[Callable[[int, int, int, int], FusionAdapter]] = MMFusionTokenAdapter,
        num_outputs: int = 1,
        loss_func: Callable[..., torch.nn.Module] = torch.nn.MSELoss,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.__class__ is MMSequenceRegression:
            self.save_hyperparameters()
        self.num_backbones = 2 + int(backbone2 is not None)
        self.backbone_order = backbone_order
        self.backbone = backbone(None, None)
        self.backbone1 = backbone1(None, None)
        self.backbone2 = backbone2(None, None) if backbone2 is not None else None
        self.adapter_fn = adapter
        self.adapter = None
        self.num_outputs = num_outputs
        self.loss = loss_func()
        for stage in ["train", "val", "test"]:
            self.metrics[f"{stage}_metrics"] = nn.ModuleDict(
                {
                    "pearson": PearsonCorrCoef(
                        num_outputs=num_outputs, multioutput="uniform_average"
                    ),
                    "spearman": SpearmanCorrCoef(
                        num_outputs=num_outputs, multioutput="uniform_average"
                    ),
                    "mae": MeanAbsoluteError(
                        num_outputs=num_outputs, multioutput="uniform_average"
                    ),
                    "r2": tm.R2Score(multioutput="uniform_average"),
                    "mse": MeanSquaredError(num_outputs=num_outputs, multioutput="uniform_average"),
                }
            )
            if stage == "test" and self.num_outputs > 1:
                # calculate scores for each task
                label_wise_spearman = nn.ModuleDict(
                    {
                        "spearman_" + str(i): SpearmanCorrCoef(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                label_wise_pearson = nn.ModuleDict(
                    {
                        "pearson_" + str(i): PearsonCorrCoef(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                label_wise_r2 = nn.ModuleDict(
                    {"r2_" + str(i): tm.R2Score() for i in range(self.num_outputs)}
                )
                label_wise_mse = nn.ModuleDict(
                    {
                        "mse_" + str(i): MeanSquaredError(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                label_wise_mae = nn.ModuleDict(
                    {
                        "mae_" + str(i): MeanAbsoluteError(num_outputs=1)
                        for i in range(self.num_outputs)
                    }
                )
                self.metrics[f"{stage}_metrics"].update(label_wise_spearman)
                self.metrics[f"{stage}_metrics"].update(label_wise_pearson)
                self.metrics[f"{stage}_metrics"].update(label_wise_r2)
                self.metrics[f"{stage}_metrics"].update(label_wise_mse)
                self.metrics[f"{stage}_metrics"].update(label_wise_mae)
        self.metrics_to_pbar = set(self.metrics["train_metrics"].keys())

    @once_only
    def configure_model(self) -> None:
        # backbones
        self.backbone.setup()
        self.backbone1.setup()
        # fusion adapter
        if self.backbone2 is not None:
            self.backbone2.setup()
            context_input_size_2 = self.backbone2.get_embedding_size()
        else:
            context_input_size_2 = None

        self.adapter = self.adapter_fn(
            self.num_outputs,
            self.backbone.get_embedding_size(),
            self.backbone1.get_embedding_size(),
            context_input_size_2,
        )

    def on_save_checkpoint(self, checkpoint: dict):
        if hasattr(self.backbone, "on_save_checkpoint"):
            self.backbone.on_save_checkpoint(checkpoint)
        if hasattr(self.backbone1, "on_save_checkpoint"):
            self.backbone1.on_save_checkpoint(checkpoint, prefix="backbone1")
        if self.backbone2 is not None and hasattr(self.backbone2, "on_save_checkpoint"):
            self.backbone2.on_save_checkpoint(checkpoint, prefix="backbone2")

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data into a format that can be passed to the forward and evaluate methods.
        Note: for empty sequence, the input is "".

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data containing sequences and labels
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch containing sequences, input_ids, attention_mask, and labels
        """
        labels = None
        if batch.get("labels") is not None:
            if torch.is_tensor(batch["labels"]):
                labels = batch["labels"].to(self.device, dtype=self.dtype)
            else:
                labels = torch.tensor(batch["labels"], dtype=self.dtype, device=self.device)
        collated_batch = {"labels": labels}
        omic_name = self.backbone_order[0] + "_"
        omic_name1 = self.backbone_order[1] + "_"
        input_batches = {
            omic_name: {
                k.removeprefix(omic_name): v for k, v in batch.items() if k.startswith(omic_name)
            },
            omic_name1: {
                k.removeprefix(omic_name1): v for k, v in batch.items() if k.startswith(omic_name1)
            },
        }
        if self.num_backbones == 3:
            omic_name2 = self.backbone_order[2] + "_"
            input_batches[omic_name2] = {
                k.removeprefix(omic_name2): v for k, v in batch.items() if k.startswith(omic_name2)
            }
        for omic_name, backbone in zip(
            self.backbone_order, (self.backbone, self.backbone1, self.backbone2)
        ):
            omic_name = omic_name + "_"
            processed_batch = backbone.process_batch(input_batches[omic_name], device=self.device)
            for k, v in processed_batch.items():
                collated_batch[f"{omic_name}{k}"] = v
        return collated_batch

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask.

        Returns:
            Tensor: The regression predictions
        """
        omic_name = self.backbone_order[0] + "_"
        omic_name1 = self.backbone_order[1] + "_"
        backbone_input = {
            k.removeprefix(omic_name): v
            for k, v in collated_batch.items()
            if k.startswith(omic_name)
        }
        backbone1_input = {
            k.removeprefix(omic_name1): v
            for k, v in collated_batch.items()
            if k.startswith(omic_name1)
        }
        backbone_output = self.backbone(**backbone_input)
        backbone1_output = self.backbone1(**backbone1_input)
        if self.backbone2 is not None:
            omic_name2 = self.backbone_order[2] + "_"
            backbone2_input = {
                k.removeprefix(omic_name2): v
                for k, v in collated_batch.items()
                if k.startswith(omic_name2)
            }
            backbone2_output = self.backbone2(**backbone2_input)
            hidden_states2 = backbone2_output.last_hidden_state
            attention_mask2 = backbone2_output.attention_mask
        else:
            hidden_states2 = None
            attention_mask2 = None
        preds = self.adapter(
            backbone_output.last_hidden_state,
            backbone_output.attention_mask,
            backbone1_output.last_hidden_state,
            backbone1_output.attention_mask,
            hidden_states2,
            attention_mask2,
        )
        return preds

    def evaluate(
        self,
        preds: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The model predictions
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing labels
            loss_only (bool, optional): Whether to only return the loss.

        Returns:
            dict[str, Union[Tensor, float]]: A dictionary of metrics containing loss and mse
        """
        labels = collated_batch["labels"]
        loss = self.loss(preds, labels)
        if loss_only:
            return {"loss": loss}
        metrics = self.get_metrics_by_stage(stage)
        if self.num_outputs > 1 and stage == "test":
            for name, metric in metrics.items():
                if len(name.split("_")) == 1:
                    self.call_or_update_metric(stage, metric, preds, labels)
                else:
                    i = int(name.split("_")[-1])
                    self.call_or_update_metric(stage, metric, preds[:, i], labels[:, i])
        else:
            for metric in metrics.values():
                self.call_or_update_metric(stage, metric, preds, labels)
        return {"loss": loss}

    def required_data_columns(self, stage: str) -> list[str]:
        """The required data columns for the task.

        Returns:
            list[str]: The required data columns
        """
        columns = [f"{omic_name}_sequences" for omic_name in self.backbone_order]
        if stage != "predict":
            columns += ["labels"]
        return columns

    def provided_data_columns(self, stage):
        return ["sequences"]

    def check_data_compatibility(self, stage: str) -> None:
        """Checks if the data is compatible with the task."""
        if self.data_module is not None:
            data_provided_columns = set(self.data_module.provided_columns)
            task_provided_columns = set(self.provided_data_columns(stage))
            task_required_columns = set(self.required_data_columns(stage))
            backbone_required_columns = set(self.backbone.required_data_columns())
            backbone1_required_columns = set(self.backbone1.required_data_columns())
            backbone2_required_columns = set()
            if self.backbone2 is not None:
                backbone2_required_columns = set(self.backbone2.required_data_columns())
            error_msgs = []
            # Task gets data from data modules
            if not task_required_columns.issubset(data_provided_columns):
                error_msgs.append(
                    f"Task is missing required columns: {task_required_columns - data_provided_columns}. "
                    f"Provided columns: {data_provided_columns}. "
                    f"Required columns: {task_required_columns}. "
                )
            # Backbone gets data from both data module and task
            provided_columns = data_provided_columns.union(task_provided_columns)
            if not backbone_required_columns.issubset(provided_columns):
                error_msgs.append(
                    f"Backbone is missing required columns: {backbone_required_columns - provided_columns}. "
                    f"Data Provided columns: {data_provided_columns}. "
                    f"Task Provided columns: {task_provided_columns}. "
                    f"Backbone Required columns: {backbone_required_columns}. "
                )
            if not backbone1_required_columns.issubset(provided_columns):
                error_msgs.append(
                    f"Backbone1 is missing required columns: {backbone1_required_columns - provided_columns}. "
                    f"Data Provided columns: {data_provided_columns}. "
                    f"Task Provided columns: {task_provided_columns}. "
                    f"Backbone1 Required columns: {backbone1_required_columns}. "
                )
            if not backbone2_required_columns.issubset(provided_columns):
                error_msgs.append(
                    f"Backbone2 is missing required columns: {backbone2_required_columns - provided_columns}. "
                    f"Data Provided columns: {data_provided_columns}. "
                    f"Task Provided columns: {task_provided_columns}. "
                    f"Backbone2 Required columns: {backbone2_required_columns}. "
                )
            if error_msgs:
                raise ValueError("\n".join(error_msgs))
