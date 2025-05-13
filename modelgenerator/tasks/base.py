from typing import Callable, Literal, Optional, Set, Union

import torch
from torch import Tensor
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

import torchmetrics as tm

from jsonargparse import ArgumentParser
from docstring_inheritance import GoogleDocstringInheritanceInitMeta

from modelgenerator.lr_schedulers import LazyLRScheduler
from modelgenerator.backbones import (
    HFSequenceBackbone,
    DefaultConfig,
    LegacyAdapterType,
)

BackboneCallable = Callable[
    [Union[LegacyAdapterType, None], Union[DefaultConfig, None]], HFSequenceBackbone
]


class TaskInterface(pl.LightningModule, metaclass=GoogleDocstringInheritanceInitMeta):
    """Interface class to ensure consistent implementation of essential methods for all tasks.

    Note:
        Tasks will usually take a backbone and adapter as arguments, but these are not strictly required.
        See [SequenceRegression](./#modelgenerator.tasks.SequenceRegression) task for an succinct example implementation.
        Handles the boilerplate of setting up training, validation, and testing steps,
        as well as the optimizer and learning rate scheduler. Subclasses must implement
        the __init__, configure_model, collate, forward, and evaluate methods.

    Args:
        use_legacy_adapter: Whether to use the adapter from the backbone (HF head support). 
            **Warning**: This is not supported for all tasks and will be depreciated in the future.
        strict_loading: Whether to strictly load the model from the checkpoint. 
            If False, replaces missing weights with pretrained weights. 
            Should be enabled when loading a checkpoint from a run with `use_peft` and `save_peft_only`.
        batch_size: The batch size to use for training.
        optimizer: The optimizer to use for training.
        reset_optimizer_states: Whether to reset the optimizer states.
            Set it to True if you want to replace the adapter (e.g. for continued pretraining).
        lr_scheduler: The learning rate scheduler to use for training.
        **kwargs: Additional arguments passed to the parent class.
    """

    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        batch_size: Optional[int] = None,
        use_legacy_adapter: bool = False,
        strict_loading: bool = True,
        reset_optimizer_states: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # NOTE: A very explicit way of preventing unwanted hparams from being
        # saved due to inheritance. All subclasses should include the
        # following condition under super().__init__().
        # Converting it to a reusable method could work but it would rely
        # on the implementation detail of save_hyperparameters() walking up
        # the call stack, which can change at any time.
        if self.__class__ is TaskInterface:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.use_legacy_adapter = use_legacy_adapter
        self.metrics = nn.ModuleDict(
            {
                "train_metrics": nn.ModuleDict(),
                "val_metrics": nn.ModuleDict(),
                "test_metrics": nn.ModuleDict(),
            }
        )
        self.metrics_to_pbar: Set[str] = {}
        self.strict_loading = strict_loading
        self.reset_optimizer_states = reset_optimizer_states

    def configure_model(self) -> None:
        """Configures the model for training and interence. Subclasses must implement this method."""
        raise NotImplementedError

    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: int
    ) -> dict[str, Union[list, Tensor]]:
        """Collates and tokenizes a batch of data into a format that can be passed to the forward and evaluate methods. Subclasses must implement this method.

        Note:
            Tokenization is handled here using the backbone interface.
            Tensor typing and device moving should be handled here.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch
        """
        raise NotImplementedError

    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model on the collated batch of data. Subclasses must implement this method.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data from collate.

        Returns:
            Tensor: The model predictions
        """
        raise NotImplementedError

    def evaluate(
        self,
        preds: Tensor,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
    ) -> dict[str, Union[Tensor, float]]:
        """Calculate loss and update metrics states. Subclasses must implement this method.

        Args:
            preds (Tensor): The model predictions from forward.
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data from collate.
            stage (str, optional): The stage of training (train, val, test). Defaults to None.
            loss_only (bool, optional): If true, only update loss metric. Defaults to False.

        Returns:
            dict[str, Union[Tensor, float]]: The loss and any additional metrics.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for training.

        Returns:
            list: A list of optimizers and learning rate schedulers
        """
        config = {
            "optimizer": self.optimizer(
                filter(lambda p: p.requires_grad, self.parameters())
            )
        }
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(config["optimizer"])
            if isinstance(scheduler, LazyLRScheduler):
                scheduler.initialize(self.trainer)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "train_loss",  # Only used for torch.optim.lr_scheduler.ReduceLROnPlateau
            }
        return config

    def on_save_checkpoint(self, checkpoint: dict):
        if hasattr(self.backbone, "on_save_checkpoint"):
            self.backbone.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict):
        if self.reset_optimizer_states:
            checkpoint["optimizer_states"] = {}
            checkpoint["lr_schedulers"] = {}

    def training_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> Tensor:
        """Runs a training step on a batch of data. Calls collate, forward, and evaluate methods in order.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The loss from the training step
        """
        collated_batch = self.transform(batch, batch_idx)
        preds = self.forward(collated_batch)
        outputs = self.evaluate(preds, collated_batch, "train", loss_only=False)
        self.log_loss_and_metrics(outputs["loss"], "train")
        return outputs

    def validation_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> Tensor:
        """Runs a validation step on a batch of data. Calls collate, forward, and evaluate methods in order.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The loss from the validation step
        """
        collated_batch = self.transform(batch, batch_idx)
        preds = self.forward(collated_batch)
        outputs = self.evaluate(preds, collated_batch, "val", loss_only=False)
        self.log_loss_and_metrics(outputs["loss"], "val")
        return {"predictions": preds, **batch, **collated_batch, **outputs}

    def test_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> Tensor:
        """Runs a test step on a batch of data. Calls collate, forward, and evaluate methods in order.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            Tensor: The loss from the test step
        """
        collated_batch = self.transform(batch, batch_idx)
        preds = self.forward(collated_batch)
        outputs = self.evaluate(preds, collated_batch, "test", loss_only=False)
        self.log_loss_and_metrics(outputs["loss"], "test")
        return {"predictions": preds, **batch, **collated_batch, **outputs}

    def predict_step(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: Optional[int] = None
    ) -> dict[str, Union[list, Tensor]]:
        """Infers predictions from a batch of data. Calls collate and forward methods in order.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data from the DataLoader
            batch_idx (int, optional): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The predictions from the model along with the collated batch.
        """
        collated_batch = self.transform(batch, batch_idx)
        preds = self.forward(collated_batch)
        return {"predictions": preds, **batch, **collated_batch}

    def get_metrics_by_stage(
        self, stage: Literal["train", "val", "test"]
    ) -> nn.ModuleDict:
        """Returns the metrics dict for a given stage.

        Args:
            stage (str): The stage of training (train, val, test)

        Returns:
            nn.ModuleDict: The metrics for the given stage
        """
        try:
            return self.metrics[f"{stage}_metrics"]
        except KeyError:
            raise ValueError(
                f"Stage must be one of 'train', 'val', or 'test'. Got {stage}"
            )

    def log_loss_and_metrics(
        self, loss: Tensor, stage: Literal["train", "val", "test"]
    ) -> None:
        """Logs the loss and metrics for a given stage.

        Args:
            loss (Tensor): The loss from the training, validation, or testing step
            stage (str): The stage of training (train, val, test)
        """
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=stage != "train")
        for k, v in self.metrics[f"{stage}_metrics"].items():
            self.log(f"{stage}_{k}", v, prog_bar=k in self.metrics_to_pbar)

    def call_or_update_metric(
        self, stage: Literal["train", "val", "test"], metric: tm.Metric, *args, **kwargs
    ):
        if stage == "train":
            # in addition to .update(), metric.__call__ also .compute() the metric
            # for the current batch. However, .compute() may fail if data is insufficient.
            try:
                metric(*args, **kwargs)
            except ValueError:
                metric.update(*args, **kwargs)
        else:
            # update only since per step metrics are not logged in val and test stages
            metric.update(*args, **kwargs)

    @classmethod
    def from_config(cls, config: dict) -> "TaskInterface":
        """Creates a task model from a configuration dictionary

        Args:
            config (Dict[str, Any]): Configuration dictionary

        Returns:
            TaskInterface: Task model
        """
        parser = ArgumentParser()
        parser.add_class_arguments(cls, "model")
        init = parser.instantiate_classes(parser.parse_object(config))
        init.model.configure_model()
        return init.model

