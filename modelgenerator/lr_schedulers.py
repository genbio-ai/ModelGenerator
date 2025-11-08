import torch
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from transformers import get_scheduler, SchedulerType
from lightning.pytorch import Trainer


class LazyLRScheduler(LambdaLR):
    """
    A wrapper class for torch.optim.lr_scheduler.LambdaLR that allows for lazy initialization.
    """

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def initialize(self, trainer: Trainer):
        """
        Initialize the scheduler with the trainer object.

        Args:
            trainer (lightning.pytorch.Trainer): The trainer object that will be used during training.
        """
        lr_lambdas = self._initialize(trainer)
        super().__init__(self.optimizer, lr_lambdas, last_epoch=self.last_epoch)

    def _initialize(self, trainer: Trainer):
        raise NotImplementedError(
            "The _initialize method must be implemented in the derived class."
        )


class CosineWithWarmup(LazyLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_ratio: Optional[float] = None,
        num_warmup_steps: Optional[int] = None,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                The optimizer that will be used during training.
            warmup_ratio (float, optional):
                The ratio of warmup steps to the total number of training steps. Defaults to None.
            num_warmup_steps (int, optional):
                The number of warmup steps to do. Defaults to None.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch=last_epoch)
        self.warmup_ratio = warmup_ratio
        self.num_warmup_steps = num_warmup_steps
        if self.warmup_ratio is None and self.num_warmup_steps is None:
            raise ValueError("Either warmup_ratio or num_warmup_steps must be provided.")

    def _initialize(self, trainer: Trainer):
        num_training_steps = trainer.estimated_stepping_batches
        if num_training_steps == float("inf"):
            raise ValueError(
                "A deterministic number of training steps is required to use this scheduler."
            )
        if self.num_warmup_steps is not None:
            num_warmup_steps = self.num_warmup_steps
        else:
            num_warmup_steps = int((self.warmup_ratio * num_training_steps))
        scheduler = get_scheduler(
            SchedulerType.COSINE,
            Optimizer([torch.zeros(1)], {"lr": 0}),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler.lr_lambdas


class ConstantWithWarmup(LazyLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_ratio: Optional[float] = None,
        num_warmup_steps: Optional[int] = None,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                The optimizer that will be used during training.
            warmup_ratio (float, optional):
                The ratio of warmup steps to the total number of training steps. Defaults to None.
            num_warmup_steps (int, optional):
                The number of warmup steps to do. Defaults to None.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch=last_epoch)
        self.warmup_ratio = warmup_ratio
        self.num_warmup_steps = num_warmup_steps
        if self.warmup_ratio is None and self.num_warmup_steps is None:
            raise ValueError("Either warmup_ratio or num_warmup_steps must be provided.")

    def _initialize(self, trainer: Trainer):
        num_training_steps = trainer.estimated_stepping_batches
        if num_training_steps == float("inf"):
            raise ValueError(
                "A deterministic number of training steps is required to use this scheduler."
            )
        if self.num_warmup_steps is not None:
            num_warmup_steps = self.num_warmup_steps
        else:
            num_warmup_steps = int((self.warmup_ratio * num_training_steps))
        scheduler = get_scheduler(
            SchedulerType.CONSTANT_WITH_WARMUP,
            Optimizer([torch.zeros(1)], {"lr": 0}),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler.lr_lambdas


class LinearWithWarmup(LazyLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_ratio: Optional[float] = None,
        num_warmup_steps: Optional[int] = None,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                The optimizer that will be used during training.
            warmup_ratio (float, optional):
                The ratio of warmup steps to the total number of training steps. Defaults to None.
            num_warmup_steps (int, optional):
                The number of warmup steps to do. Defaults to None.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch=last_epoch)
        self.warmup_ratio = warmup_ratio
        self.num_warmup_steps = num_warmup_steps
        if self.warmup_ratio is None and self.num_warmup_steps is None:
            raise ValueError("Either warmup_ratio or num_warmup_steps must be provided.")

    def _initialize(self, trainer: Trainer):
        num_training_steps = trainer.estimated_stepping_batches
        if num_training_steps == float("inf"):
            raise ValueError(
                "A deterministic number of training steps is required to use this scheduler."
            )
        if self.num_warmup_steps is not None:
            num_warmup_steps = self.num_warmup_steps
        else:
            num_warmup_steps = int((self.warmup_ratio * num_training_steps))
        scheduler = get_scheduler(
            SchedulerType.LINEAR,
            Optimizer([torch.zeros(1)], {"lr": 0}),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler.lr_lambdas
