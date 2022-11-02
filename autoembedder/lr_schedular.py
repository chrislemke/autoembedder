# -*- coding: utf-8 -*-

from typing import Optional

import torch
from ignite.engine import Engine
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ReduceLROnPlateauScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Adam,
        metric_name: str,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        """

        Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
        This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs,
        the learning rate is reduced.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            metric_name (str): Name of the metric to use for the scheduler
            mode (str): One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
            threshold (float): Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of rel, abs. In rel mode, it compares the relative change in metrics compared to the best value in the history. In abs mode, it compares the direct difference with the best value in the history. Default: ‘rel’.
            cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
            verbose (bool): Whether to print the status of the scheduler. Default: False.

        Returns:
            ReduceLROnPlateauScheduler: A ReduceLROnPlateauScheduler instance.
        """

        self.metric_name = metric_name
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )

    def __call__(self, ignite_engine: Engine, name: Optional[str] = None) -> None:
        self.scheduler.step(ignite_engine.state.metrics[self.metric_name])

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()
