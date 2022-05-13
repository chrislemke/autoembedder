#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0613

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, NamedTuple

import tensorwatch as tw
import torch
from fps_ai.training.autoencoder.evaluator import loss_delta
from fps_ai.training.autoencoder.lr_schedular import ReduceLROnPlateauScheduler
from fps_ai.training.autoencoder.model import Autoembedder, model_input
from ignite.contrib.handlers.tensorboard_logger import *  # pylint: disable=W0401,W0614
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, TerminateOnNan
from ignite.metrics import RunningAverage
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

date = datetime.now()


def fit(
    parameters: Dict,
    model: Autoembedder,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
):
    model = model.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ).double()
    optimizer = Adam(
        model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"]
    )
    criterion = MSELoss()

    if parameters["xavier_init"] == 1:
        model.init_xavier_weights()

    tb_logger = None
    if parameters["tensorboard_log_path"]:
        tb_logger = TensorboardLogger(
            log_dir=f"{parameters['tensorboard_log_path']}/{date.strftime('%Y.%m.%d-%H_%M')}"
        )
    if parameters["use_tensorwatch"] == 1:
        tensorwatch = tw.Watcher()

    trainer = Engine(
        partial(
            __training_step,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            parameters=parameters,
        )
    )
    validator = Engine(partial(__validation_step, model=model, criterion=criterion))
    evaluator = Engine(
        partial(loss_delta, model=model, eval_input_path=parameters["eval_input_path"])
    )

    __print_summary(model, train_dataloader)
    __attach_scheduler_if_needed(validator, optimizer, parameters)
    __attach_progress_bar(trainer)
    __attach_tb_logger_if_needed(
        trainer, validator, evaluator, tb_logger, model, optimizer, parameters
    )
    __attach_terminate_on_nan(trainer)
    __attach_validation(trainer, validator, test_dataloader)
    if parameters["eval_input_path"]:
        __attach_evaluation(trainer, evaluator, test_dataloader)
    __attach_checkpoint_saving_if_needed(
        trainer, validator, model, optimizer, criterion, parameters
    )
    if parameters["use_tensorwatch"] == 1:
        __attach_tensorwatch(trainer, tensorwatch)
    __attach_save_model_if_needed(trainer, model, optimizer, criterion, parameters)
    __attach_tb_teardown_if_needed(tb_logger, trainer, validator, evaluator, parameters)

    if parameters["load_checkpoint_path"]:
        checkpoint = torch.load(
            parameters["load_checkpoint_path"],
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        Checkpoint.load_objects(
            to_load={"model": model, "optimizer": optimizer, "trainer": trainer},
            checkpoint=checkpoint,
        )

    trainer.run(
        train_dataloader,
        max_epochs=parameters["epochs"],
        epoch_length=(
            len(train_dataloader.dataset.df.index) // train_dataloader.batch_size
        ),
    )


def __training_step(
    engine: Engine,
    batch: NamedTuple,
    model: Autoembedder,
    optimizer: Adam,
    criterion: MSELoss,
    parameters: Dict,
) -> torch.float64:
    model.train()
    optimizer.zero_grad()
    cat, cont = model_input(batch)
    outputs = model(cat, cont)
    train_loss = criterion(outputs, model.last_target)

    if parameters["l1_lambda"] > 0:
        l1_lambda = parameters["l1_lambda"]
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        train_loss = train_loss + l1_lambda * l1_norm

    train_loss.backward()
    optimizer.step()
    return train_loss.item()


def __validation_step(
    engine: Engine, batch: NamedTuple, model: Autoembedder, criterion: MSELoss
) -> torch.float64:
    model.eval()
    with torch.no_grad():
        cat, cont = model_input(batch)
        val_outputs = model(cat, cont)
        val_loss = criterion(val_outputs, model.last_target)
    return val_loss.item()


def __attach_progress_bar(trainer: Engine):
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(True).attach(trainer, ["loss"])


def __attach_validation(
    trainer: Engine, validator: Engine, test_dataloader: DataLoader
):
    def run_validator(trainer: Engine, validator: Engine, dataloader: DataLoader):
        validator.run(
            dataloader,
            epoch_length=(len(dataloader.dataset.df.index) // dataloader.batch_size),
            max_epochs=1,
        )
        ProgressBar(True).log_message(
            f"Epoch [{trainer.state.epoch}/{trainer.state.max_epochs}]: validation loss: {validator.state.metrics['loss']:.7f}"
        )

    RunningAverage(output_transform=lambda x: x).attach(validator, "loss")
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        partial(run_validator, validator=validator, dataloader=test_dataloader),
    )


def __attach_evaluation(trainer: Engine, evaluator: Engine, dataloader: DataLoader):
    def run_evaluator(trainer: Engine, evaluator: Engine, dataloader: DataLoader):
        evaluator.run(
            dataloader,
            epoch_length=(len(dataloader.dataset.df.index) // dataloader.batch_size),
            max_epochs=1,
        )
        ProgressBar(True).log_message(
            f"Epoch [{trainer.state.epoch}/{trainer.state.max_epochs}]: mean loss delta: {evaluator.state.metrics['mean_loss_delta']:.7f}"
        )
        ProgressBar(True).log_message(
            f"Epoch [{trainer.state.epoch}/{trainer.state.max_epochs}]: median loss delta: {evaluator.state.metrics['median_loss_delta']:.7f}"
        )  # pylint: disable=C0301

    RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, "mean_loss_delta")
    RunningAverage(output_transform=lambda x: x[1]).attach(
        evaluator, "median_loss_delta"
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        partial(run_evaluator, evaluator=evaluator, dataloader=dataloader),
    )


def __attach_scheduler_if_needed(engine: Engine, optimizer: Adam, parameters: Dict):
    if parameters["lr_scheduler"] == 0 or parameters["scheduler_patience"] < 0:
        return

    torch_lr_scheduler = ReduceLROnPlateauScheduler(
        optimizer,
        "loss",
        parameters["scheduler_mode"],
        patience=parameters["scheduler_patience"],
    )
    engine.add_event_handler(Events.COMPLETED, torch_lr_scheduler)


def __attach_tb_logger_if_needed(
    trainer: Engine,
    validator: Engine,
    evaluator: Engine,
    tb_logger: TensorboardLogger,
    model: Autoembedder,
    optimizer: Adam,
    parameters: Dict,
):
    if parameters["tensorboard_log_path"] is None:
        return

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="loss/training",
        metric_names=["loss"],
    )
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="iterations/training",
        metric_names=["loss"],
    )
    tb_logger.attach_output_handler(
        validator,
        event_name=Events.EPOCH_COMPLETED,
        tag="loss/validation",
        metric_names=["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="loss_delta",
        metric_names=["mean_loss_delta"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="loss_delta",
        metric_names=["median_loss_delta"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model),
    )
    tb_logger.attach(
        trainer, event_name=Events.EPOCH_COMPLETED, log_handler=GradsHistHandler(model)
    )
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsScalarHandler(model),
    )
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsScalarHandler(model),
    )


def __attach_tb_teardown_if_needed(
    tb_logger: TensorboardLogger,
    trainer: Engine,
    validator: Engine,
    evaluator: Engine,
    parameters: Dict,
):
    def teardown_tb_logger(
        trainer: Engine,
        validator: Engine,
        tb_logger: TensorboardLogger,
        parameters: Dict,
    ):
        metrics = {
            "hparam/train_loss": trainer.state.metrics["loss"],
            "hparam/val_loss": validator.state.metrics["loss"],
        }
        if parameters["eval_input_path"]:
            metrics["hparam/mean_loss_delta"] = evaluator.state.metrics[
                "median_loss_delta"
            ]
            metrics["hparam/median_loss_delta"] = evaluator.state.metrics[
                "median_loss_delta"
            ]

        tb_logger.writer.add_hparams(
            hparam_dict=parameters,
            metric_dict=metrics,
            run_name=date.strftime("%Y.%m.%d-%H_%M"),
        )

        tb_logger.close()

    if parameters["tensorboard_log_path"] is None:
        return

    trainer.add_event_handler(
        Events.COMPLETED,
        partial(
            teardown_tb_logger,
            validator=validator,
            tb_logger=tb_logger,
            parameters=parameters,
        ),
    )


def __attach_checkpoint_saving_if_needed(
    engine: Engine,
    validator: Engine,
    model: nn.Module,
    optimizer: Adam,
    loss: MSELoss,
    parameters: Dict,
):

    checkpoint_dir = f"{parameters['model_save_path']}/checkpoints/{date.strftime('%Y-%m-%d')}/{date.strftime('%H-%M-%S')}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        engine: Engine,
        validator: Engine,
        model: nn.Module,
        optimizer: Adam,
        loss: MSELoss,
        path: str,
    ):
        loss = round(validator.state.metrics["loss"], 5)
        torch.save(
            {
                "epoch": engine.state.epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"{path}/checkpoint_val-loss_{loss}.pt",
        )

    if parameters["n_save_checkpoints"] == 0:
        return

    engine.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        partial(
            save_checkpoint,
            model=model,
            validator=validator,
            optimizer=optimizer,
            loss=loss,
            path=checkpoint_dir,
        ),
    )


def __attach_save_model_if_needed(
    engine: Engine, model: nn.Module, optimizer: Adam, loss: MSELoss, parameters: Dict
):
    def save_model(
        engine: Engine,
        model: nn.Module,
        optimizer: Adam,
        loss: MSELoss,
        parameters: Dict,
    ):
        Path(parameters["model_save_path"]).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": engine.state.epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"{parameters['model_save_path']}/{parameters['model_title']}",
        )

    if parameters["model_save_path"] is None:
        return

    engine.add_event_handler(
        Events.COMPLETED,
        partial(
            save_model,
            model=model,
            optimizer=optimizer,
            loss=loss,
            parameters=parameters,
        ),
    )


def __attach_tensorwatch(trainer: Engine, tensorwatch: tw.Watcher):
    def attach_to_tensorwatch(trainer: Engine, tensorwatch: tw.Watcher):
        tensorwatch.observe(train_loss=trainer.state.metrics["loss"])

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        partial(attach_to_tensorwatch, tensorwatch=tensorwatch),
    )


def __attach_terminate_on_nan(trainer: Engine):
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


def __print_summary(model: Autoembedder, train_dataloader: DataLoader):
    batch_size = train_dataloader.batch_size
    cat, cont = model_input(next(train_dataloader.dataset.df.itertuples(index=False)))
    if cat.shape[0] == 1:
        summary(model)
        return
    summary(model, [(cat.shape[1], batch_size), (cont.shape[1], batch_size)])  # type: ignore
