# pylint: disable=W0613

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from ignite.contrib.handlers.tensorboard_logger import (
    GradsHistHandler,
    GradsScalarHandler,
    OptimizerParamsHandler,
    TensorboardLogger,
    WeightsHistHandler,
    WeightsScalarHandler,
    global_step_from_engine,
)
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

from autoembedder.evaluator import loss_delta
from autoembedder.model import Autoembedder, model_input

date = datetime.now()


def fit(
    parameters: Dict,
    model: Autoembedder,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    eval_df: Optional[Union[dd.DataFrame, pd.DataFrame]] = None,
) -> Autoembedder:

    """
    This method is the general wrapper around the fitting process. It is preparing the optimizer, the loss function, the trainer,
    the validator and the evaluator. Then it attaches everything to the corresponding engines and runs the training.

    Args:
        parameters (Dict[str, Any]): The parameters of the training process.
            In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.
        model (Autoembedder): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        df (Optional[Union[dd.DataFrame, pd.DataFrame]], optional): Dask or Pandas DataFrame for the evaluation step.
            If the path to the evaluation data is given in the parameters (`eval_input_path`) this argument is not needed.
            If neither `eval_input_path` nor `eval_df` is given, no evaluation step is performed.

    Returns:
        Autoembedder: Trained Autoembedder model.
    """

    model = model.to(
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available() and parameters.get("use_mps", 0) == 1
            else "cpu"
        )
    )
    if torch.backends.mps.is_available() is False or parameters.get("use_mps", 0) == 0:
        model = model.double()

    optimizer = Adam(
        model.parameters(),
        lr=parameters.get("lr", 1e-3),
        weight_decay=parameters.get("weight_decay", 0),
        amsgrad=parameters.get("amsgrad", 0) == 1,
    )
    criterion = MSELoss()

    if parameters.get("xavier_init", 0) == 1:
        model.init_xavier_weights()

    tb_logger = None
    if parameters.get("tensorboard_log_path", None):
        tb_logger = TensorboardLogger(
            log_dir=f"{parameters['tensorboard_log_path']}/{date.strftime('%Y.%m.%d-%H_%M')}"
        )

    trainer = Engine(
        partial(
            __training_step,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            parameters=parameters,
        )
    )
    validator = Engine(
        partial(
            __validation_step, model=model, criterion=criterion, parameters=parameters
        )
    )
    evaluator = Engine(
        partial(loss_delta, model=model, parameters=parameters, df=eval_df)
    )

    if parameters.get("verbose", 0) >= 1:
        __print_summary(model, train_dataloader, parameters)
    __attach_progress_bar(trainer, parameters.get("verbose", 0) == 2)
    __attach_tb_logger_if_needed(
        trainer, validator, evaluator, tb_logger, model, optimizer, parameters
    )
    __attach_terminate_on_nan(trainer)
    __attach_validation(
        trainer, validator, test_dataloader, parameters.get("verbose", 0) >= 1
    )

    if (
        parameters.get("eval_input_path", None) or eval_df is not None
    ) and parameters.get("target", None):
        __attach_evaluation(
            trainer, evaluator, test_dataloader, parameters.get("verbose", 0) >= 1
        )
    __attach_checkpoint_saving_if_needed(
        trainer, validator, model, optimizer, parameters
    )

    __attach_tb_teardown_if_needed(tb_logger, trainer, validator, evaluator, parameters)

    if parameters.get("load_checkpoint_path", None):
        checkpoint = torch.load(
            parameters["load_checkpoint_path"],
            map_location=torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                and parameters.get("use_mps", 0) == 1
                else "cpu"
            ),
        )
        Checkpoint.load_objects(
            to_load={"model": model, "optimizer": optimizer, "trainer": trainer},
            checkpoint=checkpoint,
        )
        print(
            f"""
            Checkpoint loaded!
            Epoch_length: {checkpoint['trainer']['epoch_length']}
            Iterations: {checkpoint['trainer']['iteration']}
            """
        )

    trainer.run(
        train_dataloader,
        max_epochs=parameters["epochs"],
        epoch_length=(
            len(train_dataloader.dataset.ddf.index) // train_dataloader.batch_size
        ),
    )
    return model


def __training_step(
    engine: Engine,
    batch: NamedTuple,
    model: Autoembedder,
    optimizer: Adam,
    criterion: MSELoss,
    parameters: Dict,
) -> Union[np.float32, np.float64]:

    """
    Here the actual training step is performed. It is called by the training engine.
    Not using [PyTorch ignite](https://github.com/pytorch/ignite)
    this code would be wrapped in some kind of training loop over a range of epochs and batches.
    But using ignite this is handled by the engine.

    Args:
        engine (ignite.engine.Engine): The engine that is calling this method.
        batch (NamedTuple): The batch that is passed to the engine for training.
        model (Autoembedder): The model to be trained.
        optimizer (torch.optim): The optimizer to be used for training.
        criterion (torch.nn.MSELoss): The loss function to be used for training.
        parameters (Dict[str, Any]): The parameters of the training process.

    Returns:
        Union[np.float32, np.float64]: The loss of the current batch.
    """

    model.train()
    optimizer.zero_grad()
    cat, cont = model_input(batch, parameters)
    outputs = model(cat, cont)
    train_loss = criterion(outputs, model.last_target)

    if parameters.get("l1_lambda", 0) > 0:
        l1_lambda = parameters["l1_lambda"]
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        train_loss = train_loss + l1_lambda * l1_norm

    train_loss.backward()
    optimizer.step()
    return train_loss.item()


def __validation_step(
    engine: Engine,
    batch: NamedTuple,
    model: Autoembedder,
    criterion: MSELoss,
    parameters: Dict,
) -> Union[np.float32, np.float64]:

    """
    Args:
        engine (ignite.engine.Engine): The engine that is calling this method.
        batch (NamedTuple): The batch that is passed to the engine for validation.
        model (Autoembedder): The model used for validation.
        criterion (torch.nn.MSELoss): The loss function to be used for validation.
        parameters (Dict[str, Any]): The parameters of the validation process.

    Returns:
        Union[np.float32, np.float64]: The loss of the current batch.
    """

    model.eval()
    with torch.no_grad():
        cat, cont = model_input(batch, parameters)
        val_outputs = model(cat, cont)
        val_loss = criterion(val_outputs, model.last_target)
    return val_loss.item()


def __attach_progress_bar(trainer: Engine, show_bar: bool) -> None:
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    if show_bar:
        ProgressBar(True).attach(trainer, ["loss"])


def __attach_validation(
    trainer: Engine, validator: Engine, test_dataloader: DataLoader, show_bar: bool
) -> None:
    def run_validator(
        trainer: Engine, validator: Engine, dataloader: DataLoader, show_bar: bool
    ) -> None:
        validator.run(
            dataloader,
            epoch_length=(len(dataloader.dataset.ddf.index) // dataloader.batch_size),
            max_epochs=1,
        )
        if show_bar:
            ProgressBar(True).log_message(
                f"Epoch [{trainer.state.epoch}/{trainer.state.max_epochs}]: validation loss: {validator.state.metrics['loss']:.7f}"
            )

    RunningAverage(output_transform=lambda x: x).attach(validator, "loss")
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        partial(
            run_validator,
            validator=validator,
            dataloader=test_dataloader,
            show_bar=show_bar,
        ),
    )


def __attach_evaluation(
    trainer: Engine, evaluator: Engine, dataloader: DataLoader, show_bar: bool
) -> None:
    def run_evaluator(
        trainer: Engine, evaluator: Engine, dataloader: DataLoader, show_bar: bool
    ) -> None:
        evaluator.run(
            dataloader,
            epoch_length=(len(dataloader.dataset.ddf.index) // dataloader.batch_size),
            max_epochs=1,
        )
        if show_bar:
            ProgressBar(True).log_message(
                f"Mean loss delta: {evaluator.state.metrics['mean_loss_delta']:.7f}"
            )
            ProgressBar(True).log_message(
                f"Median loss delta: {evaluator.state.metrics['median_loss_delta']:.7f}"
            )  # pylint: disable=C0301

    RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, "mean_loss_delta")
    RunningAverage(output_transform=lambda x: x[1]).attach(
        evaluator, "median_loss_delta"
    )
    trainer.add_event_handler(
        Events.COMPLETED,
        partial(
            run_evaluator, evaluator=evaluator, dataloader=dataloader, show_bar=show_bar
        ),
    )


def __attach_tb_logger_if_needed(
    trainer: Engine,
    validator: Engine,
    evaluator: Engine,
    tb_logger: TensorboardLogger,
    model: Autoembedder,
    optimizer: Adam,
    parameters: Dict,
) -> None:
    if parameters.get("tensorboard_log_path", None) is None:
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
) -> None:
    def teardown_tb_logger(
        trainer: Engine,
        validator: Engine,
        tb_logger: TensorboardLogger,
        parameters: Dict,
    ) -> None:
        metrics = {
            "hparam/train_loss": trainer.state.metrics["loss"],
            "hparam/val_loss": validator.state.metrics["loss"],
        }
        if parameters.get("eval_input_path", None):
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

    if parameters.get("tensorboard_log_path", None) is None:
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
    trainer: Engine,
    engine: Engine,
    model: nn.Module,
    optimizer: Adam,
    parameters: Dict,
    metric: str = "loss",
) -> None:
    def score_function(engine: Engine) -> float:
        return engine.state.metrics[metric]

    if (
        parameters.get("n_save_checkpoints", 0) == 0
        or parameters.get("model_save_path", None) is None
    ):
        return

    checkpoint_dir = f"{parameters['model_save_path']}/checkpoints/{date.strftime('%Y-%m-%d')}/{date.strftime('%H-%M-%S')}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpointer = Checkpoint(
        to_save={"model": model, "optimizer": optimizer, "trainer": trainer},
        save_handler=checkpoint_dir,
        score_name=metric,
        score_function=score_function,
        n_saved=parameters.get("n_save_checkpoints", 0),
        greater_or_equal=True,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer)


def __attach_terminate_on_nan(trainer: Engine) -> None:
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


def __print_summary(
    model: Autoembedder, train_dataloader: DataLoader, parameters: Dict
) -> None:
    batch_size = train_dataloader.batch_size
    cat, cont = model_input(
        next(train_dataloader.dataset.ddf.itertuples(index=False)), parameters
    )
    if cat.shape[0] == 1:
        summary(model)
        return
    summary(model, [(cat.shape[1], batch_size), (cont.shape[1], batch_size)])  # type: ignore
