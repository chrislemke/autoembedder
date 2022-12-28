import ast
import itertools
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd
import torch
import typer

from autoembedder import learner
from autoembedder.data import dataloader
from autoembedder.model import Autoembedder, embedded_sizes_and_dims, num_cont_columns

app = typer.Typer(rich_markup_mode="rich")


@app.command(help="Runs the training process for the autoembedder.")  # type: ignore
def main(
    batch_size: int = typer.Option(32, min=1, help="Batch size for the training."),
    epochs: int = typer.Option(30, min=1, help="Number of epochs for the training."),
    drop_last: bool = typer.Option(
        True, help="Drop last batch if it is smaller than the batch size."
    ),
    pin_memory: bool = typer.Option(True, help="Pin memory for the dataloader."),
    num_workers: int = typer.Option(0, help="Number of workers for the dataloader."),
    use_mps: bool = typer.Option(
        False,
        help="Set this to `True` if you want to use the MPS Backend for running on Mac using the M1 GPU.",
    ),
    model_title: str = typer.Option(
        f"autoembedder_{str(datetime.now()).replace(' ', '_').replace(':', '-')}.pt",
        help="Title of the model.",
    ),
    model_save_path: str = typer.Option(None, help="Path to save the model."),
    n_save_checkpoints: int = typer.Option(6, help="Number of stored checkpoints."),
    load_checkpoint_path: str = typer.Option(
        None, help="Path of the checkpoint to load."
    ),
    lr: float = typer.Option(0.001, help="Learning rate for the optimizer."),
    amsgrad: bool = typer.Option(False, help="Use amsgrad for the optimizer."),
    layer_bias: bool = typer.Option(True, help="Bias for the encoder layers."),
    dropout_rate: float = typer.Option(0, help="Dropout rate for the encoder layers."),
    activation: str = typer.Option(
        "tanh", help="Activation function for the encoder layers."
    ),
    weight_decay: float = typer.Option(0, help="Weight decay for the optimizer."),
    l1_lambda: float = typer.Option(0, help="L1 regularization for the optimizer."),
    xavier_init: bool = typer.Option(
        False, help="Use xavier initialization for the encoder layers."
    ),
    tensorboard_log_path: str = typer.Option(
        None, help="Path to save the tensorboard logs."
    ),
    drop_cat_columns: bool = typer.Option(
        False, help="If `True`, drop categorical columns from the datasets."
    ),
    trim_eval_errors: bool = typer.Option(
        False, help="If `True`, trim the evaluation errors."
    ),
    target: str = typer.Option("target", help="Name of the target column."),
    train_input_path: str = typer.Option(None, help="Path to the training dataset."),
    test_input_path: str = typer.Option(None, help="Path to the test dataset."),
    eval_input_path: str = typer.Option(None, help="Path to the evaluation dataset."),
    verbose: int = typer.Option(
        0,
        help="""
        Set this to 1 if you want to see the model summary and the validation and evaluation results.
        set this to 2 if you want to see the training progress bar. 0 means no output.
        """,
    ),
    hidden_layers: str = typer.Option(
        None,
        help="""
        Contains a string representation of a list of list of integers which represents the hidden layer structure.
        E.g.: `"[[64, 32], [32, 16], [16, 8]]"`
        """,
    ),
    cat_columns: str = typer.Option(
        "[]",
        help="""
        Contains a string representation of a list of list of categorical columns (strings).
        The columns which use the same encoder should be together in a list. E.g.: `"[['a', 'b'], ['c']]"`.
        If you don't need or want to use categorical columns from your dataset you may consider using: `--drop_cat_columns`.
        """,
    ),
) -> None:
    """
    Main function for parsing arguments and start `_prepare_and_fit`.
    In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.

    Returns:
        None

    """

    cat_columns = cat_columns.replace("\\", "")
    hidden_layers = hidden_layers.replace("\\", "")
    m_config = {
        "hidden_layers": ast.literal_eval(hidden_layers),
        "layer_bias": layer_bias,
    }

    parameters = {
        "batch_size": batch_size,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "use_mps": use_mps,
        "model_title": model_title,
        "model_save_path": model_save_path,
        "n_save_checkpoints": n_save_checkpoints,
        "load_checkpoint_path": load_checkpoint_path,
        "lr": lr,
        "amsgrad": amsgrad,
        "epochs": epochs,
        "dropout_rate": dropout_rate,
        "activation": activation,
        "weight_decay": weight_decay,
        "l1_lambda": l1_lambda,
        "xavier_init": xavier_init,
        "tensorboard_log_path": tensorboard_log_path,
        "drop_cat_columns": drop_cat_columns,
        "trim_eval_errors": trim_eval_errors,
        "target": target,
        "train_input_path": train_input_path,
        "test_input_path": test_input_path,
        "eval_input_path": eval_input_path,
        "verbose": verbose,
    }

    _prepare_and_fit(parameters, m_config)


def _prepare_and_fit(parameters: Dict, model_params: Dict) -> None:

    """
    Prepares the data by creating a training and testing `dataloader`. `num_continuous_cols` is determined and passed to the
    `Autoembedder` model for creating the needed layers.

    Args:
        parameters (Dict): Dictionary containing the parameters for the training, and creating the `dataloaders`.
        model_params (Dict): Dictionary containing the model parameters.

    Returns:
        None
    """

    if (
        torch.backends.mps.is_available() is False
        or parameters.get("use_mps", False) is False
    ):
        torch.set_default_tensor_type(torch.DoubleTensor)

    train_dl = dataloader(parameters["train_input_path"], parameters)
    test_dl = dataloader(parameters["test_input_path"], parameters)
    num_continuous_cols = num_cont_columns(train_dl.dataset.ddf)
    if parameters.get("drop_cat_columns", False) is False:
        _check_for_consistent_cat_rows(
            train_dl.dataset.ddf, ast.literal_eval(parameters["cat_columns"])
        )
    embedded_sizes = embedded_sizes_and_dims(
        train_dl.dataset.ddf,
        test_dl.dataset.ddf,
        ast.literal_eval(parameters.get("cat_columns", "[]")),
    )
    model = Autoembedder(model_params, num_continuous_cols, embedded_sizes)

    learner.fit(parameters, model, train_dl, test_dl)


def _check_for_consistent_cat_rows(
    df: pd.DataFrame, cat_columns: Iterable[List[str]]
) -> None:

    """
    Checks if the categorical rows in the the dataframe (`df`) are consistent with the categorical columns (`cat_columns`).
    This is needed so the is set up correctly. Please check the `--cat_columns` parameter for more information.

    Args:
        df (pandas.DataFrame): The dataframe containing the categorical columns.
        cat_columns (List[List[str]]): A list of lists representing the categorical columns which were encoded using the same encoder. E.g.: [['a', 'b'], ['c']]. Check the `--cat_columns` parameter for more information.

    Returns:
        None
    """

    df_columns = df.select_dtypes(include="category").columns.to_list()
    cat_columns = list(itertools.chain(*cat_columns))  # type: ignore
    assert set(df_columns) == set(
        cat_columns
    ), f"""
        The rows from the dataframe should be consistent to the ones defined in `--cat_columns`! Remember to adjust the
        `--cat_columns` according to the dataframe.
        `df_columns` = {df_columns}
        vs.
        `cat_columns` = {cat_columns}
        """


if __name__ == "__main__":
    app()
