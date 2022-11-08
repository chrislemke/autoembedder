import argparse
import ast
import itertools
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd
import torch

from autoembedder import learner
from autoembedder.data import dataloader
from autoembedder.model import Autoembedder, embedded_sizes_and_dims, num_cont_columns


def main() -> None:
    """
    Main function for parsing arguments and start `__prepare_and_fit`.
    In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.

    Returns:
        None

    """
    date = str(datetime.now()).replace(" ", "_").replace(":", "-")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--drop_last", type=int, required=False, default=1)
    parser.add_argument("--pin_memory", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=0)
    parser.add_argument(
        "--use_mps",
        type=int,
        required=False,
        default=0,
        help="Set this to `1` if you want to use the MPS Backend for running on Mac using the M1 GPU.",
    )
    parser.add_argument(
        "--model_title", type=str, required=False, default=f"autoembedder_{date}.pt"
    )
    parser.add_argument("--model_save_path", type=str, required=False)
    parser.add_argument(
        "--n_save_checkpoints",
        type=int,
        required=False,
        default=6,
        help="Number of stored checkpoints.",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        required=False,
        help="Path of the checkpoint to load.",
    )

    parser.add_argument("--lr", type=float, required=False, default=0.001)
    parser.add_argument("--amsgrad", type=int, required=False, default=0)
    parser.add_argument("--epochs", type=int, required=False, default=30)
    parser.add_argument(
        "--layer_bias",
        type=int,
        required=False,
        default=1,
        help="Bias for the encoder layers.",
    )
    parser.add_argument("--dropout_rate", type=float, required=False, default=0)
    parser.add_argument("--activation", type=str, required=False, default="tanh")
    parser.add_argument("--weight_decay", type=float, required=False, default=0)
    parser.add_argument("--l1_lambda", type=float, required=False, default=0)
    parser.add_argument("--xavier_init", type=int, required=False, default=0)
    parser.add_argument("--tensorboard_log_path", type=str, required=False)
    parser.add_argument(
        "--drop_cat_columns",
        type=int,
        required=False,
        default=0,
        help="If `1`, drop categorical columns from the datasets.",
    )
    parser.add_argument("--trim_eval_errors", type=int, required=False, default=0)
    parser.add_argument("--target", type=str, required=False)
    parser.add_argument("--train_input_path", type=str, required=True)
    parser.add_argument("--test_input_path", type=str, required=True)
    parser.add_argument("--eval_input_path", type=str, required=False)
    parser.add_argument("--verbose", type=int, required=False, default=1)

    parser.add_argument(
        "--hidden_layer_representation",
        type=str,
        required=True,
        help="""
        Contains a string representation of a list of list of integers which represents the hidden layer structure.
        E.g.: `"[[64, 32], [32, 16], [16, 8]]"`
        """,
    )
    parser.add_argument(
        "--cat_columns",
        type=str,
        required=False,
        default="[]",
        help="""
        Contains a string representation of a list of list of categorical columns (strings).
        The columns which use the same encoder should be together in a list. E.g.: `"[['a', 'b'], ['c']]"`.
        If you don't need or want to use categorical columns from your dataset you may consider using: `--drop_cat_columns`.
        """,
    )

    args, _ = parser.parse_known_args()
    args.cat_columns = args.cat_columns.replace("\\", "")
    args.hidden_layer_representation = args.hidden_layer_representation.replace(
        "\\", ""
    )
    m_config = {
        "hidden_layers": ast.literal_eval(args.hidden_layer_representation),
        "layer_bias": args.layer_bias,
    }
    __prepare_and_fit(vars(args), m_config)


def __prepare_and_fit(parameters: Dict, model_params: Dict) -> None:

    """
    Prepares the data by creating a training and testing `dataloader`. `num_continuous_cols` is determined and passed to the
    `Autoembedder` model for creating the needed layers.

    Args:
        parameters (Dict): Dictionary containing the parameters for the training, and creating the `dataloaders`.
        model_params (Dict): Dictionary containing the model parameters.

    Returns:
        None
    """

    if torch.backends.mps.is_available() is False or parameters["use_mps"] == 0:
        torch.set_default_tensor_type(torch.DoubleTensor)

    train_dl = dataloader(parameters["train_input_path"], parameters)
    test_dl = dataloader(parameters["test_input_path"], parameters)
    num_continuous_cols = num_cont_columns(train_dl.dataset.ddf)
    if parameters["drop_cat_columns"] == 0:
        __check_for_consistent_cat_rows(
            train_dl.dataset.ddf, ast.literal_eval(parameters["cat_columns"])
        )
    embedded_sizes = embedded_sizes_and_dims(
        train_dl.dataset.ddf,
        test_dl.dataset.ddf,
        ast.literal_eval(parameters["cat_columns"]),
    )
    model = Autoembedder(model_params, num_continuous_cols, embedded_sizes)

    learner.fit(parameters, model, train_dl, test_dl)


def __check_for_consistent_cat_rows(
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
    main()
