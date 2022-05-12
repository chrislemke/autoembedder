#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import itertools
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch
from fps_ai.training.autoencoder import learner
from fps_ai.training.autoencoder.data import dataloader
from fps_ai.training.autoencoder.model import (
    Autoembedder,
    embedded_sizes_and_dims,
    num_cont_columns,
)


def main():
    date = str(datetime.now()).replace(" ", "_").replace(":", "-")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    parser.add_argument("--drop_last", type=int, required=False, default=1)
    parser.add_argument("--pin_memory", type=int, required=False, default=1)
    parser.add_argument("--num_workers", type=int, required=False, default=0)
    parser.add_argument(
        "--model_title", type=str, required=False, default=f"autoembedder_{date}.bin"
    )
    parser.add_argument("--model_save_path", type=str, required=False)
    parser.add_argument(
        "--n_save_checkpoints",
        type=int,
        required=False,
        default=3,
        help="Number of stored checkpoints.",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        required=False,
        help="Path of the checkpoint to load.",
    )
    parser.add_argument("--lr_scheduler", type=int, required=False, default=1)
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        required=False,
        default="min",
        choices=["min", "max"],
    )
    parser.add_argument("--scheduler_patience", type=int, required=False, default=2)

    parser.add_argument("--lr", type=float, required=False, default=0.0001)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument(
        "--layer_bias",
        type=int,
        required=False,
        default=1,
        help="Bias for the encoder layers.",
    )
    parser.add_argument("--weight_decay", type=float, required=False, default=0)
    parser.add_argument("--l1_lambda", type=float, required=False, default=0)
    parser.add_argument("--xavier_init", type=int, required=False, default=1)
    parser.add_argument("--tensorboard_log_path", type=str, required=False)
    parser.add_argument("--train_input_path", type=str, required=True)
    parser.add_argument("--test_input_path", type=str, required=True)
    parser.add_argument("--eval_input_path", type=str, required=False)

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


def __prepare_and_fit(parameters: Dict, model_params: Dict):
    torch.set_default_tensor_type(torch.DoubleTensor)
    train_dl = dataloader(parameters["train_input_path"], parameters)
    test_dl = dataloader(parameters["test_input_path"], parameters)
    num_continuous_cols = num_cont_columns(train_dl.dataset.df)  # type: ignore
    __check_for_consistent_cat_rows(
        train_dl.dataset.df, ast.literal_eval(parameters["cat_columns"])
    )
    embedded_sizes = embedded_sizes_and_dims(
        train_dl.dataset.df,
        test_dl.dataset.df,
        ast.literal_eval(parameters["cat_columns"]),
    )  # type: ignore
    model = Autoembedder(model_params, num_continuous_cols, embedded_sizes)

    learner.fit(parameters, model, train_dl, test_dl)


def __check_for_consistent_cat_rows(df: pd.DataFrame, cat_columns: List[List[str]]):
    df_columns = df.select_dtypes(include="category").columns.to_list()
    cat_columns = list(itertools.chain(*cat_columns))
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
