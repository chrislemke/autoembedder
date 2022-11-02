# -*- coding: utf-8 -*-

from typing import Dict, List, NamedTuple, Tuple

import dask.dataframe as dd
import pandas as pd
import torch
from einops import rearrange
from torch.nn import MSELoss

from autoembedder.model import Autoembedder, model_input


def loss_delta(_, __, model: Autoembedder, parameters: Dict) -> Tuple[float, float]:  # type: ignore
    """
    Args:
        _ (None): Not in use. Needed by Pytorch-ignite.
        __ (None): Not in use. Needed by Pytorch-ignite.
        model (Autoembedder): Instance from the model used for prediction.
        parameters (Dict): Dictionary with the parameters used for training and prediction.

    Returns:
        Tuple[float, float]: `loss_mean_delta`, `loss_std_delta` and dataframe .
    """
    target = parameters["target"]
    df = (
        dd.read_parquet(parameters["eval_input_path"], infer_divisions=True)
        .compute()
        .sample(frac=1)
        .reset_index(drop=True)
    )
    df_0 = df.query(f"{target} == 0").drop([target], axis=1)
    df_1 = df.query(f"{target} == 1").drop([target], axis=1)

    df_0 = df_0.head(df_1.shape[0])
    df_1 = df_1.head(df_1.shape[0])
    losses_0: List[float] = []
    losses_1: List[float] = []

    loss = MSELoss()
    for batch in df_0.itertuples(index=False):
        losses_0.append(__predict(model, batch, loss, parameters))
    for batch in df_1.itertuples(index=False):
        losses_1.append(__predict(model, batch, loss, parameters))

    df = pd.DataFrame(zip(losses_0, losses_1), columns=["loss_0", "loss_1"])
    df_mean = df.mean(axis=0)
    df_median = df.median(axis=0)
    mean_loss_delta = df_mean[1] - df_mean[0]
    median_loss_delta = df_median[1] - df_median[0]
    return mean_loss_delta, median_loss_delta


def __predict(
    model: Autoembedder, batch: NamedTuple, loss_fn: MSELoss, parameters: Dict
) -> float:

    """
    Args:
        model (Autoembedder): Instance from the model used for prediction.
        batch (NamedTuple): A batch of data.
        loss_fn (MSELoss): Instance of the loss function.
        parameters (Dict): Dictionary with the parameters used for evaluation.

    Returns:
        float: Loss value.
    """

    with torch.no_grad():
        model.eval()
        cat, cont = model_input(batch, parameters)
        cat = rearrange(cat, "c r -> r c")
        cont = rearrange(cont, "c r -> r c")
        out = model(cat, cont)
    return loss_fn(out, model.last_target).item()
