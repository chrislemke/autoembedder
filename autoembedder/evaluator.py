# -*- coding: utf-8 -*-

from typing import Dict, List, NamedTuple, Tuple

import dask.dataframe as dd
import numpy as np
import torch
from einops import rearrange
from torch.nn import MSELoss

from autoembedder.model import Autoembedder, model_input


def loss_diff(_, __, model: Autoembedder, parameters: Dict) -> Tuple[float, float]:  # type: ignore
    """
    Args:
        _ (None): Not in use. Needed by Pytorch-ignite.
        __ (None): Not in use. Needed by Pytorch-ignite.
        model (Autoembedder): Instance from the model used for prediction.
        parameters (Dict[str, Any]): Dictionary with the parameters used for training and prediction.
            In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.

    Returns:
        Tuple[float, float]: `loss_mean_diff`, `loss_std_diff` and DataFrame.
    """
    target = parameters["target"]
    df = (
        dd.read_parquet(parameters["eval_input_path"], infer_divisions=True)
        .compute()
        .sample(frac=1)
    )

    df_1 = df.query(f"{target} == 1").drop([target], axis=1)
    df_0 = df.query(f"{target} == 0").drop([target], axis=1).sample(n=df_1.shape[0])

    losses_0: List[float] = []
    losses_1: List[float] = []

    for losses_df, losses in [(df_0, losses_0), (df_1, losses_1)]:
        loss = MSELoss()
        for batch in losses_df.itertuples(index=False):
            losses.append(__predict(model, batch, loss, parameters))

    if parameters.get("trim_eval_errors", 0) == 1:
        losses_0.remove(max(losses_0))
        losses_0.remove(min(losses_0))
        losses_1.remove(max(losses_1))
        losses_1.remove(min(losses_1))

    return np.absolute(np.mean(losses_1) - np.mean(losses_0)), np.absolute(
        np.median(losses_1) - np.median(losses_0)
    )


def __predict(
    model: Autoembedder, batch: NamedTuple, loss_fn: MSELoss, parameters: Dict
) -> float:

    """
    Args:
        model (Autoembedder): Instance from the model used for prediction.
        batch (NamedTuple): A batch of data.
        loss_fn (torch.nn.MSELoss): Instance of the loss function.
        parameters (Dict): Dictionary with the parameters used for evaluation.
            In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.

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
