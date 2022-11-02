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
    df = dd.read_parquet(parameters["eval_input_path"], infer_divisions=True).compute()
    nf_df = df.query(f"{'is_fraud'} == 0").drop(["baseline_pred", "is_fraud"], axis=1)
    f_df = df.query(f"{'is_fraud'} == 1").drop(["baseline_pred", "is_fraud"], axis=1)

    nf_df = nf_df.head(f_df.shape[0])
    f_df = f_df.head(f_df.shape[0])
    nf_losses: List[float] = []
    f_losses: List[float] = []

    for df, losses in [(nf_df, nf_losses), (f_df, f_losses)]:
        loss = MSELoss()
        for batch in df.itertuples(index=False):
            losses.append(__predict(model, batch, loss, parameters))

    df = pd.DataFrame(zip(nf_losses, f_losses), columns=["no_fraud_loss", "fraud_loss"])
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
