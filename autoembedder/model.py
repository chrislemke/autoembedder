# -*- coding: utf-8 -*-

from typing import Dict, List, NamedTuple, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import orthogonal


def embedded_sizes_and_dims(
    train_df: dd.DataFrame, test_df: dd.DataFrame, col_collections: List[List[str]]
) -> List[Tuple[int, int]]:
    """
    This method iterates over the columns of the dataframe. For every column it checks if the `col_collections` contains a list
    with additional columns. If this is the case the unique values are collected from both columns. Afterwards the values and
    dimensions are calculated.

    Args:
        train_df (dd.DataFrame): Training Dask DataFrame used to create the list of sizes and dimensions.
        test_df (dd.DataFrame): Validation Dask DataFrame used to create the list of sizes and dimensions. Both dataframes will be concatenated.
        col_collections (List[List[str]]): A list of lists of strings. It must contain lists of columns which include same values.
    Returns:
        List[Tuple[int, int]]: Each tuple contains the number of values and the dimensions for the corresponding embedding layer.
    """
    assert (
        train_df.columns == test_df.columns
    ).all(), "Columns of both dataframes must be the same!"
    df = dd.concat([train_df, test_df]).compute()
    df = df.select_dtypes(include="category")

    unique = []
    for column in df.columns:
        for col_collection in col_collections:
            if column in col_collection:
                unique += [max(np.unique(df[col_collection].values))]

    return [(int(v + 1), int(min(50, (v + 1) // 2))) for v in unique]


def num_cont_columns(df: dd.DataFrame) -> int:
    return len(df.select_dtypes(exclude=["category"]).columns)


def model_input(
    batch: NamedTuple, parameters: Dict
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Since the `Autoembedder` expects that the continues values and the categorical values are passed by
    different arguments this function splits the batch by type. It works with a batch of `torch.Tensor` and with floats
    and ints.

    Args:
        batch (NamedTuple): Batch provided by dataset.
        parameters (Dict): Parameters for the model.
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: The first item contains the categorical values, the second item the continues values.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and parameters["use_mps"] == 1
        else "cpu"
    )
    cat = []
    cont = []
    if isinstance(batch[0], torch.Tensor):
        for feature in batch:
            if feature.dtype in [torch.int32, torch.int64]:
                cat.append(feature)
            elif feature.dtype in [torch.float32, torch.float64]:
                if (
                    feature.dtype == torch.float64
                    and torch.backends.mps.is_available()
                    and parameters["use_mps"] == 1
                ):
                    feature = feature.to(torch.float32)
                cont.append(feature)
            else:
                raise ValueError(f"Unsupported dtype: {feature.dtype}!")
        if not cat:
            return torch.empty((1, 0), dtype=torch.int8), torch.stack(cont, 0).to(
                device
            )
        return torch.stack(cat, 0).to(device), torch.stack(cont, 0).to(device)

    # Used if `batch` does not contains tensors.
    for feature in batch:
        if isinstance(feature, int):
            cat.append(feature)
        elif isinstance(feature, float):
            cont.append(feature)
        else:
            raise ValueError(f"Unsupported type: {type(feature)}!")

    cat = [torch.tensor(cat)]
    cont = [torch.tensor(cont)]
    return torch.stack(cat, 0).to(device), torch.stack(cont, 0).to(device)


class Autoembedder(nn.Module):
    def __init__(
        self,
        config: Dict,
        num_cont_features: int,
        embedding_sizes: List[Tuple[int, int]],
    ) -> None:
        """
        Args:
            config (Dict): JSON config file for the model. When `hidden_layers` is not empty `num_hidden_layers` will be ignored. Otherwise the number of units for the hidden layers will be calculated. `exponent_addition` are used in the `linear_layers` function. Check the documentation below for more information.
            num_cont_features (int): Number of continues features.
            embedding_sizes (List[Tuple[int, int]]): List of tuples. Each tuple contains the size of the dictionary (unique values) of embeddings and the size of each embedding vector.

        Returns:
            None
        """
        super().__init__()
        print(f"Model config: {config}")

        self.last_target: Optional[torch.Tensor] = None
        self.code_value: Optional[torch.Tensor] = None
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(t[0], t[1]) for t in embedding_sizes]
        )
        (
            self.encoder_input,
            self.encoder_hidden_layers,
            self.decoder_hidden_layers,
            self.decoder_output,
        ) = self.__linear_layers(config, num_cont_features)

        print(f"Model `in_features`: {self.encoder_input.in_features}")

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat (torch.Tensor): Tensor including the categorical values. Shape: [columns count, batch size]
            x_cont (torch.Tensor): Tensor including the continues values. Shape: [columns count, batch size]
        Returns:
            torch.Tensor :Output of the 'Autoembedder'. It contains the concatenated and processed continues and categorical data.
        """

        x_cont = rearrange(x_cont, "c r -> r c")

        x_emb = []
        for i, layer in enumerate(self.embeddings):
            value = x_cat[i].int()
            try:
                x_emb.append(layer(value))
            except IndexError:
                value = max(value.tolist())
                raise IndexError(  # pylint: disable=W0707
                    f"""
                There seems to be a problem with the index of the embedding layer: `index out of range in self`. The `num_embeddings`
                of the {i}. layer is {layer.num_embeddings}. The maximum value which should be embedded from the tensor is {value}.
                If the value ({value}) is bigger than the `num_embeddings` ({layer.num_embeddings})
                the embeddings layer can not embed the value. This could have multiple reasons:
                1. Check if the `--cat_columns` argument is a correct representation of the dataframe. Maybe it contain columns
                    which are not a part of the actual dataframe.
                2. Check if the shape of the embeddings layer no. {i} is correct.
                3. Check if the correct data is passed to the model.
                """
                )
        if self.embeddings:
            x_emb = torch.cat(x_emb, 1)
            x = torch.cat([x_cont, x_emb], 1)
        else:
            x = x_cont
        self.last_target = (
            x.clone().detach()
        )  # Concatenated x values - used with the custom loss function: `AutoEmbLoss`.

        x = self.__encode(x)
        self.code_value = x.clone().detach()  # Stores the values of the code layer.
        return self.__decode(x)

    def init_xavier_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def __encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.encoder_input(x))
        for layer in self.encoder_hidden_layers:
            x = self.leaky_relu(layer(x))
        return x

    def __decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_hidden_layers:
            x = self.leaky_relu(layer(x))
        return self.leaky_relu(self.decoder_output(x))

    def __linear_layers(
        self, config: Dict, num_cont_features: int
    ) -> Tuple[nn.Module, nn.ModuleList, nn.ModuleList, nn.Module]:
        """
        Args:
            config (Dict): Configuration containing the hidden layer structure of the model.
        Returns:
            A tuple containing the linear layers.
        """

        sum_emb_dims = sum(emb.embedding_dim for emb in self.embeddings)
        in_features = num_cont_features + sum_emb_dims

        hl = config["hidden_layers"]
        encoder_input = orthogonal(
            nn.Linear(in_features, hl[0][0], bias=config["layer_bias"] == 1)
        )
        encoder_hidden_layers = nn.ModuleList(
            [
                orthogonal(
                    nn.Linear(hl[x][0], hl[x][1], bias=config["layer_bias"] == 1)
                )
                for x in range(len(hl))
            ]
        )

        decoder_hidden_layers = nn.ModuleList([])
        for x in reversed(range(len(hl))):
            layer = nn.Linear(hl[x][1], hl[x][0], bias=False)
            layer.weight = nn.Parameter(encoder_hidden_layers[x].weight.t())
            decoder_hidden_layers.append(layer)

        decoder_output = nn.Linear(hl[0][0], in_features, bias=False)
        decoder_output.weight = nn.Parameter(encoder_input.weight.t())

        return (
            encoder_input,
            encoder_hidden_layers,
            decoder_hidden_layers,
            decoder_output,
        )
