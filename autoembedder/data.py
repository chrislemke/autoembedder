#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import dask.dataframe as dd
from torch.utils.data import DataLoader, IterableDataset


class PandasDataset(IterableDataset):
    def __init__(self, source: str):
        super().__init__()
        self.df = dd.read_parquet(source, infer_divisions=True, engine="pyarrow")

    def __iter__(self):
        return self.df.itertuples(index=False)

    def __getitem__(self, index):
        raise NotImplementedError


def pandas_dataloader(source: str, parameters: Dict) -> DataLoader:
    return DataLoader(
        dataset=PandasDataset(source),
        batch_size=parameters["batch_size"],
        pin_memory=parameters["pin_memory"] == 1,
        num_workers=parameters["num_workers"],
        drop_last=parameters["drop_last"] == 1,
    )
