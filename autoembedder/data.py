#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import dask.dataframe as dd
from torch.utils.data import DataLoader, IterableDataset


class Dataset(IterableDataset):
    def __init__(self, source: str, drop_cat_columns: bool = False):
        super().__init__()
        self.df = dd.read_parquet(source, infer_divisions=True, engine="pyarrow")
        if drop_cat_columns:
            self.df = self.df.drop(
                self.df.columns[self.df.dtypes == "category"], axis=1
            )

    def __iter__(self):
        return self.df.itertuples(index=False)

    def __getitem__(self, index):
        raise NotImplementedError


def dataloader(source: str, parameters: Dict) -> DataLoader:
    return DataLoader(
        dataset=Dataset(source, parameters["drop_cat_columns"] == 1),
        batch_size=parameters["batch_size"],
        pin_memory=parameters["pin_memory"] == 1,
        num_workers=parameters["num_workers"],
        drop_last=parameters["drop_last"] == 1,
    )
