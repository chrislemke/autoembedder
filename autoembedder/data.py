# -*- coding: utf-8 -*-

from typing import Any, Dict

import dask.dataframe as dd
from torch.utils.data import DataLoader, IterableDataset


class Dataset(IterableDataset):
    def __init__(self, source: str, drop_cat_columns: bool = False) -> None:
        """
        Args:
            source (str): path of the Dask dataframe
            drop_cat_columns (bool): whether to drop categorical columns

        Returns:
            None
        """
        super().__init__()
        self.df = dd.read_parquet(source, infer_divisions=True, engine="pyarrow")
        if drop_cat_columns:
            self.df = self.df.drop(
                self.df.columns[self.df.dtypes == "category"], axis=1
            )

    def __iter__(self) -> Any:
        return self.df.itertuples(index=False)

    def __getitem__(self, index: int) -> None:
        raise NotImplementedError


def dataloader(source: str, parameters: Dict) -> DataLoader:
    """
    Args:
        source (str): Path of the Dask dataframe
        parameters (Dict): Parameters for the DataLoader

    Returns:
        DataLoader: A DataLoader object
    """
    return DataLoader(
        dataset=Dataset(source, parameters["drop_cat_columns"] == 1),
        batch_size=parameters["batch_size"],
        pin_memory=parameters["pin_memory"] == 1,
        num_workers=parameters["num_workers"],
        drop_last=parameters["drop_last"] == 1,
    )
