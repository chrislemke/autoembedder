# -*- coding: utf-8 -*-

from typing import Any, Dict, Union

import dask.dataframe as dd
from torch.utils.data import DataLoader, IterableDataset


class _Dataset(IterableDataset):
    def __init__(
        self, source: Union[str, dd.DataFrame], drop_cat_columns: bool = False
    ) -> None:
        """
        Args:
            source (Union[str, dd.DataFrame]): Path of the Dask dataframe or a Dask dataframe.
            drop_cat_columns (bool): whether to drop categorical columns

        Returns:
            None
        """
        super().__init__()
        if isinstance(source, str):
            self.ddf = dd.read_parquet(source, infer_divisions=True, engine="pyarrow")
        elif isinstance(source, dd.DataFrame):
            self.ddf = source
        else:
            raise ValueError("`source` must be a string or a Dask DataFrame!")
        if drop_cat_columns:
            self.ddf = self.ddf.drop(
                self.ddf.columns[self.ddf.dtypes == "category"], axis=1
            )

    def __iter__(self) -> Any:
        return self.ddf.itertuples(index=False)

    def __getitem__(self, index: int) -> None:
        raise NotImplementedError


def dataloader(source: Union[str, dd.DataFrame], parameters: Dict) -> DataLoader:
    """
    Args:
        source (Union[str, dd.DataFrame]): Path of the Dask dataframe or a Dask dataframe.
        parameters (Dict): Parameters for the DataLoader

    Returns:
        DataLoader: A DataLoader object
    """
    return DataLoader(
        dataset=_Dataset(source, parameters["drop_cat_columns"] == 1),
        batch_size=parameters["batch_size"],
        pin_memory=parameters["pin_memory"] == 1,
        num_workers=parameters["num_workers"],
        drop_last=parameters["drop_last"] == 1,
    )
