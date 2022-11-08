from typing import Any, Dict, Optional, Union

import dask.dataframe as dd
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset


class _Dataset(IterableDataset):
    def __init__(
        self,
        source: Union[str, dd.DataFrame, pd.DataFrame],
        drop_cat_columns: bool = False,
    ) -> None:
        """
        Args:
            source (Union[str, dask.DataFrame, pandas.DataFrame]): Path to Dask/Pandas DataFrame stored as Parquet or a Dask/Pandas DataFrame.
            drop_cat_columns (bool): whether to drop categorical columns

        Returns:
            None
        """
        super().__init__()
        self.ddf = _Dataset.__data_from_source(source)
        if drop_cat_columns:
            self.ddf = self.ddf.drop(
                self.ddf.columns[self.ddf.dtypes == "category"], axis=1
            )

    def __iter__(self) -> Any:
        return self.ddf.itertuples(index=False)

    def __getitem__(self, index: int) -> None:
        raise NotImplementedError

    @staticmethod
    def __data_from_source(
        source: Union[str, dd.DataFrame, pd.DataFrame]
    ) -> dd.DataFrame:
        if isinstance(source, str):
            try:
                ddf = dd.read_parquet(source, infer_divisions=True, engine="pyarrow")
            except ValueError:
                ddf = dd.from_pandas(pd.read_parquet(source), npartitions=1)
        elif isinstance(source, dd.DataFrame):
            ddf = source
        elif isinstance(source, pd.DataFrame):
            ddf = dd.from_pandas(source, npartitions=1)
        else:
            raise ValueError("`source` must be a string or a Dask DataFrame!")
        return ddf


def dataloader(
    source: Union[str, dd.DataFrame, pd.DataFrame],
    parameters: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """
    Args:
        source (Union[str, dask.DataFrame, pandas.DataFrame]): Path to Dask/Pandas DataFrame stored as Parquet or a Dask/Pandas DataFrame.
        parameters (Optional[Dict[str, Any]]): Parameters for the DataLoader.
            In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.

    Returns:
        torch.utils.data.DataLoader: A DataLoader object
    """
    if parameters is None:
        parameters = {}

    return DataLoader(
        dataset=_Dataset(source, parameters.get("drop_cat_columns", 0) == 1),
        batch_size=parameters.get("batch_size", 32),
        pin_memory=parameters.get("pin_memory", 1) == 1,
        num_workers=parameters.get("num_workers", 0),
        drop_last=parameters.get("drop_last", 1) == 1,
    )
