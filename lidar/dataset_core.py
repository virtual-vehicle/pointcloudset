from __future__ import annotations

import datetime
import warnings
from typing import List, Union

import dask
import pandas as pd
from dask.delayed import Delayed, DelayedLeaf

aggdict = {
    "mean": pd.core.groupby.groupby.Series.mean,
    "std": pd.core.groupby.groupby.Series.std,
    "max": pd.core.groupby.groupby.Series.max,
    "min": pd.core.groupby.groupby.Series.min,
    "sum": pd.core.groupby.groupby.Series.sum,
    "median": pd.core.groupby.groupby.Series.median,
    "count": pd.core.groupby.groupby.Series.count,
    "var": pd.core.groupby.groupby.Series.var,
    "sem": pd.core.groupby.groupby.Series.sem,
    "mad": pd.core.groupby.groupby.Series.mad,
    "skew": pd.core.groupby.groupby.Series.skew,
    "kurt": pd.core.groupby.groupby.Series.kurt,
}


class DatasetCore:
    def __init__(
        self,
        data: List[dask.delayed.DelayedLeaf] = [],
        timestamps: List[datetime.datetime] = [],
        meta: dict = {"orig_file": "", "topic": ""},
    ) -> None:
        self.data = data
        self.timestamps = timestamps
        self.meta = meta
        self._check()

    @property
    def start_time(self) -> datetime.datetime:
        return self.timestamps[0]

    @property
    def end_time(self) -> datetime.datetime:
        return self.timestamps[-1]

    @property
    def daskdataframe(self) -> dask.dataframe.core.DataFrame:
        return dask.dataframe.from_delayed(self.data)

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)"""
        return len(self.data)

    def __str__(self):
        return f"Lidar Dataset with {len(self)} frame(s)"

    def __repr__(self) -> str:
        return (
            f"""{self.__class__.__name__}({self.data},{self.timestamps},{self.meta})"""
        )

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self):
            raise StopIteration

        result = self[self.n]
        self.n += 1
        return result

    def agg(self, agg: str):
        data = self.daskdataframe.groupby("original_id").agg(agg)
        data["N"] = self.daskdataframe.groupby("original_id").size()
        data["original_id"] = data.index
        data = data.reset_index(drop=True)
        return data

    def _agg_explicit(
        self, agg_type: str, depth: int = 0, agg_type_depth1: str = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """A general aggreation function ob point or whole dataset level.

        Args:
            agg_type (str): [description]
            depth (int, optional): 0 means on point level and 1 toal dataset level.
            Defaults to 0.

        Returns:
            Union[pd.DataFrame, pd.Series]: [description]
        """
        data = self.agg(agg_type)
        if depth == 0:
            res = data.compute()
        elif depth == 1:
            if agg_type_depth1 is not None:
                func = aggdict[agg_type_depth1]
            else:
                func = aggdict[agg_type]
            data = data.drop(["N", "original_id"], axis=1)
            res = func(data.compute())
            res.name = agg_type
        return res

    def mean(self, depth: int = 0) -> Union[pd.DataFrame, pd.Series]:
        return self._agg_explicit(agg_type="mean", depth=depth)

    def std(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="std", depth=depth)

    def min(self, depth: int = 0) -> Union[pd.DataFrame, pd.Series]:
        return self._agg_explicit(agg_type="min", depth=depth)

    def max(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="max", depth=depth)

    def sum(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="sum", depth=depth)

    def median(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="median", depth=depth)

    def count(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="count", depth=depth, agg_type_depth1="sum")

    def var(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="var", depth=depth)

    def sem(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="sem", depth=depth)

    def mad(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="mad", depth=depth)

    def skew(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="skew", depth=depth)

    def kurt(self, depth: int = 0) -> pd.DataFrame:
        return self._agg_explicit(agg_type="kurt", depth=depth)

    def has_frames(self) -> bool:
        """Check if dataset has frames.

        Returns:
            bool: ``True`` if the dataset contains frames.
        """
        return len(self) > 0

    def get_frames_between_timestamps(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> DatasetCore:
        if not start_time < end_time:
            raise ValueError("start_time must be smaller than end_time")
        start_i = self._get_frame_number_from_time(start_time)
        end_i = self._get_frame_number_from_time(end_time) + 1
        return self[start_i:end_i]

    def _get_frame_number_from_time(self, time: datetime.datetime) -> int:
        """Get the frame number from a timestamp.

        Args:
            time (datetime.datetime): The time of interest

        Raises:
            ValueError: If time is outside of range.

        Returns:
            int: Frame number
        """
        if time < self.start_time or time > self.end_time:
            raise ValueError("time is outside of range")
        return min(
            range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - time)
        )

    def _check(self):
        assert "orig_file" in self.meta, "meta data does not contain orig_file"
        if len(self) > 0:
            assert len(self.timestamps) == len(
                self.data
            ), "Lenght of timestamps do not match the data"
            if any(
                self.timestamps[i] >= self.timestamps[i + 1]
                for i in range(len(self.timestamps) - 1)
            ):
                warnings.warn("Timestamps are not monotonic increasing")
            assert isinstance(
                self.data[0], (DelayedLeaf, Delayed)
            ), f"data needs to be a dask delayed object got {type(self.data[0])}"
