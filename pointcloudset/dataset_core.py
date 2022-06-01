from __future__ import annotations

import datetime
from typing import Union
import pandas as pd

import dask
from dask.delayed import Delayed, DelayedLeaf


class DatasetCore:
    """
    DatasetCore Class with all the main methods and properties of the Dataset Class.
    """

    def __init__(
        self,
        data: list[dask.delayed.DelayedLeaf] = [],
        timestamps: list[datetime.datetime] = [],
        meta: dict = {"orig_file": "", "topic": ""},
    ) -> None:
        self.data = data
        self.timestamps = timestamps
        self.meta = meta
        self._check()

    @property
    def start_time(self) -> datetime.datetime:
        """
        Returns:
            datetime.datetime: Time of first pointcloud in Dataset.
        """
        return self.timestamps[0]

    @property
    def end_time(self) -> datetime.datetime:
        """
        Returns:
            datetime.datetime: Time of last pointcloud in Dataset.
        """
        return self.timestamps[-1]

    @property
    def duration(self) -> datetime.timedelta:
        """
        Returns:
            datetime.timedelta: Duration between first and last
            pointcloud in Dataset.
        """
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def daskdataframe(self) -> dask.dataframe.core.DataFrame:
        """
        Returns:
            dask.dataframe.DataFrame: Dask DataFrame with data of Dataset.
        """
        return dask.dataframe.from_delayed(self.data)

    @property
    def bounding_box(self) -> pd.DataFrame:
        """The axis aligned boundary box of the whole dataset as a :class:`pandas.DataFrame`."""

        def bb(pc):
            return pc.bounding_box

        list_of_bb = self.apply(bb, warn=False).compute()
        bb_all_df = pd.concat(list_of_bb)
        return pd.DataFrame([bb_all_df.min(), bb_all_df.max()], index=["min", "max"])

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)"""
        return len(self.data)

    def __str__(self):
        return f"Lidar Dataset with {len(self)} pointcloud(s)"

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

    def _agg(self, agg: Union[str, list, dict]) -> dask.dataframe.DataFrame:
        """Aggregate using one or more operations over the whole dataset.
            Similar to pandas agg. Used dask dataframes with parallel processing.

        Example:
            .. code-block:: python

                dataset.agg("max")
                datset.agg(["min","max","mean","std"])
                datset.agg({"x" : ["min","max","mean","std"]})

        Args:
            agg (Union[str, list, dict]): [description]

        Returns:
            dask.dataframe.DataFrame: use .compute to get final result.
        """
        if self.has_original_id:
            data = self.daskdataframe.groupby("original_id").agg(agg)
            data["N"] = self.daskdataframe.groupby("original_id").size()
            data["original_id"] = data.index
            data = data.reset_index(drop=True)
        else:
            data = self.daskdataframe
            data["dummy"] = data.index
            data = data.groupby("dummy").agg(agg)
        return data

    def has_pointclouds(self) -> bool:
        """Check if Dataset has PointCloud.

        Returns:
            bool: ``True`` if the Dataset does contain PointClouds, ``False`` if Dataset
            does not contain any PointClouds.
        """
        return len(self) > 0

    def get_pointclouds_between_timestamps(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> DatasetCore:
        """Select pointclouds between start_time and end_time.

        Args:
            start_time (datetime.datetime): Timestamp of first pointcloud.
            end_time (datetime.datetime): Timestamp of last pointcloud.

        Returns:
            Dataset: Dataset with PointClouds between two timestamps.

        Raises:
            ValueError: If start_time is bigger than end_time.
        """
        if start_time >= end_time:
            raise ValueError("start_time must be smaller than end_time")
        start_i = self._get_pointcloud_number_from_time(start_time)
        end_i = self._get_pointcloud_number_from_time(end_time) + 1
        return self[start_i:end_i]

    def _get_pointcloud_number_from_time(self, time: datetime.datetime) -> int:
        """Get the pointcloud number from a timestamp.

        Args:
            time (datetime.datetime): The time of interest.

        Returns:
            int: PointCloud number.

        Raises:
            ValueError: If time is outside of range.
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
            ), f"Length of timestamps {len(self.timestamps)} do not match the data {len(self.data)}"

            if not pd.Series(self.timestamps).is_monotonic_increasing:
                raise ValueError("Timestamps are not monotonic increasing")

            assert isinstance(
                self.data[0], (DelayedLeaf, Delayed)
            ), f"data needs to be a dask delayed object got {type(self.data[0])}"
