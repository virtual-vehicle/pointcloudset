from __future__ import annotations

import datetime
from typing import List, Union

import dask.dataframe as dd
from dask import delayed
import dask

from .frame import Frame


class DatasetCore:
    def __init__(
        self,
        data: List[dask.delayed.DelayedLeaf],
        timestamps: List[datetime.datetime],
        meta: dict = {},
    ) -> None:
        self.data = data
        self.timestamps = timestamps
        self.meta = meta

    @property
    def start_time(self) -> datetime.datetime:
        return self.timestamps[0]

    @property
    def end_time(self) -> datetime.datetime:
        return self.timestamps[-1]

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)"""
        return len(self.data)

    def __str__(self):
        return f"Lidar Dataset with {len(self)} frame(s)"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

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
