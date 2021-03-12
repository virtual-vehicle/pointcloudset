from __future__ import annotations


import dask.dataframe as dd
from dask import delayed
from .frame import Frame
from typing import Union, List
import datetime


class DatasetCore:
    def __init__(
        self,
        data: dd.DataFrame,
        timestamps: List[datetime.datetime],
        meta: dict = {},
    ) -> None:
        self.data = data
        self.timestamps = timestamps
        self.meta = meta

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)"""
        return self.data.npartitions

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

    def __getitem__(self, frame_number: Union[slice, int]) -> Union[DatasetCore, Frame]:
        if isinstance(frame_number, slice):
            data = [delayed(self.data.get_partition(i)) for i in range(0, 10)]
            self.data = dd.from_delayed(data)
            self.timestamps = self.timestamps[frame_number]
            self.meta = self.meta
        elif isinstance(frame_number, int):
            df = self.data.get_partition(frame_number).compute()
            timestamp = self.timestamps[frame_number]
            return Frame(data=df, orig_file=self.meta["orig_file"], timestamp=timestamp)
        else:
            raise TypeError("Wrong type {}".format(type(frame_number).__name__))
