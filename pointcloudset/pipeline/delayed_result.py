from __future__ import annotations

from collections import UserList
from typing import Any

import dask


class DelayedResult(UserList):
    def __init__(self, data):
        self.data = data

    def compute(self) -> list:
        return list(dask.compute(*self.data))

    def __getitem__(self, pointcloud_number: slice | int) -> DelayedResult | Any:
        if isinstance(pointcloud_number, (slice, int)):
            return super().__getitem__(pointcloud_number).compute()
        else:
            raise TypeError(f"Wrong type {type(pointcloud_number).__name__}")
