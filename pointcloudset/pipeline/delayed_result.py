from __future__ import annotations

from collections import UserList
from typing import Any, Union

import dask


class DelayedResult(UserList):
    def __init__(self, data):
        self.data = data

    def compute(self) -> list:
        return list(dask.compute(*self.data))

    def __getitem__(
        self, pointcloud_number: Union[slice, int]
    ) -> Union[DelayedResult, Any]:
        if isinstance(pointcloud_number, (slice, int)):
            return super().__getitem__(pointcloud_number).compute()
        else:
            raise TypeError("Wrong type {}".format(type(pointcloud_number).__name__))
