from __future__ import annotations

from collections import UserList
from typing import Any, Union

import dask


class DelayedResult(UserList):
    def __init__(self, data):
        self.data = data

    def compute(self) -> list:
        return list(dask.compute(*self.data))

    def __getitem__(self, frame_number: Union[slice, int]) -> Union[DelayedResult, Any]:
        if isinstance(frame_number, slice):
            return self[frame_number]
        elif isinstance(frame_number, int):
            return super().__getitem__(frame_number).compute()
        else:
            raise TypeError("Wrong type {}".format(type(frame_number).__name__))
