"""
# Dataset Class
The Dataset class which contains many frames.

For more details on how to use it please refer to the usage.ipynb Notebook for an interactive tuturial.

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union, get_type_hints

from dask import delayed

from lidar.dataset_core import DatasetCore
from lidar.frame import Frame
from lidar.io import DATASET_FROM_FILE, DATASET_TO_FILE
from lidar.pipeline.delayed_result import DelayedResult


def _is_pipline_returing_frame(pipeline) -> bool:
    type_hints = get_type_hints(pipeline)
    res = False
    if "return" in type_hints:
        res = get_type_hints(pipeline)["return"] == Frame
    else:
        print(
            f"No return type was defined in {pipeline.__name__}: will not return a new dataset"
        )
    return res


class Dataset(DatasetCore):
    """Lidar Dataset which contains individual frames, timestamps and meta data."""

    def __getitem__(self, frame_number: Union[slice, int]) -> Union[DatasetCore, Frame]:
        if isinstance(frame_number, slice):
            data = self.data[frame_number]
            timestamps = self.timestamps[frame_number]
            meta = self.meta
            return Dataset(data, timestamps, meta)
        elif isinstance(frame_number, int):
            df = self.data[frame_number].compute()
            timestamp = self.timestamps[frame_number]
            return Frame(data=df, orig_file=self.meta["orig_file"], timestamp=timestamp)
        else:
            raise TypeError("Wrong type {}".format(type(frame_number).__name__))

    @classmethod
    def from_file(cls, file_path: Path, **kwargs):
        if not isinstance(file_path, Path):
            raise TypeError("Expecting a Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext == "":
            ext = "DIR"
        if ext not in DATASET_FROM_FILE:
            raise ValueError(
                f"Unsupported file format {ext}; supported formats are: {DATASET_FROM_FILE.keys()}"
            )
        res = DATASET_FROM_FILE[ext](file_path, **kwargs)
        return cls(data=res["data"], timestamps=res["timestamps"], meta=res["meta"])

    def to_file(self, file_path: Path = Path(), **kwargs) -> None:
        DATASET_TO_FILE["DIR"](self, file_path=file_path, **kwargs)

    def apply(
        self, func: Union[Callable[[Frame], Frame], Callable[[Frame], Any]], **kwargs
    ) -> Union[Dataset, DelayedResult]:
        """Applies a function onto the dataset. It is also possible to pass keyword
        arguments

        Example:

        def func(frame:lidar.Frame) -> lidar.Frame:
            return frame.limit(x,0,1)

        dataset.apply(func)

        This results in a new dataset

        Example2:

        def func(frame:lidar.Frame) -> float:
            return frame.data.x.max()

        dataset.apply(func)


        Example3:

        def func(frame:lidar.Frame, test: float) -> float:
            return frame.data.x.max() + test

        dataset.apply(func, test=10)

        Args:
            func (Union[Callable[[Frame], Frame], Callable[[Frame], Any]]): [description]

        Returns:
            Union[Dataset, DelayedResult]: A dataset if the function returns a Frame, or
            a DelayedResult object which is a tuple of dask delayed objects.
        """

        returns_frame = _is_pipline_returing_frame(func)

        if returns_frame:

            def pipeline_delayed(element_in, timestamp):
                frame = Frame(data=element_in, timestamp=timestamp)
                frame = func(frame, **kwargs)
                return frame.data

        else:

            def pipeline_delayed(element_in, timestamp):
                frame = Frame(data=element_in, timestamp=timestamp)
                return func(frame, **kwargs)

        res = []
        for i in range(0, len(self)):
            item = delayed(pipeline_delayed)(self.data[i], self.timestamps[i])
            res.append(item)

        if returns_frame:
            return Dataset(data=res, timestamps=self.timestamps, meta=self.meta)
        else:
            return DelayedResult(res)

    def extend(self, dataset: Dataset) -> Dataset:
        """Extends the dataset by another one.

        Args:
            dataset (Dataset): [description]

        Returns:
            Dataset: [description]
        """
        key = "extended"
        meta = self.meta
        if key in meta:
            new = meta[key]
            new.extend(dataset.meta)
            meta[key] = new
        else:
            meta[key] = [dataset.meta]
        self.data.extend(dataset.data)
        self.timestamps.extend(dataset.timestamps)
        self._check()
        return self
