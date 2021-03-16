"""
# Dataset Class
The Dataset class which contains many frames.

For more details on how to use it please refer to the usage.ipynb Notebook for an interactive tuturial.

# Developer notes
* The important stuff happens in the __getitem__ method. Only then the rosbag is actually read with the help of
generators.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union, get_type_hints

import dask
from dask import delayed

from .dataset_core import DatasetCore
from .frame import Frame
from .io import DATASET_FROM_FILE, DATASET_TO_FILE


def _is_pipline_returing_frame(pipeline) -> bool:
    type_hints = get_type_hints(pipeline)
    res = False
    if "return" in type_hints:
        res = get_type_hints(pipeline)["return"] == Frame
    else:
        print(
            "No return type was defined of the pipeline: will not return a new datset"
        )
    return res


class Dataset(DatasetCore):
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
            raise TypeError("Expectinga Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext == "":
            ext = "DIR"
        if ext not in DATASET_FROM_FILE:
            raise ValueError(
                f"Unsupported file format {ext}; supported formats are: {DATASET_FROM_FILE.keys()}"
            )
        else:
            res = DATASET_FROM_FILE[ext](file_path, **kwargs)
            return cls(data=res["data"], timestamps=res["timestamps"], meta=res["meta"])

    def to_file(
        self,
        file_path: Path = Path(),
    ) -> None:
        DATASET_TO_FILE["DIR"](self, file_path=file_path)

    def apply(
        self,
        func: Union[Callable[[Frame], Frame], Callable[[Frame], Any]],
    ) -> Union[Dataset, tuple]:
        """Applies a function onto the dataset.

        Example:

        def func(frame:lidar.Frame) -> lidar.Frame:
            return frame.limit(x,0,1)

        dataset.apply(func)

        This results in a new dataset

        Example2:

        def func(frame:lidar.Frame) -> float:
            return frame.data.x.max()

        dataset.apply(func)

        Args:
            func (Union[Callable[[Frame], Frame], Callable[[Frame], Any]]): [description]

        Returns:
            Union[Dataset, Any]: A dataset if the function retunrs a Frame object or tuple
            with the results
        """

        returns_frame = _is_pipline_returing_frame(func)

        if returns_frame:

            def pipeline_delayed(element_in):
                frame = Frame(element_in)
                frame = func(frame)
                return frame.data

        else:

            def pipeline_delayed(element_in):
                frame = Frame(element_in)
                return func(frame)

        res = []
        for i in range(0, len(self)):
            item = delayed(pipeline_delayed)(self.data[i])
            res.append(item)

        if returns_frame:
            return Dataset(data=res, timestamps=self.timestamps, meta=self.meta)
        else:
            return dask.compute(*res)

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
