from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union, get_type_hints, List, Literal

from dask import delayed
import pandas as pd

from lidar.dataset_core import DatasetCore
from lidar.frame import Frame
from lidar.io import DATASET_FROM_FILE, DATASET_FROM_INSTANCE, DATASET_TO_FILE
from lidar.pipeline.delayed_result import DelayedResult


def _is_pipline_returing_frame(pipeline, warn=True) -> bool:
    type_hints = get_type_hints(pipeline)
    res = False
    if "return" in type_hints:
        res = get_type_hints(pipeline)["return"] == Frame
    else:
        if warn:
            print(
                (
                    f"No return type was defined in {pipeline.__name__}:"
                    "will not return a new dataset"
                )
            )
    return res


class Dataset(DatasetCore):
    """
    Dataset Class which contains multiple frames, timestamps and metadata.
    For more details on how to use the Dataset Class please refer to the usage.ipynb
    notebook for an interactive tutorial. The notebook can also be found in the tutorial
    section of the docu.
    """

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
        """Reads a Dataset from a file.

        Args:
            file_path (pathlib.Path): File path where Dataset should be read from.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Dataset: Dataset object from file.

        Raises:
            ValueError: If file format is not supported.
            TypeError: If file_path is not a Path object.
        """
        if not isinstance(file_path, Path):
            raise TypeError("Expecting a Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext == "":
            ext = "DIR"
        if ext not in DATASET_FROM_FILE:
            raise ValueError(
                (
                    f"Unsupported file format {ext}; supported formats are:"
                    " {DATASET_FROM_FILE.keys()}"
                )
            )
        res = DATASET_FROM_FILE[ext](file_path, **kwargs)
        return cls(data=res["data"], timestamps=res["timestamps"], meta=res["meta"])

    def to_file(self, file_path: Path = Path(), **kwargs) -> None:
        """Writes a Dataset to a file.

        Args:
            file_path (pathlib.Path): File path where Dataset should be saved.
            **kwargs: Keyword arguments to pass to func.
        """
        DATASET_TO_FILE["DIR"](self, file_path=file_path, **kwargs)

    @classmethod
    def from_instance(
        cls,
        library: str,
        instance: list[Frame],
        **kwargs,
    ) -> Dataset:
        """Converts a libary instance to a lidar Dataset.

        Args:
            library (str): Name of the library.
            instance (list[Frame]): Instance from which to convert.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Dataset: Dataset object derived from the instance.

        Raises:
            ValueError: If instance is not supported.
        """
        library = library.upper()
        if library not in DATASET_FROM_INSTANCE:
            raise ValueError(
                "Unsupported library; supported libraries are: {}".format(
                    list(DATASET_FROM_INSTANCE)
                )
            )
        else:
            return cls(**DATASET_FROM_INSTANCE[library](instance, **kwargs))

    def apply(
        self,
        func: Union[Callable[[Frame], Frame], Callable[[Frame], Any]],
        warn: bool = True,
        **kwargs,
    ) -> Union[Dataset, DelayedResult]:
        """Applies a function to the dataset. It is also possible to pass keyword
        arguments.

        Args:
            func (Union[Callable[[Frame], Frame], Callable[[Frame], Any]]): Function to
                apply. If it returns a Frame and has the according type hint a new
                Dataset will be generated.
            warn (bool): If ``True`` warning if result is not a Dataset, if ``False``
                warning is turned off.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Union[Dataset, DelayedResult]: A Dataset if the function returns a Frame,
            otherwise a DelayedResult object which is a tuple of dask delayed objects.

        Examples:

            .. code-block:: python

                def func(frame:lidar.Frame) -> lidar.Frame:
                    return frame.limit(x,0,1)

                dataset.apply(func)
                # This results in a new Dataset

            .. code-block:: python

                def func(frame:lidar.Frame) -> float:
                    return frame.data.x.max()

                dataset.apply(func)

            .. code-block:: python

                def func(frame:lidar.Frame, test: float) -> float:
                    return frame.data.x.max() + test

                dataset.apply(func, test=10)
        """

        returns_frame = _is_pipline_returing_frame(func, warn=warn)

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

    def agg(
        self,
        agg: Union[str, list, dict],
        depth: Literal["dataset", "frame", "point"] = "dataset",
    ) -> Union[pd.Series, List[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """Aggregate using one or more operations over the whole dataset.
        Similar to pandas.DataFrame.aggregate(). Uses dask dataframes with
        parallel processing.

        Args:
            agg (Union[str, list, dict]): Function to use for aggregating.
            depth (Literal[, optional): Aggregation level: "dataset", "frame" or
                "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Raises:
            ValueError: If depth is not "dataset", "frame" or "point".

        Examples:

            .. code-block:: python

                dataset.agg("max", "frame")

            .. code-block:: python

                dataset.agg(["min","max","mean","std"])

            .. code-block:: python

                dataset.agg({"x" : ["min","max","mean","std"]})
        """
        if depth == "point":
            data = self._agg(agg).compute()
            if isinstance(agg, str):
                data.columns = [
                    i if i in ["N", "original_id"] else f"{i} {agg}"
                    for i in data.columns
                ]

            return data
        elif depth == "frame":
            return self._agg_per_frame(agg)
        elif depth == "dataset":
            data = self._agg(agg).compute()
            data = data.agg(agg)
            if not isinstance(agg, list):
                data.index = [f"{i} {agg}" for i in data.index]
            return data
        else:
            raise ValueError("depth needs to be dataset, frame or point")

    def min(self, depth: str = "dataset"):
        """Aggregate using min operation over the whole dataset.
        Similar to pandas.DataFrame.aggregate(). Uses dask dataframes with
        parallel processing.

        Args:
            depth (Literal[, optional): Aggregation level: "dataset", "frame" or
                "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Hint:

            See also: :func:`lidar.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["min"])
        """
        return self.agg("min", depth=depth)

    def max(self, depth: str = "dataset"):
        """Aggregate using max operation over the whole dataset.
        Similar to pandas.DataFrame.aggregate(). Uses dask dataframes with
        parallel processing.

        Args:
            depth (Literal[, optional): Aggregation level: "dataset", "frame" or
                "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Hint:

            See also: :func:`lidar.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["max"])
        """
        return self.agg("max", depth=depth)

    def mean(self, depth: str = "dataset"):
        """Aggregate using mean operation over the whole dataset.
        Similar to pandas.DataFrame.aggregate(). Uses dask dataframes with
        parallel processing.

        Args:
            depth (Literal[, optional): Aggregation level: "dataset", "frame" or
                "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Hint:

            See also: :func:`lidar.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["mean"])
        """
        return self.agg("mean", depth=depth)

    def std(self, depth: str = "dataset"):
        """Aggregate using std operation over the whole dataset.
        Similar to pandas.DataFrame.aggregate(). Uses dask dataframes with
        parallel processing.

        Args:
            depth (Literal[, optional): Aggregation level: "dataset", "frame" or
                "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Hint:

            See also: :func:`lidar.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["std"])
        """
        return self.agg("std", depth=depth)

    def _agg_per_frame(
        self, agg: Union[str, list, dict]
    ) -> Union[pd.DataFrame, list, pd.DataFrame]:
        def get(frame, agg: Union[str, list, dict]):
            return frame.data.agg(agg)

        res = self.apply(get, warn=False, agg=agg).compute()
        if isinstance(agg, list):
            return res
        else:
            res = pd.DataFrame(res)
            if not isinstance(agg, dict):
                res = res.drop("original_id", axis=1)
            res.columns = [f"{column} {agg}" for column in res.columns]
            res.index.name = "frame"
            res["timestamp"] = self.timestamps
            return res

    def extend(self, dataset: Dataset) -> Dataset:
        """Extends the dataset by another one.

        Args:
            dataset (Dataset): Dataset to extend another dataset.

        Returns:
            Dataset: Extended dataset.
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
