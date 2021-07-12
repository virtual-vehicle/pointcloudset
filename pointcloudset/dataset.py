from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Literal, Union, get_type_hints

import numpy as np
import pandas
from dask import delayed

from pointcloudset.dataset_core import DatasetCore
from pointcloudset.io import DATASET_FROM_FILE, DATASET_FROM_INSTANCE, DATASET_TO_FILE
from pointcloudset.pipeline.delayed_result import DelayedResult
from pointcloudset.pointcloud import PointCloud


def _is_pipline_returing_pointcloud(pipeline, warn=True) -> bool:
    type_hints = get_type_hints(pipeline)
    res = False
    if "return" in type_hints:
        res = get_type_hints(pipeline)["return"] == PointCloud
    elif warn:
        print(
            (
                f"No return type was defined in {pipeline.__name__}:"
                "will not return a new dataset"
            )
        )
    return res


class Dataset(DatasetCore):
    """
    Dataset Class which contains multiple pointclouds, timestamps and metadata.
    For more details on how to use the Dataset Class please refer to the usage.ipynb
    notebook for an interactive tutorial. The notebook can also be found in the tutorial
    section of the docu.
    """

    def __getitem__(
        self, pointcloud_number: Union[slice, int]
    ) -> Union[DatasetCore, PointCloud]:
        if isinstance(pointcloud_number, slice):
            data = self.data[pointcloud_number]
            timestamps = self.timestamps[pointcloud_number]
            meta = self.meta
            return Dataset(data, timestamps, meta)
        elif isinstance(pointcloud_number, int):
            df = self.data[pointcloud_number].compute()
            timestamp = self.timestamps[pointcloud_number]
            return PointCloud(
                data=df, orig_file=self.meta["orig_file"], timestamp=timestamp
            )
        else:
            raise TypeError("Wrong type {}".format(type(pointcloud_number).__name__))

    @classmethod
    def from_file(cls, file_path: Path, **kwargs):
        """Reads a Dataset from a file.
        For larger ROS bagfiles files use the commandline tool bag2dataset to convert the ROS bagfile
        beforehand.

        Supported are the native format which is a directore filled with fastparquet frames and
        ROS bag files (.bag).


        Args:
            file_path (pathlib.Path): File path where Dataset should be read from.\n
                If file format is a directory: :func:`pointcloudset.io.dataset.dir.dataset_from_dir`\n
                If file format is a ROS bag file: :func:`pointcloudset.io.dataset.bag.dataset_from_rosbag`
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Dataset: Dataset object from file.

        Raises:
            ValueError: If file format is not supported.
            TypeError: If file_path is not a Path object.
        """
        from_dir = False
        if not isinstance(file_path, Path):
            raise TypeError("Expecting a Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext == "":
            ext = "DIR"
            from_dir = True
        if ext not in DATASET_FROM_FILE:
            raise ValueError(
                (
                    f"Unsupported file format {ext}; supported formats are:"
                    " {DATASET_FROM_FILE.keys()}"
                )
            )
        res = DATASET_FROM_FILE[ext](file_path, **kwargs)
        meta = res["meta"]
        out = cls(data=res["data"], timestamps=res["timestamps"], meta=meta)
        if from_dir:
            out = out._replace_nan_frames_with_empty(res["empty_data"])
        return out

    def to_file(self, file_path: Path = Path(), **kwargs) -> None:
        """Writes a Dataset to a file.

        Supported is the native format which is a directory full of fastparquet files
        with meta data.

        Args:
            file_path (pathlib.Path): File path where Dataset should be saved.\n
                If file format is a directory: :func:`pointcloudset.io.dataset.dir.dataset_to_dir`
            **kwargs: Keyword arguments to pass to func.
        """
        DATASET_TO_FILE["DIR"](self, file_path=file_path, **kwargs)

    @classmethod
    def from_instance(
        cls,
        library: str,
        instance: list[PointCloud],
        **kwargs,
    ) -> Dataset:
        """Converts a library instance to a pointcloudset Dataset.

        Args:
            library (str): Name of the library.\n
                If "pointclouds": :func:`pointcloudset.io.dataset.pointclouds.dataset_from_pointclouds`
            instance (list[PointCloud]): Instance from which to convert.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Dataset: Dataset object derived from the instance.

        Raises:
            ValueError: If instance is not supported.

        Examples:

            .. code-block:: python

                pointcloudset.Dataset.from_instance("pointclouds", [pc1, pc2])

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
        func: Union[Callable[[PointCloud], PointCloud], Callable[[PointCloud], Any]],
        warn: bool = True,
        **kwargs,
    ) -> Union[Dataset, DelayedResult]:
        """Applies a function to the dataset. It is also possible to pass keyword
        arguments.

        Args:
            func (Union[Callable[[PointCloud], PointCloud], Callable[[PointCloud], Any]]): Function to
                apply. If it returns a PointCloud and has the according type hint a new
                Dataset will be generated.
            warn (bool): If ``True`` warning if result is not a Dataset, if ``False``
                warning is turned off.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Union[Dataset, DelayedResult]: A Dataset if the function returns a PointCloud,
            otherwise a DelayedResult object which is a tuple of dask delayed objects.

        Examples:

            .. code-block:: python

                def func(pointcloud:pointcloudset .PointCloud) -> pointcloudset.PointCloud:
                    return pointcloud.limit(x,0,1)

                dataset.apply(func)
                # This results in a new Dataset

            .. code-block:: python

                def func(pointcloud:pointcloudset.PointCloud) -> float:
                    return pointcloud.data.x.max()

                dataset.apply(func)

            .. code-block:: python

                def func(pointcloud:pointcloudset.PointCloud, test: float) -> float:
                    return pointcloud.data.x.max() + test

                dataset.apply(func, test=10)
        """

        returns_pointcloud = _is_pipline_returing_pointcloud(func, warn=warn)

        columns = list(self[0].data.columns)

        if returns_pointcloud:

            def pipeline_delayed(element_in, timestamp):
                pointcloud_in = PointCloud(data=element_in, timestamp=timestamp)
                pointcloud = func(pointcloud_in, **kwargs)
                if not pointcloud._has_data():
                    pointcloud = PointCloud(columns=columns)
                return pointcloud.data  # to generate an empty pointcloud

        else:

            def pipeline_delayed(element_in, timestamp):
                pointcloud = PointCloud(data=element_in, timestamp=timestamp)
                return func(pointcloud, **kwargs)

        res = []
        for i in range(0, len(self)):
            item = delayed(pipeline_delayed)(self.data[i], self.timestamps[i])
            res.append(item)

        if returns_pointcloud:
            return Dataset(data=res, timestamps=self.timestamps, meta=self.meta)
        else:
            return DelayedResult(res)

    def agg(
        self,
        agg: Union[str, list, dict],
        depth: Literal["dataset", "pointcloud", "point"] = "dataset",
    ) -> Union[
        pandas.Series, List[pandas.DataFrame], pandas.DataFrame, pandas.DataFrame
    ]:
        """Aggregate using one or more operations over the whole dataset.
        Similar to :meth:`pandas.DataFrame.aggregate`.
        Uses :class:`dask.dataframe.DataFrame` with parallel processing.


        Args:
            agg (Union[str, list, dict]): Function to use for aggregating.
            depth (Literal["dataset", "pointcloud", "point"], optional): Aggregation level: "dataset", "pointcloud" or
                "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Results of the
            aggregation. This can be a pandas DataFrame or Series, depending on the
            depth and aggregation.

        Raises:
            ValueError: If depth is not "dataset", "pointcloud" or "point".

        Examples:

            .. code-block:: python

                dataset.agg("max", "pointcloud")

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
        elif depth == "pointcloud":
            return self._agg_per_pointcloud(agg)
        elif depth == "dataset":
            data = self._agg(agg).compute()
            data = data.agg(agg)
            if not isinstance(agg, list):
                data.index = [f"{i} {agg}" for i in data.index]
            return data
        else:
            raise ValueError("depth needs to be dataset, pointcloud or point")

    def min(self, depth: str = "dataset"):
        """Aggregate using min operation over the whole dataset.
        Similar to :meth:`pandas.DataFrame.aggregate`.
        Uses :class:`dask.dataframe.DataFrame` with parallel processing.

        Args:
            depth (Literal["dataset", "pointcloud", "point"], optional): Aggregation level:
            "dataset", "pointcloud" or "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated
            Dataset.

        Examples:

            .. code-block:: python

                dataset.min()

            .. code-block:: python

                dataset.min("pointcloud")

            .. code-block:: python

                dataset.min("point")

        Hint:

            See also: :func:`pointcloudset.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["min"])
        """
        return self.agg("min", depth=depth)

    def max(self, depth: str = "dataset"):
        """Aggregate using max operation over the whole dataset.
        Similar to :meth:`pandas.DataFrame.aggregate`.
        Uses :class:`dask.dataframe.DataFrame` with parallel processing.

        Args:
            depth (Literal["dataset", "pointcloud", "point"], optional): Aggregation level:
            "dataset", "pointcloud" or "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated
            Dataset.

        Examples:

            .. code-block:: python

                dataset.max()

            .. code-block:: python

                dataset.max("pointcloud")

            .. code-block:: python

                dataset.max("point")

        Hint:

            See also: :func:`pointcloudset.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["max"])
        """
        return self.agg("max", depth=depth)

    def mean(self, depth: str = "dataset"):
        """Aggregate using mean operation over the whole dataset.
        Similar to :meth:`pandas.DataFrame.aggregate`.
        Uses :class:`dask.dataframe.DataFrame` with parallel processing.

        Args:
            depth (Literal["dataset", "pointcloud", "point"], optional): Aggregation level:
            "dataset", "pointcloud" or "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Examples:

            .. code-block:: python

                dataset.mean()

            .. code-block:: python

                dataset.mean("pointcloud")

            .. code-block:: python

                dataset.mean("point")

        Hint:

            See also: :func:`pointcloudset.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["mean"])
        """
        return self.agg("mean", depth=depth)

    def std(self, depth: str = "dataset"):
        """Aggregate using std operation over the whole dataset.
        Similar to :meth:`pandas.DataFrame.aggregate`.
        Uses :class:`dask.dataframe.DataFrame` with parallel processing.

        Args:
            depth (Literal["dataset", "pointcloud", "point"], optional): Aggregation level:
            "dataset", "pointcloud" or "point". Defaults to "dataset".

        Returns:
            Union[pandas.DataFrame, pandas.DataFrame, pandas.Series]: Aggregated Dataset.

        Examples:

            .. code-block:: python

                dataset.std()

            .. code-block:: python

                dataset.std("pointcloud")

            .. code-block:: python

                dataset.std("point")

        Hint:

            See also: :func:`pointcloudset.dataset.Dataset.agg`\n
            Same as:

            .. code-block:: python

                dataset.agg(["std"])
        """
        return self.agg("std", depth=depth)

    def _agg_per_pointcloud(
        self, agg: Union[str, list, dict]
    ) -> Union[pandas.DataFrame, list, pandas.DataFrame]:
        def get(pointcloud, agg: Union[str, list, dict]):
            return pointcloud.data.agg(agg)

        res = self.apply(get, warn=False, agg=agg).compute()
        if isinstance(agg, list):
            return res
        else:
            res = pandas.DataFrame(res)
            if not isinstance(agg, dict):
                res = res.drop("original_id", axis=1)
            res.columns = [f"{column} {agg}" for column in res.columns]
            res.index.name = "pointcloud"
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

    def _replace_empty_frames_with_nan(self, empty_data: pandas.DataFrame):
        """Function to replace empty pointclouds with pointclouds wiht 1 point with all
        nan values. Needed to save files with dask.
        """

        def _exchange_empty_pointclouds_with_nan(frame: PointCloud) -> PointCloud:
            if not frame._has_data():
                frame.data = empty_data
            return frame

        return self.apply(_exchange_empty_pointclouds_with_nan)

    def _replace_nan_frames_with_empty(self, empty_data: pandas.DataFrame):
        """Function to replace nan pointclouds with empty pointcouds
        Needed to after reading dataset files.
        """

        def _exchange_nan_pointclouds_with_empty(frame: PointCloud) -> PointCloud:
            if (len(frame) == 1) and np.allclose(empty_data.values, frame.data.values):
                frame = PointCloud(columns=frame.data.columns)
            return frame

        return self.apply(_exchange_nan_pointclouds_with_empty)
