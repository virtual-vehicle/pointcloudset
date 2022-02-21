from __future__ import annotations

import traceback
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import pyntcloud


class PointCloudCore:
    """
    PointCloudCore Class with all the main methods and properties of the
    PointCloud Class.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        orig_file: str = "",
        timestamp: datetime = None,
        columns: list = ["x", "y", "z"],
    ):
        self.timestamp = datetime.now() if timestamp is None else timestamp
        """Timestamp."""
        self.orig_file = orig_file
        """Path to orginal file. Defaults to empty."""

        if data is None:
            # "empty" PointCloud with one line of all nans. This is necessary in order
            # to save datasets with dask see issue#6
            values = np.repeat(np.nan, len(columns))
            empty_data = pd.DataFrame([values], columns=columns)
            self.data = empty_data
            self.data = self.data.drop([0])
        else:
            self.data = data
        """The data as a pandas DataFrame."""

        with warnings.catch_warnings():
            # ignore warnings produced by pyntcloud when the pointcloud is empty
            warnings.simplefilter("ignore")
            self.points = pyntcloud.PyntCloud(self.data, mesh=None)
        """PyntCloud object with x,y,z coordinates."""
        self._check_index()

    @property
    def timestamp_str(self) -> str:
        """Timestamp to human readable date and time string.

        Returns:
            str: Date/time string.
        """
        return self.timestamp.strftime("%A, %B %d, %Y %I:%M:%S")

    @property
    def data(self):
        """All the data, x,y,z and auxiliary data such as intensity, range and more."""
        return self.__data

    @data.setter
    def data(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data argument must be a DataFrame")
        elif not set(["x", "y", "z"]).issubset(df.columns):
            raise ValueError("Data must have x, y and z coordinates")
        self._update_data(df)
        self._check_index()

    @property
    def bounding_box(self) -> pd.DataFrame:
        """The axis aligned boundary box as a :class:`pandas.DataFrame`."""
        return self.data[["x", "y", "z"]].agg(["min", "max"])

    @property
    def centroid(self) -> np.array:
        """Geometric center for the point cloud."""
        return self.points.centroid

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.data}, "
            f"{self.timestamp}, {self.orig_file})"
        )

    def __str__(self) -> str:
        return (
            f"pointcloud: with {len(self)} points, data:{list(self.data.columns)},"
            f" from {self.timestamp_str}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, id: Union[int, slice]) -> pd.DataFrame:
        if isinstance(id, slice):
            return self.data.iloc[id]
        elif isinstance(id, int):
            return self.data.iloc[[id]]
        else:
            raise TypeError("Wrong type {}".format(type(id).__name__))

    def _update_data(self, df: pd.DataFrame):
        """Utility function. Implicitly called when self.data is assigned."""
        self.__data = df
        self.points = pyntcloud.PyntCloud(self.__data[["x", "y", "z"]], mesh=None)

    def _check_index(self):
        """A private function to check if the index of self.data is sane."""
        if len(self) > 0:
            assert self.data.index[0] == 0, "index should start with 0"
            assert self.data.index[-1] + 1 == len(
                self
            ), "index should be as long as the data"
            assert (
                self.data.index.is_monotonic_increasing
            ), "index should be monotonic increasing"

    def _add_column(self, column_name: str, values: np.array) -> PointCloudCore:
        """Adding a new column with a scalar value to the data of the pointcloud.

        Args:
            column_name (str): name of the new column.
            values (numpy.ndarray): Values of the new column.
        """
        self.data[column_name] = values
        return self

    def _has_data(self) -> bool:
        """Check if pointcloudset pointcloud has data. Data here means point coordinates and
        measurements at each point of the pointcloud.

        Returns:
            bool: ``True`` if the pointcloudset pointcloud contains data.
        """
        return not self.data.empty

    @property
    def has_original_id(self) -> bool:
        """Checks if original_id column is present in the data.
        Original_id identifies a lidar point and makes them comparable.

        Returns:
            bool: ``True`` if the PointCloud contains original_id data, ``False`` if PointCloud
            does not contain original_id data.
        """
        return "original_id" in self.data.columns

    def _contains_original_id_number(self, original_id: int) -> bool:
        """Check if pointcloudset pointcloud contains a specific original_id.

        Args:
            original_id (int): the original_id to check

        Returns:
            bool: ``True`` if the original_id exists.
        """
        return original_id in self.data["original_id"].values

    def describe(self) -> pd.DataFrame:
        """Generate descriptive statistics based on PointCloud.data.describe() and therefore on
        :meth:`pandas:pandas.DataFrame.describe`.

        Returns:
            pandas.DataFrame: Summary statistics of the data of the PointCloud.
        """
        return self.data.describe()

    def extract_point(self, id: int, use_original_id: bool = False) -> pd.DataFrame:
        """Extract a specific point from the PointCloud defined by the point id. The id
        can be the current index of the data from the PointCloud or the original_id.

        Args:
            id (int): ID number
            use_original_id (bool, optional): If ``True`` use original_id, if ``False``
                use current index in PointCloud. Defaults to ``False``.

        Returns:
            pandas.DataFrame: A pointcloud which only contains the defined point.
        """
        try:
            if use_original_id:
                point = self.data[self.data["original_id"] == id]
                if len(point) == 0:
                    raise IndexError
                elif len(point) != 1:
                    raise Exception()
            else:
                point = self[id]
        except IndexError:
            raise IndexError(f"point with {id} does note exist.")
        except Exception:
            print(traceback.print_exc())
        return point.reset_index(drop=True)
