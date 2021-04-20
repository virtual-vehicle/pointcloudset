"""
FrameCore Class

With all the main methods and properties of the Frame Class.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import pyntcloud


class FrameCore:
    def __init__(
        self,
        data: pd.DataFrame,
        orig_file: str = "",
        timestamp: datetime = datetime.now(),
    ):
        """One Frame of lidar measurements.

        Example:
        testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
        testset = lidar.Dataset(testbag,topic="/os1_cloud_node/points",keep_zeros=False)
        testframe = testset[0]

        """
        self.data = data
        """All the data, x,y.z and intensity, range and more"""
        self.timestamp = timestamp
        """timestamp"""
        self.points = pyntcloud.PyntCloud(self.data, mesh=None)
        """Pyntcloud object with x,y,z coordinates"""
        self.orig_file = orig_file
        """Path to bag file. Defaults to empty"""

        self._check_index()

    @property
    def timestamp_str(self) -> str:
        """Converted ROS timestamp to human readable date and time.

        Returns:
            str: date time string
        """
        return self.timestamp.strftime("%A, %B %d, %Y %I:%M:%S")

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data argument must be a DataFrame")
        elif not set(["x", "y", "z"]).issubset(df.columns):
            raise ValueError("Data must have x, y and z coordinates")
        self._update_data(df)
        self._check_index()

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
        self.timestamp = datetime.now()
        self.points = pyntcloud.PyntCloud(self.__data[["x", "y", "z"]], mesh=None)
        self.orig_file = ""

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

    def _add_column(self, column_name: str, values: np.array) -> FrameCore:
        """Adding a new column with a scalar value to the data of the frame.

        Args:
            column_name (str): name of the new column.
            values (np.array): Values of the new column.
        """
        self.data[column_name] = values
        return self

    def _has_data(self) -> bool:
        """Check if lidar frame has data. Data here means point coordinates and
        measruments at each point of the pointcloud.

        Returns:
            bool: `True`` if the lidar frame contains data.
        """
        return not self.data.empty

    def has_original_id(self) -> bool:
        """Checks if orginal_id column is present in the data.
        Original_id identifies a lidar point and makes them coparable.

        Returns:
            bool: `True`` if the lidar frame contains orginal_id data.
        """
        return "original_id" in self.data.columns

    def _contains_original_id_number(self, original_id: int) -> bool:
        """Check if lidar frame contains a specific orginal_id.

        Args:
            original_id (int): the orginal_id to check

        Returns:
            bool: True if the original_id exists.
        """
        return original_id in self.data["original_id"].values

    def describe(self) -> pd.DataFrame:
        """Generate descriptive statistics based on .data.describe()."""
        return self.data.describe()

    def extract_point(self, id: int, use_original_id: bool = False) -> pd.DataFrame:
        """Extract a specific point from the Frame defined by the point id. The id
        can be the current index of the data from the Frame or the original_id.

        Args:
            id (int): Id number
            use_orginal_id (bool, optional): Use normal index or the orginal_id.
            Defaults to False.

        Returns:
            pd.DataFrame: a frame which only containse the defined points.
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
