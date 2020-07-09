"""
Frame Class.

For one lidar measurement frame. Typically an automotive lidar records many frames per
second.

One Frame consists mainly of pyntcloud pointcloud (.pointcloud) and a pandas dataframe
(.data) with all the associated data.

Note that the index of the poits is not preserved when applying processing. This
is necessary since pyntcloud does not allow to pass the index. Therfore, a new
Frame object is generated at each processing stage.


All operations have to act on both, pointcloud and data and keep the timestamp.
"""

import operator
import warnings
from datetime import datetime
from typing import List, Union, Type

import numpy as np
import open3d as o3d
import pandas as pd
import pyntcloud
import rospy

from .convert import convert
from .plot.frame import plotly_3d, pyntcloud_3d

ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


class Frame:
    "One Frame of lidar measurements."

    def __init__(
        self, data: pd.DataFrame, timestamp: rospy.rostime.Time = rospy.rostime.Time()
    ):
        self.data = data
        """All the data, x,y.z and intensity, range and more"""
        self.timestamp = timestamp
        """ROS timestamp"""
        self.points = pyntcloud.PyntCloud(self.data[["x", "y", "z"]], mesh=None)
        """Pyntcloud object with x,y,z coordinates"""
        self.measurments = self.data.drop(["x", "y", "z"], axis=1)
        """Measurments aka. scalar field of values at each point"""
        self._check_index()

    def __str__(self):
        return f"pointcloud: with {len(self)} points, data:{list(self.data.columns)}, from {self.convert_timestamp()}"

    def __len__(self):
        return len(self.data)

    def describe(self):
        """Generate descriptive statistics based on .data.describe().
        """
        return self.data.describe()

    def get_open3d_points(self) -> o3d.open3d_pybind.geometry.PointCloud:
        """Extract points as open3D PointCloud object. Needed for processing with the
        open3d package.

        Returns:
            o3d.open3d_pybind.geometry.PointCloud: the pointcloud
        """
        converted = convert.convert_df2pcd(self.points.points)
        assert len(np.asarray(converted.points)) == len(
            self
        ), "len of open3d points should be the same as len of the Frame"
        return convert.convert_df2pcd(self.points.points)

    def convert_timestamp(self) -> str:
        """Convert ROS timestamp to human readable date and time.

        Returns:
            str: date time string
        """
        return datetime.fromtimestamp(self.timestamp.to_sec()).strftime(
            "%A, %B %d, %Y %I:%M:%S"
        )

    def has_data(self) -> bool:
        """Check if lidar frame has data. Data here means point coordinates and
        measruments at each point of the pointcloud.

        Returns:
            bool: `True`` if the lidar frame contains measurment data.
        """
        return not self.data.empty

    def distances_to_origin(self) -> np.array:
        """For each point in the pointcloud calculate the euclidian distance
        to the origin (0,0,0).

        Returns:
            np.array: List of distances for each point
        """
        point_a = np.array((0.0, 0.0, 0.0))
        points = self.points.xyz
        dists = [np.linalg.norm(point_a - point) for point in points]
        return np.array(dists)

    def plot_interactive(
        self, backend: str = "plotly", color: str = "intensity", **kwargs
    ):
        args = locals()
        args.update(kwargs)
        backend = args.pop("backend")
        if backend == "pyntcloud":
            return pyntcloud_3d(self, **kwargs)
        elif backend == "plotly":
            return plotly_3d(self, color=color, **kwargs)
        else:
            raise ValueError("wrong backend")

    def apply_filter(self, boolean_array: np.ndarray):
        """Generating a new Frame by removing points where filter is False.
        Usefull for pyntcloud generate boolean arrays and by filtering DataFrames.

        Args:
            boolean_array (np.ndarray): True where the point should remain.

        Returns:
            Frame: Frame with filterd rows and reindexed data and points.
        """
        new_data = self.data.loc[boolean_array].reset_index(drop=True)
        return Frame(new_data, timestamp=self.timestamp)

    def select_by_index(self, index_to_keep: List[int]):
        """Generating a new Frame by keeping which are in the same idex
        Usefull for open3d generate index lists. Similar to the the select_by_index
        function of open3d.

        Args:
            index_to_keep (List[int]): List of indices to keep

        Returns:
            Frame: Frame with keeped rows and reindexed data and points
        """
        new_data = self.data.iloc[index_to_keep].reset_index(drop=True)
        return Frame(new_data, timestamp=self.timestamp)

    def limit(self, dim: "str", minvalue: float, maxvalue: float):
        """Limit the range of certain values in lidar Frame. Can be chained together.

        Args:
            dim (str): dimension to limit, any column in data not just x, y, or z
            minvalue (float): min value to limit. (greater equal)
            maxvalue (float): max value to limit. (smaller equal)
        Returns:
            Frame: limited frame, were columns which did not match the criteria were 
            dropped.
        """
        if maxvalue < minvalue:
            raise ValueError("maxvalue must be greater than minvalue")
        bool_array = (
            (self.data[dim] <= maxvalue) & (self.data[dim] >= minvalue)
        ).to_numpy()
        return self.apply_filter(bool_array)

    def get_cluster(self, eps: float, min_points: int) -> pd.DataFrame:
        """Get the clusters based on open3D cluster_dbscan. Process futher with
            take_cluster.

        Args:
            eps (float): Density parameter that is used to find neighbouring points.
            min_points (int): Minimum number of points to form a cluster.

        Returns:
            pd.DataFrame: Dataframe with list of clusters.
        """
        labels = np.array(
            self.get_open3d_points().cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )
        labels_df = pd.DataFrame(labels, columns=["cluster"])
        return labels_df

    def take_cluster(self, cluster_number: int, cluster_labels: pd.DataFrame):
        """Takes only the points belonging to the cluster_number.

        Args:
            cluster_number (int): Cluster id to keep.
            cluster_labels (pd.DataFrame): clusters generated with get_cluster.

        Returns:
            Frame: with cluster of ID cluster_number.
        """
        bool_array = (cluster_labels["cluster"] == cluster_number).values
        return self.apply_filter(bool_array)

    def remove_radius_outlier(self, nb_points: int, radius: float):
        """    Function to remove points that have less than nb_points in a given 
        sphere of a given radius Parameters.
        Args:
            nb_points (int) – Number of points within the radius.
            radius (float) – Radius of the sphere.
        Returns:
            Tuple[open3d.geometry.PointCloud, List[int]] :
        """
        pcd = self.get_open3d_points()
        cl, index_to_keep = pcd.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        return self.select_by_index(index_to_keep)

    def quantile_filter(
        self, dim: str, relation: str = ">=", cut_quantile: float = 0.5
    ):
        """Filtering based on quantile values of dimension dim of the data.

        testframe.quantile_filter("intensity","==",0.5)

        Args:
            dim (str): column in data, for example "intensity"
            relation (str, optional): Any operator as string. Defaults to ">=".
            cut_quantile (float, optional): Qunatile to compare to. Defaults to 0.5.

        Returns:
            Frame: Frame which fullfils the criteria.
        """
        cut_value = self.data[dim].quantile(cut_quantile)
        filter_array = ops[relation](self.data[dim], cut_value)
        return self.apply_filter(filter_array.to_numpy())

    def plane_segmentation(
        self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        return_plane_model: bool = False,
    ):
        """Segments a plane in the point cloud using the RANSAC algorithm.
        Based on open3D plane segmentation.

        Args:
            distance_threshold (float): Max distance a point can be from the plane model, and still be considered an inlier.
            ransac_n (int):  Number of initial points to be considered inliers in each iteration.
            num_iterations (int): Number of iterations.
            return_plane_model (bool, optional): Return also plane model parameters. Defaults to False.

        Returns:
            Frame or dict: Frame with inliers or a dict of Frame with inliers and the plane parameters.
        """
        pcd = self.get_open3d_points()
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(self) > 200:
            warnings.warn(
                """Might not produce reproducable resuts, If the number of points
                is high. Try to reduce the area of interesst before using
                plane_segmentation. Caused by open3D."""
            )
        inlier_Frame = self.select_by_index(inliers)
        if return_plane_model:
            return {"Frame": inlier_Frame, "plane_model": plane_model}
        else:
            return inlier_Frame

    def _check_index(self):
        """A private function to check if the index of the self.data is sane.
        """
        if len(self) > 0:
            assert self.data.index[0] == 0, "index should start with 0"
            assert self.data.index[-1] + 1 == len(
                self
            ), "index should be as long as the data"
            assert (
                self.data.index.is_monotonic_increasing
            ), "index should be monotonic increasing"
