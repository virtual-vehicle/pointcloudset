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

import numpy as np
import open3d as o3d
import pandas as pd
import rospy
from datetime import datetime
from pyntcloud import PyntCloud
import pyntcloud

from .convert import convert
from .plot.frame import plotly_3d, pyntcloud_3d


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

    def __str__(self):
        return f"pointcloud: with {len(self)} points, data:{list(self.data.columns)}, from {self.convert_timestamp()}"

    def __len__(self):
        return len(self.data)

    def get_open3d_points(self) -> o3d.open3d_pybind.geometry.PointCloud:
        """Extract points as open3D PointCloud object. Needed for processing with the
        open3d package.

        Returns:
            o3d.open3d_pybind.geometry.PointCloud: the pointcloud
        """
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
        """Update self.data removing points where filter is False.

        Args:
            boolean_array (np.ndarray): True where the point should remain.

        Returns:
            Frame: Frame with filterd rows and reindexed data and points.
        """
        new_data = self.data.loc[boolean_array].reset_index(drop=True)
        return Frame(new_data, timestamp=self.timestamp)

    def limit(self, dim: "str", minvalue: float, maxvalue: float):
        """Limit the range of certain values in lidar Frame. Can be chained together.

        Args:
            dim (str): dimension to limit, any column in data not just x, y, or z
            minvalue (float): min value to limit. (greater equal)
            maxvalue (float): max value to limit. (smaller equal)
        Returns:
            Frame: limited frame, were columns which did not match the criteria were dropped
        """
        if maxvalue < minvalue:
            raise ValueError("maxvalue must be greater than minvalue")
        bool_array = (
            (self.data[dim] <= maxvalue) & (self.data[dim] >= minvalue)
        ).to_numpy()
        return self.apply_filter(bool_array)

    def get_cluster(self, eps: float, min_points: int) -> pd.DataFrame:
        labels = np.array(
            self.get_open3d_points().cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=True
            )
        )
        labels_df = pd.DataFrame(labels, columns=["cluster"])
        return labels_df

    def take_cluster(self, cluster_number: int, cluster_labels: pd.DataFrame):
        bool_array = (cluster_labels["cluster"] == cluster_number).values
        return self.apply_filter(bool_array)

    def remove_outlier(self, nb_points: int, radius: float):
        pcd = self.get_open3d_points()
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return ind

    def quantile_filter(self, dim: str, cut_quantile: float):
        filter_array = (
            self.data[dim] >= self.data[dim].quantile(cut_quantile)
        ).to_numpy()
        return self.apply_filter(filter_array)

    def plane_segmentation(
        self, max_dist: float, max_iterations: int, n_inliers_to_stop=None
    ):
        self.points.add_scalar_field(
            "plane_fit",
            max_dist=max_dist,
            max_iterations=max_iterations,
            n_inliers_to_stop=n_inliers_to_stop,
        )

