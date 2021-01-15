"""
# Frame Class.

For one lidar measurement frame. Typically an automotive lidar records many frames per
second.

One Frame consists mainly of pyntcloud pointcloud (.points) and a pandas dataframe
(.data) with all the associated data.

Note that the index of the points is not preserved when applying processing. This
is necessary since pyntcloud does not allow to pass the index. Therefore, a new
Frame object is generated at each processing stage.

## Developer notes:
* All operations have to act on both, pointcloud and data and keep the timestamp.
* All processing methods need to return another Frame.
"""
from __future__ import annotations

import operator
import warnings
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import plotly
import plotly.express as px

from .frame_core import FrameCore
from .geometry import plane
from .plot.frame import plot_overlay

ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


def is_documented_by(original):
    """A decorator to get the docstring from anoter function."""

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


class Frame(FrameCore):
    def plot(
        self,
        color: Union[None, str] = None,
        overlay: dict = {},
        point_size: float = 2,
        prepend_id: str = "",
        hover_data: List[str] = [],
        **kwargs,
    ) -> plotly.graph_objs._figure.Figure:
        """Plot a Frame as a 3D scatter plot with plotly. It handles plots of single
        frames and overlay with other objects, such as other frames from clustering or
        planes from plane segmentation.

        You can also pass arguments to the plotly express function scatter_3D.

        Args:
            frame (Frame): the frame to plot
            color (str or None): Which column to plot. For example "intensity"
            overlay (dict, optional): Dict with of rames to overlay
                {"Cluster 1": cluster1,"plan1 1": plane_model}
            point_size (float, optional): Size of each point. Defaults to 2.
            prepend_id (str, optional): string before point id to display in hover
            hover data (list, optional): data columns to display in hover. Default is
                all of them.

        Raises:
            ValueError: if the color column name is not in the data

        Returns:
            Plotly plot: The interactive plotly plot, best used inside a jupyter
            notebook.
        """
        if color is not None and color not in self.data.columns:
            raise ValueError(f"choose any of {list(self.data.columns)} or None")

        ids = [prepend_id + "id=" + str(i) for i in range(0, self.data.shape[0])]

        if hover_data == []:
            hover_data = self.data.columns

        if not all([x in self.data.columns for x in hover_data]):
            raise ValueError(f"choose a list of {list(self.data.columns)} or []")

        fig = px.scatter_3d(
            self.data,
            x="x",
            y="y",
            z="z",
            color=color,
            hover_name=ids,
            hover_data=hover_data,
            title=self.timestamp_str,
            **kwargs,
        )
        fig.update_traces(
            marker=dict(size=point_size, line=dict(width=0)),
            selector=dict(mode="markers"),
        )

        if len(overlay) > 0:
            fig = plot_overlay(fig, self, overlay)

        fig.update_layout(
            scene_aspectmode="data",
        )
        return fig

    def calculate_single_point_difference(
        self, frameB: Frame, original_id: int
    ) -> pd.DataFrame:
        """Calculate the difference of one element of a Point in the current Frame to
        the correspoing point in Frame B. Both frames must contain the same orginal_id.

        Args:
            frameB (Frame): Frame which contains the point to comapare to.
            original_id (int): Orginal ID of the point.

        Returns:
            pd.DataFrame: A single row DataFrame with the differences (A - B).
        """
        pointA = self.extract_point(original_id, use_orginal_id=True)
        try:
            pointB = frameB.extract_point(original_id, use_orginal_id=True)
            difference = pointA - pointB
        except IndexError:
            # there is no point with the orignal_id in frameB
            difference = pointA
            difference.loc[:] = np.nan
        difference = difference.drop(["original_id"], axis=1)
        difference.columns = [f"{column} difference" for column in difference.columns]
        difference["original_id"] = pointA["original_id"]
        return difference

    def calculate_all_point_differences(self, frameB: Frame) -> Frame:
        """Calculate the point differences for each point which is also in frameB. Only
        points with the same orginal_id are compared. The results are added to the data
        of the frame. (frame - frameB)

        Args:
            frameB (Frame): A Frame object to compute the differences.

        Raises:
            ValueError: If there are no points in FrameB with the same orginal_id
        """
        refrence_orginial_ids = self.data.original_id.values
        frameB_original_ids = frameB.data.original_id.values
        intersection = np.intersect1d(refrence_orginial_ids, frameB_original_ids)
        if len(intersection) > 0:
            diff_list = [
                self.calculate_single_point_difference(frameB, id)
                for id in intersection
            ]
            orginal_types = [str(types) for types in diff_list[0].dtypes.values]
            target_type_dict = dict(zip(diff_list[0].columns.values, orginal_types))
            diff_df = pd.concat(diff_list)
            diff_df = diff_df.astype(target_type_dict)
            diff_df = diff_df.reset_index(drop=True)
            self.data = self.data.merge(diff_df, on="original_id", how="left")
            return self
        else:
            raise ValueError("no intersection found between the frames.")

    def calculate_distance_to_plane(
        self, plane_model: np.array, absolute_values: bool = True
    ) -> Frame:
        """Calculates the distance of each point to a plane and adds it as a column
        to the data of the frame. Uses the plane equation a x + b y + c z + d = 0

        Args:
            plane_model (np.array): [a, b, c, d], could be provided by
                plane_segmentation.
            absolute_values (bool, optional): Calculate absolute distances if True.
                Defaults to True.
        """
        points = self.points.xyz
        distances = np.asarray(
            [plane.distance_to_point(point, plane_model) for point in points]
        )
        if absolute_values:
            distances = np.absolute(distances)
        plane_str = np.array2string(
            plane_model, formatter={"float_kind": lambda x: "%.4f" % x}
        )
        self._add_column(f"distance to plane: {plane_str}", distances)
        return self

    def calculate_distance_to_origin(self) -> Frame:
        """For each point in the pointcloud calculate the euclidian distance
        to the origin (0,0,0). Adds a new column to the data with the values.
        """
        point_a = np.array((0.0, 0.0, 0.0))
        points = self.points.xyz
        distances = np.array([np.linalg.norm(point_a - point) for point in points])
        self._add_column("distance to origin", distances)
        return self

    def apply_filter(self, boolean_array: np.ndarray) -> Frame:
        """Generating a new Frame by removing points where filter is False.
        Usefull for pyntcloud generate boolean arrays and by filtering DataFrames.

        Args:
            boolean_array (np.ndarray): True where the point should remain.

        Returns:
            Frame: Frame with filterd rows and reindexed data and points.
        """
        new_data = self.data.loc[boolean_array].reset_index(drop=True)
        return Frame(new_data, timestamp=self.timestamp)

    def limit(self, dim: "str", minvalue: float, maxvalue: float) -> Frame:
        """Limit the range of certain values in lidar Frame. Can be chained together.

        Example:

        testframe.limit("x", -1.0, 1.0).limit("intensity", 0.0, 50.0)

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
            self._get_open3d_points().cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )
        labels_df = pd.DataFrame(labels, columns=["cluster"])
        return labels_df

    def take_cluster(self, cluster_number: int, cluster_labels: pd.DataFrame) -> Frame:
        """Takes only the points belonging to the cluster_number.

        Args:
            cluster_number (int): Cluster id to keep.
            cluster_labels (pd.DataFrame): clusters generated with get_cluster.

        Returns:
            Frame: with cluster of ID cluster_number.
        """
        bool_array = (cluster_labels["cluster"] == cluster_number).values
        return self.apply_filter(bool_array)

    def remove_radius_outlier(self, nb_points: int, radius: float) -> FrameCore:
        """Function to remove points that have less than nb_points in a given
        sphere of a given radius Parameters.
        Args:
            nb_points (int) – Number of points within the radius.
            radius (float) – Radius of the sphere.
        Returns:
            Tuple[open3d.geometry.PointCloud, List[int]] :
        """
        pcd = self._get_open3d_points()
        cl, index_to_keep = pcd.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        return self._select_by_index(index_to_keep)

    def quantile_filter(
        self, dim: str, relation: str = ">=", cut_quantile: float = 0.5
    ) -> Frame:
        """Filtering based on quantile values of dimension dim of the data.

        Example:

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
    ) -> Union[dict, FrameCore]:
        """Segments a plane in the point cloud using the RANSAC algorithm.
        Based on open3D plane segmentation.

        Args:
            distance_threshold (float): Max distance a point can be from the plane
                        model, and still be considered an inlier.
            ransac_n (int):  Number of initial points to be considered inliers in
                        each iteration.
            num_iterations (int): Number of iterations.
            return_plane_model (bool, optional): Return also plane model parameters.
                        Defaults to False.

        Returns:
            Frame or dict: Frame with inliers or a dict of Frame with inliers and the
            plane parameters.
        """
        pcd = self._get_open3d_points()
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
        inlier_Frame = self._select_by_index(inliers)
        if return_plane_model:
            return {"Frame": inlier_Frame, "plane_model": plane_model}
        else:
            return inlier_Frame

    def to_csv(self, path: Path = Path()) -> None:
        """Exports the frame as a csv for use with cloud compare or similar tools.
        Args:
            path (Path, optional): Destination. Defaults to the folder of
            the bag fiile with the timestamp of the frame.
        """
        orig_file_name = Path(self.orig_file).stem
        if path == Path():
            filename = f"{orig_file_name}_timestamp_{self.timestamp}.csv"
            destination_folder = Path(self.orig_file).parent.joinpath(filename)
        else:
            destination_folder = path
        self.data.to_csv(destination_folder, index=False)
