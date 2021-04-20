"""
Frame Class

For one lidar measurement frame. Typically an automotive lidar records many frames per
second.

One Frame consists mainly of pyntcloud pointcloud (.points) and a pandas dataframe
(.data) with all the associated data.

Note that the index of the points is not preserved when applying processing. This
is necessary since pyntcloud does not allow to pass the index. Therefore, a new
Frame object is generated at each processing stage.

Developer notes:
* All operations have to act on both, pointcloud and data and keep the timestamp.
* All processing methods need to return another Frame.
"""
from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import open3d as o3d
import pandas as pd
import plotly
import plotly.express as px
import pyntcloud

from lidar.diff import ALL_DIFFS
from lidar.filter import ALL_FILTERS
from lidar.frame_core import FrameCore
from lidar.io import (
    FRAME_FROM_FILE,
    FRAME_FROM_INSTANCE,
    FRAME_TO_FILE,
    FRAME_TO_INSTANCE,
)
from lidar.plot.frame import plot_overlay


def is_documented_by(original):
    """A decorator to get the docstring from anoter function."""

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


class Frame(FrameCore):
    """One lidar frame."""

    @classmethod
    def from_file(cls, file_path: Path, **kwargs):
        """Extract data from file and construct a Frame with it. Uses Pynthcloud as
        backend.

        Args:
            file_path (Path): pathlib Path of file to read

        Raises:
            ValueError: For unsupported files

        Returns:
            Frame: lidar frame with timestamp last modified.
        """
        if not isinstance(file_path, Path):
            raise TypeError("Expectinga Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext not in FRAME_FROM_FILE:
            raise ValueError(
                "Unsupported file format; supported formats are: {}".format(
                    list(FRAME_FROM_FILE)
                )
            )
        file_path_str = file_path.as_posix()
        timestamp = datetime.datetime.utcfromtimestamp(file_path.stat().st_mtime)
        pyntcloud_in = pyntcloud.PyntCloud.from_file(file_path_str, **kwargs)
        return cls(
            data=pyntcloud_in.points, orig_file=file_path_str, timestamp=timestamp
        )

    def to_file(self, file_path: Path = Path(), **kwargs) -> None:
        """Exports the frame as a csv for use with cloud compare or similar tools.
        Currently not all attributes of a frame are saved so some information is lost
        when using this function.

        Args:
            file_path (Path, optional): Destination. Defaults to the folder of
            the bag file and csv with the timestamp of the frame.
        """
        ext = file_path.suffix[1:].upper()
        if ext not in FRAME_TO_FILE:
            raise ValueError(
                "Unsupported file format; supported formats are: {}".format(
                    list(FRAME_TO_FILE)
                )
            )

        orig_file_name = Path(self.orig_file).stem
        if file_path == Path():
            # defaulting to csv file
            filename = f"{orig_file_name}_timestamp_{self.timestamp}.csv"
            destination_folder = Path(self.orig_file).parent.joinpath(filename)
        else:
            destination_folder = file_path

        kwargs["file_path"] = destination_folder
        kwargs["frame"] = self

        FRAME_TO_FILE[ext](**kwargs)

    @classmethod
    def from_instance(
        cls,
        library: str,
        instance: Union[pd.DataFrame, pyntcloud.PyntCloud, o3d.geometry.PointCloud],
        **kwargs,
    ) -> Frame:
        """Converts a libaries instance to a lidar Frame.

        Args:
            library (str): name of the libary
            instance (Union[ pd.DataFrame, pyntcloud.PyntCloud,
                o3d.open3d_pybind.geometry.PointCloud ]): [description]

        Raises:
            ValueError: If instance is not supported.

        Returns:
            Frame: derived from the instance
        """
        library = library.upper()
        if library not in FRAME_FROM_INSTANCE:
            raise ValueError(
                "Unsupported library; supported libraries are: {}".format(
                    list(FRAME_FROM_INSTANCE)
                )
            )
        else:
            return cls(**FRAME_FROM_INSTANCE[library](instance, **kwargs))

    def to_instance(
        self, library: str, **kwargs
    ) -> Union[pd.DataFrame, pyntcloud.PyntCloud, o3d.geometry.PointCloud]:
        """Convert Frame to another librarie instance.

        Args:
            library (str): name of the libary

        Raises:
            ValueError: If libary is not suppored

        Returns:
            Union[ pd.DataFrame, pyntcloud.PyntCloud, open3d.geometry.PointCloud ]:
            The derived instance
        """
        library = library.upper()
        if library not in FRAME_TO_INSTANCE:
            raise ValueError(
                "Unsupported library; supported libraries are: {}".format(
                    list(FRAME_TO_INSTANCE)
                )
            )

        return FRAME_TO_INSTANCE[library](self, **kwargs)

    def plot(
        self,
        color: Union[None, str] = None,
        overlay: dict = {},
        point_size: float = 2,
        prepend_id: str = "",
        hover_data: List[str] = None,
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
            hover data (list(str), optional): data columns to display in hover.
            Default is None.

        Raises:
            ValueError: if the color column name is not in the data

        Returns:
            Plotly plot: The interactive plotly plot, best used inside a jupyter
            notebook.
        """
        if color is not None and color not in self.data.columns:
            raise ValueError(f"choose any of {list(self.data.columns)} or None")

        ids = [prepend_id + "id=" + str(i) for i in range(self.data.shape[0])]

        show_hover = True
        if hover_data is None:
            show_hover = False
        elif isinstance(hover_data, list) & len(hover_data) > 0:
            default = ["original_id"]
            hover_data = list(set(default + hover_data))
            if any(x not in self.data.columns for x in hover_data):
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

        if overlay:
            fig = plot_overlay(
                fig,
                self,
                overlay,
                hover_data=hover_data,
                **kwargs,
            )
        fig.update_layout(scene_aspectmode="data")
        if not show_hover:
            fig.update_layout(hovermode=False)

        return fig

    def diff(
        self, name: str, target: Union[None, Frame, np.ndarray] = None, **kwargs
    ) -> Frame:
        """Calculate differences and distances to the origin, plane, point and frame.

        Args:
            name (str): "orgin", "plane", "frame", "point"
            target (Union[None, Frame, np.ndarray], optional): [description].
                Defaults to None,

        Raises:
            ValueError: If name is not supported.

        Returns:
            Frame: New frame with added column of the differences
        """
        if name in ALL_DIFFS:
            ALL_DIFFS[name](frame=self, target=target, **kwargs)
            return self
        else:
            raise ValueError("Unsupported diff. Check docstring")

    def filter(self, name: str, *args, **kwargs) -> Frame:
        name = name.upper()
        if name in ALL_FILTERS:
            return ALL_FILTERS[name](self, *args, **kwargs)
        else:
            raise ValueError("Unsupported filter. Check docstring")

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
        return self.filter("value", dim, ">=", minvalue).filter(
            "value", dim, "<=", maxvalue
        )

    def apply_filter(self, filter_result: Union[np.ndarray, List[int]]) -> Frame:
        """Generating a new Frame by removing points where filter

        Args:
            filter_result (Union[np.ndarray, List[int]]): Filter result

        Raises:
            TypeError: If the filter_result has the wrong type

        Returns:
            Frame: rame with filterd rows and reindexed data and points.
        """
        if isinstance(filter_result, np.ndarray):
            # dataframe and pyntcloud based filters
            new_data = self.data.loc[filter_result].reset_index(drop=True)
        elif isinstance(filter_result, list):
            # from open3d filters
            new_data = self.data.iloc[filter_result].reset_index(drop=True)
        else:
            raise TypeError(
                (
                    "Wrong filter_result expeciting array with boolean values or"
                    "list of intices"
                )
            )
        return Frame(new_data, timestamp=self.timestamp)

    def get_cluster(self, eps: float, min_points: int) -> pd.DataFrame:
        """Get the clusters based on open3D cluster_dbscan. Process further with
            take_cluster.

        Args:
            eps (float): Density parameter that is used to find neighbouring points.
            min_points (int): Minimum number of points to form a cluster.

        Returns:
            pd.DataFrame: Dataframe with list of clusters.
        """
        labels = np.array(
            self.to_instance("open3d").cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )
        return pd.DataFrame(labels, columns=["cluster"])

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

    def plane_segmentation(
        self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        return_plane_model: bool = False,
    ) -> Union[Frame, dict]:
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
        pcd = self.to_instance("open3d")
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
        inlier_Frame = self.apply_filter(inliers)
        if return_plane_model:
            return {"Frame": inlier_Frame, "plane_model": plane_model}
        else:
            return inlier_Frame
