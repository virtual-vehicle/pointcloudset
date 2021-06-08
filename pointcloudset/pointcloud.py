from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import open3d
import pandas
import plotly
import plotly.express as px
import pyntcloud

from pointcloudset.diff import ALL_DIFFS
from pointcloudset.filter import ALL_FILTERS
from pointcloudset.io import (POINTCLOUD_FROM_FILE, POINTCLOUD_FROM_INSTANCE,
                              POINTCLOUD_TO_FILE, POINTCLOUD_TO_INSTANCE)
from pointcloudset.plot.pointcloud import plot_overlay
from pointcloudset.pointcloud_core import PointCloudCore


class PointCloud(PointCloudCore):
    """
    PointCloud Class with one pointcloud of lidar measurements, laser scanning,
    photogrammetry  or simular.

    One PointCloud consists mainly of `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_
    pointcloud
    (`PyntCloud.points <https://pyntcloud.readthedocs.io/en/latest/points.html#points>`_)
    and a pandas.DataFrame (.data) with all the associated data.

    Note that the index of the points is not preserved when applying processing. This
    is necessary since `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_
    does not allow to pass the index. Therefore, a new PointCloud object is generated at
    each processing stage.

    Developer notes:
        * All operations have to act on both, pointcloud and data and keep the timestamp.
        * All processing methods need to return another PointCloud.

    Examples:

        .. code-block:: python

            testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
            testset = pointcloudset.Dataset(testbag,topic="/os1_cloud_node/points",
                keep_zeros=False)
            testpointcloud = testset[0]
    """

    @classmethod
    def from_file(cls, file_path: Path, **kwargs):
        """Extract data from file and construct a PointCloud with it. Uses
        `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_ as
        backend.

        Args:
            file_path (pathlib.Path): Path of file to read.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            PointCloud: PointCloud with timestamp last modified.

        Raises:
            ValueError: If file format is not supported.
            TypeError: If file_path is no Path object.
        """
        if not isinstance(file_path, Path):
            raise TypeError("Expecting a Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext not in POINTCLOUD_FROM_FILE:
            raise ValueError(
                "Unsupported file format; supported formats are: {}".format(
                    list(POINTCLOUD_FROM_FILE)
                )
            )
        file_path_str = file_path.as_posix()
        timestamp = datetime.datetime.utcfromtimestamp(file_path.stat().st_mtime)
        pyntcloud_in = pyntcloud.PyntCloud.from_file(file_path_str, **kwargs)
        return cls(
            data=pyntcloud_in.points, orig_file=file_path_str, timestamp=timestamp
        )

    def to_file(self, file_path: Path = Path(), **kwargs) -> None:
        """Exports the pointcloud as to a file for use with `CloudCompare <https://www.danielgm.net/cc/ake>`_ or similar tools.
        Currently not all attributes of a pointcloud are saved so some information is lost
        when using this function.
        Uses `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_ as
        backend.

        Args:
            file_path (pathlib.Path, optional): Destination. Defaults to the folder of
                the bag file and csv with the timestamp of the pointcloud.
            **kwargs: Keyword arguments to pass to func.

        Raises:
            ValueError: If file format is not supported.
        """
        ext = file_path.suffix[1:].upper()
        if ext not in POINTCLOUD_TO_FILE:
            raise ValueError(
                "Unsupported file format; supported formats are: {}".format(
                    list(POINTCLOUD_TO_FILE)
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
        kwargs["pointcloud"] = self

        POINTCLOUD_TO_FILE[ext](**kwargs)

    @classmethod
    def from_instance(
        cls,
        library: Literal["PYNTCLOUD", "OPEN3D", "DATAFRAME", "PANDAS"],
        instance: Union[
            pandas.DataFrame, pyntcloud.PyntCloud, open3d.geometry.PointCloud
        ],
        **kwargs,
    ) -> PointCloud:
        """Converts a library instance to a pointcloudset PointCloud.

        Args:
            library (str): Name of the library.\n
                If PYNTCLOUD: :func:`pointcloudset.io.pointcloud.pyntcloud.from_pyntcloud`\n
                If OPEN3D: :func:`pointcloudset.io.pointcloud.open3d.from_open3d`\n
                If DATAFRAME: :func:`pointcloudset.io.pointcloud.pandas.from_dataframe`\n
                If PANDAS: :func:`pointcloudset.io.pointcloud.pandas.from_dataframe`
            instance
                (Union[pandas.DataFrame, pyntcloud.PyntCloud, open3d.geometry.PointCloud]):
                Library instance to convert.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            PointCloud: Derived from the instance.

        Raises:
            ValueError: If instance is not supported.

        Examples:

            .. code-block:: python

                testpointcloud = from_instance("OPEN3D", open3d_pointcloud)

        """
        library = library.upper()
        if library not in POINTCLOUD_FROM_INSTANCE:
            raise ValueError(
                "Unsupported library; supported libraries are: {}".format(
                    list(POINTCLOUD_FROM_INSTANCE)
                )
            )
        else:
            return cls(**POINTCLOUD_FROM_INSTANCE[library](instance, **kwargs))

    def to_instance(
        self, library: Literal["PYNTCLOUD", "OPEN3D", "DATAFRAME", "PANDAS"], **kwargs
    ) -> Union[pandas.DataFrame, pyntcloud.PyntCloud, open3d.geometry.PointCloud]:
        """Convert PointCloud to another library instance.

        Args:
            library (str): Name of the library.\n
                If PYNTCLOUD: :func:`pointcloudset.io.pointcloud.pyntcloud.to_pyntcloud`\n
                If OPEN3D: :func:`pointcloudset.io.pointcloud.open3d.to_open3d`\n
                If DATAFRAME: :func:`pointcloudset.io.pointcloud.pandas.to_dataframe`\n
                If PANDAS: :func:`pointcloudset.io.pointcloud.pandas.to_dataframe`
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Union[ pandas.DataFrame, pyntcloud.PyntCloud, open3d.geometry.PointCloud ]:
            The derived instance.

        Raises:
            ValueError: If library is not suppored.

        Examples:

            .. code-block:: python

                open3d_pointcloud = testpointcloud.to_instance("OPEN3D")
        """
        library = library.upper()
        if library not in POINTCLOUD_TO_INSTANCE:
            raise ValueError(
                "Unsupported library; supported libraries are: {}".format(
                    list(POINTCLOUD_TO_INSTANCE)
                )
            )

        return POINTCLOUD_TO_INSTANCE[library](self, **kwargs)

    def plot(
        self,
        color: Union[None, str] = None,
        overlay: dict = {},
        point_size: float = 2,
        prepend_id: str = "",
        hover_data: List[str] = None,
        **kwargs,
    ) -> plotly.graph_objs.Figure:
        """Plot a PointCloud as a 3D scatter plot with `Plotly <https://plotly.com/>`_.
        It handles plots of single pointclouds and overlay with other objects, such as
        other pointclouds from clustering or planes from plane segmentation.

        You can also pass arguments to the `Plotly <https://plotly.com/>`_
        express function :func:`plotly.express.scatter_3d`.

        Args:
            pointcloud (PointCloud): The pointcloud to plot.
            color (str or None): Which column to plot. For example "intensity".
                Defaults to None.
            overlay (dict, optional): Dict with PointClouds to overlay.
                {"Cluster 1": cluster1,"Plane 1": plane_model}\n
                See also: :func:`pointcloudset.plot.pointcloud.plot_overlay`\n
                Defaults to empty.
            point_size (float, optional): Size of each point. Defaults to 2.
            prepend_id (str, optional): String before point id to display in hover.
                Defaults to empty.
            hover_data (list(str), optional): Data columns to display in hover.
                Defaults to None.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            plotly.graph_objs.Figure: The interactive Plotly plot, best used inside a
            Jupyter Notebook.

        Raises:
            ValueError: If the color column name is not in the data.
        """
        if color is not None and color not in self.data.columns:
            raise ValueError(f"choose any of {list(self.data.columns)} or None")

        ids = [prepend_id + "id=" + str(i) for i in range(self.data.shape[0])]

        show_hover = True
        if hover_data is None:
            show_hover = False
        elif isinstance(hover_data, list) & len(hover_data) > 0:
            if self.has_original_id:
                default = ["original_id"]
                hover_data = hover_data + default
            hover_data = list(set(hover_data))
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
        self,
        name: Literal["origin", "plane", "pointcloud", "point"],
        target: Union[None, PointCloud, np.ndarray] = None,
        **kwargs,
    ) -> PointCloud:
        """Calculate differences and distances to the origin, plane, point and pointcloud.

        Args:
            name (str):
                "origin": :func:`pointcloudset.diff.origin.calculate_distance_to_origin` \n
                "plane": :func:`pointcloudset.diff.plane.calculate_distance_to_plane` \n
                "pointcloud": :func:`pointcloudset.diff.pointcloud.calculate_distance_to_pointcloud` \n
                "point": :func:`pointcloudset.diff.point.calculate_distance_to_point` \n
            target (Union[None, PointCloud, numpy.ndarray], optional): Pass argument
                according to chosen object. Defaults to None.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            PointCloud: New PointCloud with added column of the differences.

        Raises:
            ValueError: If name is not supported.

        Examples:

            .. code-block:: python

                newpointcloud = testpointcloud.diff("pointcloud", targetpointcloud)
        """
        if name in ALL_DIFFS:
            ALL_DIFFS[name](pointcloud=self, target=target, **kwargs)
            return self
        else:
            raise ValueError("Unsupported diff. Check docstring")

    def filter(
        self, name: Literal["quantile", "value", "radiusoutlier"], *args, **kwargs
    ) -> PointCloud:
        """Filters a PointCloud according to criteria.

        Args:
            name (str):
                "quantile": :func:`pointcloudset.filter.stat.quantile_filter` \n
                "value": :func:`pointcloudset.filter.stat.value_filter` \n
                "radiusoutlier": :func:`pointcloudset.filter.stat.remove_radius_outlier` \n
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            PointCloud: PointCloud which fullfils the criteria.

        Raises:
            ValueError: If name is not supported.

        Examples:

            .. code-block:: python

                filteredpointcloud = testpointcloud.filter("quantile","intensity","==",0.5)

            .. code-block:: python

                filteredpointcloud = testpointcloud.filter("value","intensity",">",100)
        """
        name = name.upper()
        if name in ALL_FILTERS:
            return ALL_FILTERS[name](self, *args, **kwargs)
        else:
            raise ValueError("Unsupported filter. Check docstring")

    def limit(self, dim: "str", minvalue: float, maxvalue: float) -> PointCloud:
        """Limit the range of certain values in pointcloudset PointCloud. Can be chained together.

        Args:
            dim (str): Dimension to limit, any column in data not just x, y, or z.
            minvalue (float): Min value to limit. (greater equal)
            maxvalue (float): Max value to limit. (smaller equal)

        Returns:
            PointCloud: Limited pointcloud, where columns which did not match the criteria were
            dropped.

        Examples:

            .. code-block:: python

                limitedpointcloud = testpointcloud.limit("x", -1.0, 1.0).limit("intensity", 0.0, 50.0)
        """
        if maxvalue < minvalue:
            raise ValueError("maxvalue must be greater than minvalue")
        return self.filter("value", dim, ">=", minvalue).filter(
            "value", dim, "<=", maxvalue
        )

    def apply_filter(self, filter_result: Union[np.ndarray, List[int]]) -> PointCloud:
        """Generating a new PointCloud by removing points according to a call of the
        filter method.

        Args:
            filter_result (Union[numpy.ndarray, List[int]]): Filter result.

        Returns:
            PointCloud: PointCloud with filtered rows and reindexed data and points.

        Raises:
            TypeError: If the filter_result has the wrong type.

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
                    "Wrong filter_result expecting array with boolean values or"
                    "list of indices"
                )
            )
        return PointCloud(new_data, timestamp=self.timestamp)

    def get_cluster(self, eps: float, min_points: int) -> pandas.DataFrame:
        """Get the clusters based on
        :meth:`open3d:open3d.geometry.PointCloud.cluster_dbscan`.
        Process further with :func:`pointcloudset.pointcloud.PointCloud.take_cluster`.

        Args:
            eps (float): Density parameter that is used to find neighboring points.
            min_points (int): Minimum number of points to form a cluster.

        Returns:
            pandas.DataFrame: Dataframe with list of clusters.
        """
        labels = np.array(
            self.to_instance("open3d").cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )
        return pandas.DataFrame(labels, columns=["cluster"])

    def take_cluster(
        self, cluster_number: int, cluster_labels: pandas.DataFrame
    ) -> PointCloud:
        """Takes only the points belonging to the cluster_number.

        Args:
            cluster_number (int): Cluster ID to keep.
            cluster_labels (pandas.DataFrame): Clusters generated with
                :func:`pointcloudset.pointcloud.PointCloud.get_cluster`.

        Returns:
            PointCloud: PointCloud with selected cluster.
        """
        bool_array = (cluster_labels["cluster"] == cluster_number).values
        return self.apply_filter(bool_array)

    def plane_segmentation(
        self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        return_plane_model: bool = False,
    ) -> Union[PointCloud, dict]:
        """Segments a plane in the point cloud using the RANSAC algorithm.
        Based on :meth:`open3d:open3d.geometry.PointCloud.segment_plane`.

        Args:
            distance_threshold (float): Max distance a point can be from the plane
                model, and still be considered as an inlier.
            ransac_n (int):  Number of initial points to be considered inliers in
                each iteration.
            num_iterations (int): Number of iterations.
            return_plane_model (bool, optional): Return also plane model parameters
                if ``True``. Defaults to ``False``.

        Returns:
            PointCloud or dict: PointCloud with inliers or a dict of PointCloud with inliers and the
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
                """Might not produce reproduceable resuts, if the number of points
                is high. Try to reduce the area of interest before using
                plane_segmentation. Caused by open3D."""
            )
        inlier_pointcloud = self.apply_filter(inliers)
        if return_plane_model:
            return {"PointCloud": inlier_pointcloud, "plane_model": plane_model}
        else:
            return inlier_pointcloud
