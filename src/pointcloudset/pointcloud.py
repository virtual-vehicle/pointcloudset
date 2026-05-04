from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas
import plotly
import plotly.express as px
import pyntcloud
from sklearn.cluster import DBSCAN

from pointcloudset.config import PLOTLYSIZELIMIT
from pointcloudset.diff import ALL_DIFFS
from pointcloudset.filter import ALL_FILTERS
from pointcloudset.io import (
    POINTCLOUD_FROM_FILE,
    POINTCLOUD_FROM_INSTANCE,
    POINTCLOUD_TO_FILE,
    POINTCLOUD_TO_INSTANCE,
)
from pointcloudset.plot.pointcloud import plot_overlay
from pointcloudset.pointcloud_core import PointCloudCore


class PointCloud(PointCloudCore):
    """
    PointCloud Class with one pointcloud of lidar measurements, laser scanning,
    photogrammetry  or simular.

    One PointCloud consists mainly of
    `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_
    pointcloud
    (`PyntCloud.points <https://pyntcloud.readthedocs.io/en/latest/points.html#points>`_)
    and a pandas.DataFrame (.data) with all the associated data.

    Note that the index of the points is not preserved when applying processing. This
    is necessary since `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_
    does not allow to pass the index. Therefore, a new PointCloud object is generated at
    each processing stage.

    Developer notes:
        * All operations have to act on both, pointcloud, data and keep the timestamp.
        * All processing methods need to return another PointCloud.

    Examples:

        .. code-block:: python

            testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
            testset = pointcloudset.Dataset(testbag,topic="/os1_cloud_node/points",
                keep_zeros=False)
            testpointcloud = testset[0]
    """

    @classmethod
    def from_file(
        cls,
        file_path: Path,
        timestamp: str | datetime.datetime = "from_file",
        **kwargs,
    ):
        """Extract data from file and construct a PointCloud with it. Uses
        `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_ as
        backend.

        Args:
            file_path (pathlib.Path): Path of file to read.
            use_file_timestamp (bool): use the file creation date as timestamp.
                Defaults to True.
            timestamp (str | datetime.datetime): timestamp of pointcloud. If "from_file"
                then the timesamp is taken from file creation datetimne.
                (Defaults to "from_file")
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
            raise ValueError("Unsupported file format; supported formats are: {}".format(list(POINTCLOUD_FROM_FILE)))
        file_path_str = file_path.as_posix()
        if timestamp == "from_file":
            timestamp = datetime.datetime.utcfromtimestamp(file_path.stat().st_mtime)
        pyntcloud_in = pyntcloud.PyntCloud.from_file(file_path_str, **kwargs)
        return cls(data=pyntcloud_in.points, orig_file=file_path_str, timestamp=timestamp)

    def to_file(self, file_path: Path = Path(), **kwargs) -> None:
        """Exports the pointcloud as to a file for use with
        `CloudCompare <https://www.danielgm.net/cc/ake>`_ or similar tools.
        Currently not all attributes of a pointcloud are saved so some information
        is lost when using this function.
        Uses `PyntCloud <https://pyntcloud.readthedocs.io/en/latest/>`_ and `laspy <https://laspy.readthedocs.io/en/latest/>`
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
            raise ValueError("Unsupported file format; supported formats are: {}".format(list(POINTCLOUD_TO_FILE)))

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
        library: Literal[
            "PANDAS",
            "PYNTCLOUD",
            "DATAFRAME",
        ] = "PANDAS",
        instance: (pandas.DataFrame | pyntcloud.PyntCloud) = pandas.DataFrame(),
        **kwargs,
    ) -> PointCloud:
        """Converts a library instance to a pointcloudset PointCloud.

        Args:
            library (str): Name of the library.\n
                If PYNTCLOUD: :func:`pointcloudset.io.pointcloud.pyntcloud.from_pyntcloud`\n
                If DATAFRAME: :func:`pointcloudset.io.pointcloud.pandas.from_dataframe`\n
                If PANDAS: :func:`pointcloudset.io.pointcloud.pandas.from_dataframe`
            instance (Union[pandas.DataFrame, pyntcloud.PyntCloud]): Library instance to convert.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            PointCloud: Derived from the instance.

        Raises:
            ValueError: If instance is not supported.

        """
        library = library.upper()
        if library not in POINTCLOUD_FROM_INSTANCE:
            raise ValueError("Unsupported library; supported libraries are: {}".format(list(POINTCLOUD_FROM_INSTANCE)))
        else:
            return cls(**POINTCLOUD_FROM_INSTANCE[library](instance, **kwargs))

    def to_instance(
        self, library: Literal["PYNTCLOUD", "DATAFRAME", "PANDAS"], **kwargs
    ) -> pyntcloud.PyntCloud | pandas.DataFrame:
        """Convert PointCloud to another library instance.

        Args:
            library (str): Name of the library.\n
                If PYNTCLOUD: :func:`pointcloudset.io.pointcloud.pyntcloud.to_pyntcloud`\n
                If DATAFRAME: :func:`pointcloudset.io.pointcloud.pandas.to_dataframe`\n
                If PANDAS: :func:`pointcloudset.io.pointcloud.pandas.to_dataframe`
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Union[pandas.DataFrame, pyntcloud.PyntCloud]: The derived instance.

        Raises:
            ValueError: If library is not suppored.

        """
        library = library.upper()
        if library not in POINTCLOUD_TO_INSTANCE:
            raise ValueError("Unsupported library; supported libraries are: {}".format(list(POINTCLOUD_TO_INSTANCE)))

        return POINTCLOUD_TO_INSTANCE[library](self, **kwargs)

    def plot(
        self,
        color: None | str = None,
        overlay: dict | None = None,
        point_size: float = 2,
        prepend_id: str = "",
        hover_data: Union(list[str], bool) = None,
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
            hover_data (list(str) or True, optional): Data columns to display in hover. If True
                then all the columns are are show in the hover.
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

        if len(self) > PLOTLYSIZELIMIT:
            warnings.warn(
                f"""Pointcloud above limit of {PLOTLYSIZELIMIT}.
                Plotting might fail or take a long time.
                Consider donwsampling before plotting.
                for example: pointcloud.random_down_sample(10000).plot()"""
            )

        ids = [prepend_id + "id=" + str(i) for i in range(self.data.shape[0])]

        show_hover = True
        if hover_data is None:
            show_hover = False
        elif hover_data:
            hover_data = list(self.data.columns)
        elif isinstance(hover_data, list) & len(hover_data) > 0:
            if self.has_original_id:
                default = ["original_id"]
                hover_data = hover_data + default
            hover_data = list(set(hover_data))
            if any(x not in self.data.columns for x in hover_data):
                raise ValueError(f"choose a list of {list(self.data.columns)} or []")

        scatter_kwargs = {
            "x": "x",
            "y": "y",
            "z": "z",
            "hover_name": ids,
            "hover_data": hover_data,
            "title": self.timestamp_str,
            **kwargs,
        }
        if color is not None:
            scatter_kwargs["color"] = color

        fig = px.scatter_3d(self.data, **scatter_kwargs)

        # Explicitly set a visible fallback color when no color column is used.
        # Newer Plotly versions can leave marker styling undefined in this case,
        # which makes points invisible while hover still works.
        if color is None:
            fig.update_traces(marker_color=px.colors.qualitative.Plotly[0], selector=dict(type="scatter3d"))

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

        fig.update_traces(
            marker=dict(
                size=point_size,
                symbol="circle",
                opacity=1.0,
            ),
            selector=dict(type="scatter3d"),
        )

        return fig

    def diff(
        self,
        name: Literal["origin", "plane", "pointcloud", "point", "nearest"],
        target: None | PointCloud | np.ndarray = None,
        **kwargs,
    ) -> PointCloud:
        """Calculate differences and distances to the origin, plane, point and pointcloud.

        Args:
            name (str):
                "origin": :func:`pointcloudset.diff.origin.calculate_distance_to_origin` \n
                "plane": :func:`pointcloudset.diff.plane.calculate_distance_to_plane` \n
                "pointcloud": :func:`pointcloudset.diff.pointcloud.calculate_distance_to_pointcloud` \n
                "point": :func:`pointcloudset.diff.point.calculate_distance_to_point` \n
                "nearest": :func:`pointcloudset.diff.point.calculate_distance_to_nearest` \n
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

    def filter(self, name: Literal["quantile", "value", "radiusoutlier"], *args, **kwargs) -> PointCloud:
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

    def limit(self, dim: str, minvalue: float, maxvalue: float) -> PointCloud:
        """Limit the range of certain values in pointcloudset PointCloud. Can be chained together.

        Args:
            dim (str): Dimension to limit, any column in data not just x, y, or z.
            minvalue (float): Min value to limit. (greater equal)
            maxvalue (float): Max value to limit. (smaller equal)

        Returns:
            PointCloud: Limited pointcloud, where rows which did not match the criteria were
            dropped.

        Examples:

            .. code-block:: python

                limitedpointcloud = testpointcloud.limit("x", -1.0, 1.0).limit("intensity", 0.0, 50.0)
        """
        if maxvalue < minvalue:
            raise ValueError("maxvalue must be greater than minvalue")
        return self.filter("value", dim, ">=", minvalue).filter("value", dim, "<=", maxvalue)

    def limit_less(self, dim: str, value: float) -> PointCloud:
        """Limit the range if a diminsion to a value.
            Same as filter("value", dim, "<", value)

        Args:
            dim (str): Dimension to limit, any column in data not just x, y, or z.
            value (float): Min value to limit. (less)

        Returns:
            PointCloud: Limited pointcloud, where rows which did not match the criteria
            were dropped.

        Examples:

            .. code-block:: python

                limitedpointcloud = testpointcloud.limit_less("x",1.0)
        """
        return self.filter("value", dim, "<", value)

    def limit_greater(self, dim: str, value: float) -> PointCloud:
        """Limit the range if a diminsion to a value.
            Same as filter("value", dim, ">", value)

        Args:
            dim (str): Dimension to limit, any column in data not just x, y, or z.
            value (float): Value to limit. (greater)

        Returns:
            PointCloud: Limited pointcloud, where rows which did not match the criteria
            were dropped.

        Examples:

            .. code-block:: python

                limitedpointcloud = testpointcloud.limit_greater("x",10.0)
        """
        return self.filter("value", dim, ">", value)

    def apply_filter(self, filter_result: np.ndarray | list[int]) -> PointCloud:
        """Generating a new PointCloud by removing points according to a call of the
        filter method.

        Args:
            filter_result (Union[numpy.ndarray, list[int]]): Filter result.

        Returns:
            PointCloud: PointCloud with filtered rows and reindexed data and points.

        Raises:
            TypeError: If the filter_result has the wrong type.

        """
        if isinstance(filter_result, np.ndarray):
            # dataframe and pyntcloud based filters
            new_data = self.data.loc[filter_result].reset_index(drop=True)
        elif isinstance(filter_result, list):
            # list of integer indices
            new_data = self.data.iloc[filter_result].reset_index(drop=True)
        else:
            raise TypeError("Wrong filter_result expecting array with boolean values orlist of indices")
        return PointCloud(new_data, timestamp=self.timestamp)

    def get_cluster(self, eps: float, min_points: int) -> pandas.DataFrame:
        """Get the clusters using :class:`sklearn.cluster.DBSCAN`.
        Process further with :func:`pointcloudset.pointcloud.PointCloud.take_cluster`.

        Args:
            eps (float): Density parameter that is used to find neighboring points.
            min_points (int): Minimum number of points to form a cluster.

        Returns:
            pandas.DataFrame: Dataframe with list of clusters. Noise points receive
            label ``-1`` and can be retrieved with ``take_cluster(-1, labels)``.

        Raises:
            ValueError: If ``eps`` is not positive, ``min_points`` is less than 1,
                or the point cloud is empty.
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if min_points < 1:
            raise ValueError(f"min_points must be >= 1, got {min_points}")
        if len(self) == 0:
            raise ValueError("Cannot cluster an empty PointCloud")
        labels = DBSCAN(eps=eps, min_samples=min_points).fit(self.points.xyz).labels_
        return pandas.DataFrame(labels, columns=["cluster"])

    def take_cluster(self, cluster_number: int, cluster_labels: pandas.DataFrame) -> PointCloud:
        """Takes only the points belonging to the cluster_number.

        Use ``cluster_number=-1`` to retrieve noise points.

        Args:
            cluster_number (int): Cluster ID to keep. Use ``-1`` for noise points.
            cluster_labels (pandas.DataFrame): Clusters generated with
                :func:`pointcloudset.pointcloud.PointCloud.get_cluster`.

        Returns:
            PointCloud: PointCloud with selected cluster.

        Raises:
            ValueError: If ``cluster_labels`` length does not match this PointCloud.
        """
        if len(cluster_labels) != len(self):
            raise ValueError(
                f"cluster_labels has {len(cluster_labels)} rows but PointCloud has {len(self)} points"
            )
        bool_array = (cluster_labels["cluster"] == cluster_number).values
        return self.apply_filter(bool_array)

    def plane_segmentation(
        self,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        return_plane_model: bool = False,
        seed: int = 42,
    ) -> PointCloud | dict:
        """Segments a plane in the point cloud using the RANSAC algorithm.

        After finding the best consensus set the plane is refit via SVD on all
        inliers, so the returned model is more accurate than the initial sample.

        Args:
            distance_threshold (float): Max distance a point can be from the plane
                model, and still be considered as an inlier.
            ransac_n (int): Number of points sampled per iteration to fit a candidate
                plane. Must be >= 3.
            num_iterations (int): Number of RANSAC iterations. Must be >= 1.
            return_plane_model (bool, optional): Return also plane model parameters
                if ``True``. Defaults to ``False``.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            PointCloud or dict: PointCloud with inliers or a dict of PointCloud with inliers and the
            plane parameters. The plane model is [a, b, c, d] for ax+by+cz+d=0 (normalised).

        Raises:
            ValueError: If the point cloud is empty, ``distance_threshold`` is not
                positive, ``ransac_n`` is less than 3 or exceeds the number of points,
                or ``num_iterations`` is less than 1.
        """
        if len(self) == 0:
            raise ValueError("Cannot segment a plane in an empty PointCloud")
        if distance_threshold <= 0:
            raise ValueError(f"distance_threshold must be positive, got {distance_threshold}")
        if ransac_n < 3:
            raise ValueError(f"ransac_n must be >= 3 to define a plane, got {ransac_n}")
        if ransac_n > len(self):
            raise ValueError(
                f"ransac_n ({ransac_n}) exceeds number of points ({len(self)})"
            )
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

        xyz = self.points.xyz
        n = len(xyz)
        rng = np.random.default_rng(seed)
        best_inliers: list[int] = []
        best_model = np.zeros(4)

        for _ in range(num_iterations):
            sample = xyz[rng.choice(n, ransac_n, replace=False)]
            centroid = sample.mean(axis=0)
            _, _, Vt = np.linalg.svd(sample - centroid)
            normal = Vt[-1]
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal = normal / norm_len
            d = -np.dot(normal, centroid)
            dists = np.abs(xyz @ normal + d)
            inliers = np.where(dists <= distance_threshold)[0].tolist()
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        # Refit plane on all inliers for a more accurate final model.
        if best_inliers:
            inlier_xyz = xyz[best_inliers]
            centroid = inlier_xyz.mean(axis=0)
            _, _, Vt = np.linalg.svd(inlier_xyz - centroid)
            normal = Vt[-1]
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, centroid)
            best_model = np.array([*normal, d])

        inlier_pointcloud = self.apply_filter(best_inliers)
        if return_plane_model:
            return {"PointCloud": inlier_pointcloud, "plane_model": best_model}
        else:
            return inlier_pointcloud

    def random_down_sample(self, number_of_points: int) -> PointCloud:
        """Function to downsample input pointcloud into output pointcloud randomly.
        Made

        Args:
            number_of_points ([int]): number_of_points

        Returns:
            PointCloud: subsampled PointCloud
        """
        new_data = self.data.sample(number_of_points).reset_index()
        return PointCloud(new_data, timestamp=self.timestamp)

    def _add_original_id_from_index(self) -> PointCloud:
        """Add orginal ID column from index."""
        return self._add_column("original_id", self.data.index)
