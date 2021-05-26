"""
Functions for plotting frames.
Used mainly by :func:`pointcloudset.pointcloud.PointCloud.plot` but could also be used
on its own.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


def plot_overlay(fig, pointcloud, overlay: dict, **kwargs):
    """Meta function to overlay the plot with other plots. Used in the main plot
    function.
    Not for standalone use.

    Args:
        fig (plotly.graph_objects.Figure): The original plot.
        pointcloud (PointCloud): The original pointcloud.
        overlay (dict): A dict with objects to overlay.
            For example overlay={"Cluster 1": cluster1, "plane1": plane_model}
            The name of the entry is used in tooltips and the value can either
            be a PointCloud or a model of a plane as numpy.ndarray [a,b,c,d] of a plane.
            Uses the plane equation a x + b y + c z + d = 0.

    Raises:
        ValueError: If the overlay value is wrong.

    Returns:
        plotly.graph_objects.Figure: Plot with all overlays.
    """
    fig.update_traces(opacity=0.7)
    fig.update_traces(
        marker=dict(size=1.5, line=dict(width=0)), selector=dict(mode="markers")
    )
    i = 1
    colors = px.colors.qualitative.Plotly
    for name, from_dict in overlay.items():
        marker_color = colors[i]
        if isinstance(from_dict, np.ndarray):
            fig = _plot_overlay_plane(
                fig,
                plane_model=from_dict,
                name=name,
                orig_pointcloud=pointcloud,
                colors=colors,
                i=i,
            )
        elif from_dict._has_data():
            fig = _plot_overlay_pointcloud(
                fig,
                pointcloud=from_dict,
                name=name,
                marker_color=marker_color,
                **kwargs,
            )
        else:
            raise ValueError(
                f"{from_dict} is not supported, use either a PointCloud or plane model"
            )

        i = i + 1
        if i > len(colors):
            i = 0
    return fig


def _plot_overlay_pointcloud(fig, pointcloud, name: str, marker_color: str, **kwargs):
    """Overlay the plot with another pointcloud.

    Args:
        fig (plotly.graph_objects.Figure): The original plot.
        pointcloud (PointCloud): The PointCloud to overlay.
        name (str): Name of the pointcloud to overlay, used for tooltips.
        marker_color (str): Color of the overlay.

    Returns:
        plotly.graph_objects.Figure: Plot with PointCloud overlayed with another PointCloud.
    """
    overlay_fig = pointcloud.plot(
        color=None, point_size=2.0, prepend_id=name + " ", opacity=0.7, **kwargs
    )
    overlay_fig.update_traces(marker_color=marker_color)
    overlay_trace = overlay_fig.data[0]
    return fig.add_trace(overlay_trace)


def _plot_overlay_plane(
    fig, plane_model: np.ndarray, name: str, orig_pointcloud, colors, i: int
):
    """Overlay the plot with plane.

    Args:
        fig (plotly.graph_objects.Figure): The original plot.
        plane_model (numpy.ndarray): [a,b,c,d], for the plane to overlay. Uses the plane
            equation a x + b y + c z + d = 0.
        name (str): Name of the plane to overlay, used for tooltips.
        orig_pointcloud:
        colors: colors from polot_overlay
        i (int): item number from plot_overlay

    Returns:
        plotly.graph_objects.Figure: Plot with PointCloud overlayed with plane.
    """
    bb = orig_pointcloud.bounding_box
    x = np.linspace(bb.x["min"], bb.x["max"], 100)
    y = np.linspace(bb.y["min"], bb.y["max"], 100)
    z = np.linspace(bb.z["min"], bb.z["max"], 100)

    a, b, c, d = plane_model

    eps = 0.000001
    if (abs(c) < eps) & (abs(b) < eps):
        Y, Z = np.meshgrid(y, z)
        X = (-d - b * Y - c * Z) / a
    elif (abs(c) < eps) & (abs(b) > eps):
        X, Z = np.meshgrid(x, z)
        Y = (-d - a * X - c * Z) / b
    else:
        X, Y = np.meshgrid(x, y)
        Z = (-d - a * X - b * Y) / c

    colorscale = [[0, colors[i]], [1, colors[i]]]
    surfacecolor = np.ones(shape=X.shape)

    p2 = go.Surface(
        x=X,
        y=Y,
        z=Z,
        name=name,
        surfacecolor=surfacecolor,
        colorscale=colorscale,
        showscale=False,
        cmin=0,
        cmax=1,
        opacity=0.5,
    )

    fig.add_trace(p2)
    fig.update_layout(scene_aspectmode="data")
    return fig
