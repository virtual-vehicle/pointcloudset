""" # Frame plotting
Used mainly by Frame.plot_interactive() but could also be used on its own.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import List

intensity_scale = [[0, "black"], [0.1, "blue"], [0.2, "green"], [1, "red"]]
colorscales = {
    "intensity": {"max": 1000, "min": 0, "scale": intensity_scale},
    "range": {"max": None, "min": 0, "scale": None},
}


def plotly_3d(
    frame,
    color: str,
    point_size: float = 2,
    prepend_id: str = "",
    hover_data: List[str] = [],
    **kwargs,
):
    """Plot a Frame as a 3D scatter plot with plotly.

    You can pass arguments to the plotly express function scatter_3D

    Args:
        frame (Frame): the frame to plot
        color (str): Which column to plot. For example "intensity"
        point_size (float, optional): Size of each point. Defaults to 2.
        prepend_id (str, optional): string before point id to display in hover
        hover data (list, optional): data columns to display in hover. Default is all of them.

    Raises:
        ValueError: if the color is not in the data

    Returns:
        Plotly plot: The interactive plotly plot, best use inside a jupyter notebook.
    """
    if color is not None and color not in frame.data.columns:
        raise ValueError(f"choose any of {list(frame.data.columns)} or None")

    ids = [prepend_id + "id=" + str(i) for i in range(0, frame.data.shape[0])]

    if hover_data == []:
        hover_data = frame.data.columns

    if not all([x in frame.data.columns for x in hover_data]):
        raise ValueError(f"choose a list of {list(frame.data.columns)} or []")

    fig = px.scatter_3d(
        frame.data,
        x="x",
        y="y",
        z="z",
        color=color,
        hover_name=ids,
        hover_data=hover_data,
        title=frame.timestamp_str,
        **kwargs,
    )
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=0)), selector=dict(mode="markers")
    )
    fig.update_layout(
        scene_aspectmode="data",
    )
    return fig


def plot_overlay(orig_frame, frames_dict: dict):
    p1 = orig_frame.plot_interactive(color=None, point_size=1.0, prepend_id="Orginal ")
    p1.update_traces(marker_color="black", opacity=0.7)
    i = 0
    colors = px.colors.qualitative.Plotly
    for name, frame in frames_dict.items():
        marker_color = colors[i]
        p2 = frame.plot_interactive(color=None, point_size=2.0, prepend_id=name + " ")
        p2.update_traces(marker_color=marker_color)
        trace2 = p2.data[0]
        p1.add_trace(trace2)
        i = i + 1
        if i > len(colors):
            i = 0
    return p1


def plot_overlay_plane(orig_frame, plane_dict: dict):
    p1 = orig_frame.plot_interactive(color=None, point_size=1.0, prepend_id="Orginal ")
    p1.update_traces(marker_color="black", opacity=0.7)

    x = np.linspace(min(orig_frame.data.x) * 0.95, max(orig_frame.data.x) * 1.05, 100)
    y = np.linspace(min(orig_frame.data.y) * 0.95, max(orig_frame.data.y) * 1.05, 100)

    X, Y = np.meshgrid(x, y)

    surfacecolor = np.ones(shape=X.shape)
    colors = px.colors.qualitative.Plotly
    colorscale = [[color[0] / len(colors), color[1]] for color in enumerate(colors)]

    i = 0
    for name, plane_model in plane_dict.items():
        a, b, c, d = plane_model
        Z = (-d - a * X - b * Y) / c
        p2 = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    name=name,
                    surfacecolor=surfacecolor * i,
                    colorscale=colorscale,
                    showscale=False,
                    cmin=0,
                    cmax=1,
                )
            ]
        )
        trace2 = p2.data[0]
        p1.add_trace(trace2)
        i = i + 1
    p1.update_layout(scene_aspectmode="data")
    return p1


def pyntcloud_3d(frame, **kwargs):
    """Plot a Frame with the build in function of pyntcloud. Is faster than plotly.

    Args:
        frame (Frame): the frame to plot

    Returns:
        pyntcloud plot: The interactive pyntcloud plot, best use inside a jupyter notebook.
    """
    return frame.points.plot(mesh=True, backend="threejs", **kwargs)
