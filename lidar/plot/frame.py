""" # Frame plotting
Used mainly by Frame.plot() but could also be used on its own.
"""
from typing import List, Union

import numpy as np
import plotly.express as px
import plotly.graph_objs as go


intensity_scale = [[0, "black"], [0.1, "blue"], [0.2, "green"], [1, "red"]]
colorscales = {
    "intensity": {"max": 1000, "min": 0, "scale": intensity_scale},
    "range": {"max": None, "min": 0, "scale": None},
}


def plotly_3d(
    frame,
    color: Union[None, str] = None,
    overlay: dict = {},
    point_size: float = 2,
    prepend_id: str = "",
    hover_data: List[str] = [],
    **kwargs,
):
    """Plot a Frame as a 3D scatter plot with plotly.

    You can pass arguments to the plotly express function scatter_3D

    Args:
        frame (Frame): the frame to plot
        color (str or None): Which column to plot. For example "intensity"
        overlay (dict, optional): Dict with of rames to overlay {"Cluster 1": cluster1,
            "Cluster 2": cluster2}
        point_size (float, optional): Size of each point. Defaults to 2.
        prepend_id (str, optional): string before point id to display in hover
        hover data (list, optional): data columns to display in hover. Default is all
            of them.

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

    if len(overlay) > 0:
        fig = plot_overlay(fig, frame, overlay)

    fig.update_layout(
        scene_aspectmode="data",
    )
    return fig


def plot_overlay(fig, frame, overlay: dict):
    fig.update_traces(opacity=0.7)
    i = 0
    colors = px.colors.qualitative.Plotly
    for name, from_dict in overlay.items():
        marker_color = colors[i]
        if isinstance(from_dict, np.ndarray):
            plot_overlay_plane(
                fig,
                plane_model=from_dict,
                name=name,
                orig_frame=frame,
                colors=colors,
                i=i,
            )
        elif from_dict._has_data():
            plot_overlay_frame(
                fig, frame=from_dict, name=name, marker_color=marker_color
            )

        i = i + 1
        if i > len(colors):
            i = 0
    return fig


def plot_overlay_frame(fig, frame, name: str, marker_color: str):
    overlay_fig = frame.plot(color=None, point_size=2.0, prepend_id=name + " ")
    overlay_trace = overlay_fig.data[0]
    fig.add_trace(overlay_trace)
    return fig.update_traces(marker_color=marker_color)


def plot_overlay_plane(
    fig, plane_model: np.array, name: str, orig_frame, colors, i: int
):
    fig.update_traces(opacity=0.7)

    x = np.linspace(min(orig_frame.data.x) * 0.95, max(orig_frame.data.x) * 1.05, 100)
    y = np.linspace(min(orig_frame.data.y) * 0.95, max(orig_frame.data.y) * 1.05, 100)

    X, Y = np.meshgrid(x, y)

    surfacecolor = np.ones(shape=X.shape)
    colorscale = [[color[0] / len(colors), color[1]] for color in enumerate(colors)]

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
    fig.add_trace(trace2)
    i = i + 1
    fig.update_layout(scene_aspectmode="data")
    return fig
