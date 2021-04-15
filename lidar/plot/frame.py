""" # Frame plotting
Used mainly by Frame.plot() but could also be used on its own.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


def plot_overlay(fig, frame, overlay: dict, **kwargs):
    """Meta function to overlay the plot with other plots. Used in the main plot function
    Not for standalone use.

    Args:
        fig (plotly fig): The orignal plot
        frame (Frame): The orginal frame
        overlay (dict): A dict with objects to overaly.
            For example overlay={"Cluster 1": cluster1, "plane1": plane_model}
            The name of the entry is used in tooltips and the value can either
            be a Frame or a model of a plan as numpy array (a,b,c,d) of a plan.

    Raises:
        ValueError: If the overlay value is wrong

    Returns:
        plotly.graph_objs._figure.Figure: A new plot with all overlays.
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
            fig = plot_overlay_plane(
                fig,
                plane_model=from_dict,
                name=name,
                orig_frame=frame,
                colors=colors,
                i=i,
            )
        elif from_dict._has_data():
            fig = plot_overlay_frame(
                fig, frame=from_dict, name=name, marker_color=marker_color, **kwargs
            )
        else:
            raise ValueError(
                f"{from_dict} is not supported, use either a frame of plan model"
            )

        i = i + 1
        if i > len(colors):
            i = 0
    return fig


def plot_overlay_frame(fig, frame, name: str, marker_color: str, **kwargs):
    """Overlay the plot with another frame.

    Args:
        fig (plotly figure): The original plot.
        frame (Frame): The Frame to overlay.
        name (str): Name of the frame to overlay, used for tooltips.
        marker_color (str): Color of the overlay

    Returns:
        plotly.graph_objs._figure.Figure: Plot with overlayed Frame.
    """
    overlay_fig = frame.plot(
        color=None, point_size=2.0, prepend_id=name + " ", opacity=0.7, **kwargs
    )
    overlay_fig.update_traces(marker_color=marker_color)
    overlay_trace = overlay_fig.data[0]
    return fig.add_trace(overlay_trace)


def plot_overlay_plane(
    fig, plane_model: np.array, name: str, orig_frame, colors, i: int
):

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
