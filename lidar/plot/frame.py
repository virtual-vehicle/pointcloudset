""" # Frame plotting
Used mainly by Frame.plot_interactive() but could also be used on its own. 
"""
import plotly.express as px

intensity_scale = [[0, "black"], [0.1, "blue"], [0.2, "green"], [1, "red"]]
colorscales = {
    "intensity": {"max": 1000, "min": 0, "scale": intensity_scale},
    "range": {"max": None, "min": 0, "scale": None},
}


def plotly_3d(
    frame, color: str, point_size: float = 2, **kwargs,
):
    """Plot a Frame as a 3D scatter plot with plotly.

    You can pass arguments to the plotly express function scatter_3D

    Args:
        frame (Frame): the frame to plot
        color (str): Which column to plot. For example "intensity"
        point_size (float, optional): Size of each point. Defaults to 2.

    Raises:
        ValueError: if the color is not in the data

    Returns:
        Plotly plot: The interactive plotly plot, best use inside a jupyter notebook.
    """
    if color not in frame.data.columns:
        raise ValueError(f"choose any of {list(frame.data.columns)}")

    ids = ["id=" + str(i) for i in range(0, frame.data.shape[0])]

    fig = px.scatter_3d(
        frame.data,
        x="x",
        y="y",
        z="z",
        color=color,
        hover_name=ids,
        hover_data=frame.measurments.columns,
        title=frame.convert_timestamp(),
        **kwargs,
    )
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=0)), selector=dict(mode="markers")
    )
    fig.update_layout(scene_aspectmode="data",)
    return fig


def pyntcloud_3d(frame, **kwargs):
    """Plot a Frame with the build in function of pyntcloud. Is faster than plotly.

    Args:
        frame (Frame): the frame to plot

    Returns:
        pyntcloud plot: The interactive pyntcloud plot, best use inside a jupyter notebook.
    """
    return frame.points.plot(mesh=True, backend="threejs", **kwargs)
