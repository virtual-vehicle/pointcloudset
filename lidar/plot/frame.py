import plotly.express as px

intensity_scale = [[0, "black"], [0.1, "blue"], [0.2, "green"], [1, "red"]]
colorscales = {
    "intensity": {"max": 1000, "min": 0, "scale": intensity_scale},
    "range": {"max": None, "min": 0, "scale": None},
}


def plotly_3d(
    frame, color: str, point_size: float = 2, **kwargs,
):
    if color not in frame.data.columns:
        raise ValueError(f"choose any of {list(frame.data.columns)}")
    fig = px.scatter_3d(
        frame.data,
        x="x",
        y="y",
        z="z",
        color=color,
        hover_data=frame.measurments.columns,
        title=frame.convert_timestamp(),
        **kwargs,
    )
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=0)), selector=dict(mode="markers")
    )
    return fig


def pyntcloud_3d(frame, **kwargs):
    return frame.points.plot(mesh=True, backend="threejs", **kwargs)
