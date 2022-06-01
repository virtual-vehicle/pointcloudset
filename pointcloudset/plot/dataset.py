"""
Functions for plotting datasets.
"""
import plotly.graph_objs as go
import warnings


def animate_dataset(dataset, **kwargs) -> go.Figure:
    """Geneartes animation used in the Dataset class.

    Args:
        dataset (Dataset): dataset to animate

    Returns:
        go.Figure: interactive plotly plot
    """

    warnings.warn("Experimental Feature")

    def plot_frame(pc):
        return pc.plot(**kwargs)

    start_frame = 0
    frames = dataset.apply(plot_frame, warn=False).compute()

    fig = go.Figure()

    for frame in frames:
        fig.add_trace(frame["data"][0])
    fig.data[start_frame].visible = True

    fig.update_layout(sliders=_gen_sliders(dataset, fig))

    return _limit_to_bounding_box(dataset, fig)


def _limit_to_bounding_box(dataset, fig: go.Figure) -> go.Figure:
    """limit the view to the bounding box of the complete dataset."""
    bounding_box = dataset.bounding_box
    fig.update_layout(
        title=_gen_title(dataset, 0),
        scene=dict(
            xaxis=dict(
                range=list(bounding_box["x"].values),
            ),
            yaxis=dict(
                range=list(bounding_box["y"].values),
            ),
            zaxis=dict(
                range=list(bounding_box["z"].values),
            ),
        ),
    )
    return fig


def _gen_sliders(dataset, fig):
    """Generate sliders for plotly."""
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            label=str(i),
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": _gen_title(dataset, i)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frame: "},
            pad={"t": 50},
            steps=steps,
        )
    ]
    return sliders


def _gen_title(dataset, frame_number: int) -> str:
    """Title string of frames."""
    return f"Frame: {frame_number} {dataset[frame_number].timestamp_str}"
