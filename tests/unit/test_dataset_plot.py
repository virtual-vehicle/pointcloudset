import plotly
import pytest
import pytest_check as check

from pointcloudset import Dataset
from pointcloudset.plot.dataset import _gen_sliders, _gen_title, _limit_to_bounding_box, animate_dataset


def test_animate_dataset_generates_figure_with_frames(testdataset_mini_real: Dataset):
    with pytest.warns(UserWarning, match="Experimental Feature"):
        fig = animate_dataset(testdataset_mini_real)

    check.equal(type(fig), plotly.graph_objs._figure.Figure)
    check.equal(len(fig.data), len(testdataset_mini_real))
    check.is_true(fig.data[0].visible is True)

    sliders = fig.layout.sliders
    check.equal(len(sliders), 1)
    check.equal(len(sliders[0].steps), len(testdataset_mini_real))


def test_dataset_animate_wrapper_returns_plotly_figure(testdataset_mini_real: Dataset):
    with pytest.warns(UserWarning, match="Experimental Feature"):
        fig = testdataset_mini_real.animate()

    check.equal(type(fig), plotly.graph_objs._figure.Figure)
    check.equal(len(fig.data), len(testdataset_mini_real))


def test_gen_title_uses_frame_number_and_timestamp(testdataset_mini_real: Dataset):
    title = _gen_title(testdataset_mini_real, 0)
    check.is_true(title.startswith("Frame: 0 "))
    check.is_true(testdataset_mini_real[0].timestamp_str in title)


def test_gen_sliders_creates_step_visibility_masks(testdataset_mini_real: Dataset):
    with pytest.warns(UserWarning):
        fig = animate_dataset(testdataset_mini_real)

    sliders = _gen_sliders(testdataset_mini_real, fig)
    check.equal(len(sliders), 1)
    steps = sliders[0]["steps"]
    check.equal(len(steps), len(fig.data))

    for i, step in enumerate(steps):
        visible_mask = step["args"][0]["visible"]
        check.equal(sum(visible_mask), 1)
        check.is_true(visible_mask[i])
        check.equal(step["label"], str(i))


def test_limit_to_bounding_box_sets_scene_ranges(testdataset_mini_real: Dataset):
    with pytest.warns(UserWarning):
        fig = animate_dataset(testdataset_mini_real)

    fig = _limit_to_bounding_box(testdataset_mini_real, fig)

    bounding_box = testdataset_mini_real.bounding_box
    check.equal(list(fig.layout.scene.xaxis.range), list(bounding_box["x"].values))
    check.equal(list(fig.layout.scene.yaxis.range), list(bounding_box["y"].values))
    check.equal(list(fig.layout.scene.zaxis.range), list(bounding_box["z"].values))
