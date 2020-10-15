import IPython

import plotly
import pytest
import pytest_check as check


from lidar import Frame


def test_plot1(testframe_mini: Frame):
    check.equal(
        type(testframe_mini.plot_interactive()), plotly.graph_objs._figure.Figure
    )


def test_plot_plotly(testframe_mini: Frame):
    plot = testframe_mini.plot_interactive(backend="plotly")
    check.equal(
        type(plot), plotly.graph_objs._figure.Figure,
    )


def test_plot_pytncloud(testframe_mini: Frame):
    check.equal(
        type(testframe_mini.plot_interactive(backend="pyntcloud")),
        IPython.lib.display.IFrame,
    )


def test_plot_error(testframe_mini: Frame):
    with pytest.raises(ValueError):
        testframe_mini.plot_interactive(backend="fake")


def test_plot_overlay(testframe: Frame):
    smaller = testframe.limit("x", -0.5, 0.0)
    smaller2 = testframe.limit("x", -0.1, 0.0)
    check.equal(
        type(smaller.plot_overlay({"Smaller2": smaller2})),
        plotly.graph_objs._figure.Figure,
    )


def test_plot_overlay_plane(testframe: Frame):
    smaller = testframe.limit("x", -0.5, 0.0)
    plane = smaller.plane_segmentation(0.2, 10, 10, return_plane_model=True)
    check.equal(
        type(smaller.plot_overlay_plane({"plane test": plane["plane_model"]})),
        plotly.graph_objs._figure.Figure,
    )
