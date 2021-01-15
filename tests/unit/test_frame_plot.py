import plotly
import pytest_check as check

from lidar import Frame


def test_plot1(testframe_mini: Frame):
    check.equal(type(testframe_mini.plot()), plotly.graph_objs._figure.Figure)


def test_plot_plotly(testframe_mini: Frame):
    plot = testframe_mini.plot()
    check.equal(
        type(plot),
        plotly.graph_objs._figure.Figure,
    )


def test_plot_overlay(testframe: Frame):
    smaller = testframe.limit("x", -0.5, 0.0)
    smaller2 = testframe.limit("x", -0.1, 0.0)
    check.equal(
        type(smaller.plot(overlay={"Smaller2": smaller2})),
        plotly.graph_objs._figure.Figure,
    )


def test_plot_overlay_plane(testframe: Frame):
    smaller = testframe.limit("x", -0.5, 0.0)
    plane = smaller.plane_segmentation(0.2, 10, 10, return_plane_model=True)
    check.equal(
        type(smaller.plot(overlay={"plane test": plane["plane_model"]})),
        plotly.graph_objs._figure.Figure,
    )
