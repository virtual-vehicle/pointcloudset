import plotly
import pytest_check as check

from pointcloudset import PointCloud


def test_plot1(testpointcloud_mini: PointCloud):
    check.equal(type(testpointcloud_mini.plot()), plotly.graph_objs._figure.Figure)


def test_plot_plotly(testpointcloud_mini: PointCloud):
    plot = testpointcloud_mini.plot()
    check.equal(
        type(plot),
        plotly.graph_objs._figure.Figure,
    )


def test_plot_overlay(testpointcloud: PointCloud):
    smaller = testpointcloud.limit("x", -0.5, 0.0)
    smaller2 = testpointcloud.limit("x", -0.4, 0.0)
    check.equal(
        type(smaller.plot(overlay={"Smaller2": smaller2})),
        plotly.graph_objs._figure.Figure,
    )


def test_plot_overlay_plane(testpointcloud: PointCloud):
    smaller = testpointcloud.limit("x", -0.5, 0.0)
    plane = smaller.plane_segmentation(0.2, 10, 10, return_plane_model=True)
    check.equal(
        type(smaller.plot(overlay={"plane test": plane["plane_model"]})),
        plotly.graph_objs._figure.Figure,
    )
