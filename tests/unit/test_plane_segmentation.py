import pytest_check as check

from pointcloudset import Frame
from pointcloudset.frame_core import FrameCore


def test_plane_segmentation_of_open3d(testframe):
    pcd = testframe.limit("intensity", 500, 510).to_instance("open3d")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=500,
    )
    check.equal(len(inliers), 387)


def test_plane_segmentation_of_open3d_2(testframe):
    pcd = testframe.limit("intensity", 500, 510).to_instance("open3d")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=500,
    )
    check.equal(len(inliers), 387)


def test_plane_segmentation_of_open3d_3(testframe):
    pcd = testframe.limit("intensity", 500, 510).to_instance("open3d")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=500,
    )
    check.equal(len(inliers), 387)


def test_plane_segmentation(testframe):
    plane = testframe.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=False,
    )
    check.equal(type(plane), Frame)
    check.equal(len(plane), 387)


def test_plane_segmentation_2(testframe):
    plane = testframe.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=False,
    )
    check.equal(type(plane), Frame)
    check.equal(len(plane), 387)


def test_plane_segmentation_3(testframe):
    plane = testframe.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=True,
    )
    check.equal(type(plane), dict)
    check.equal(len(plane["Frame"]), 387)
