import pytest_check as check

from pointcloudset import PointCloud
from pointcloudset.pointcloud_core import PointCloudCore
import pytest


def test_plane_segmentation_of_open3d(testpointcloud):
    pcd = testpointcloud.limit("intensity", 500, 510).to_instance("open3d")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=500,
    )
    check.equal(len(inliers), 387)


def test_plane_segmentation(testpointcloud):
    plane = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=False,
    )
    check.equal(type(plane), PointCloud)
    check.equal(len(plane), 387)


@pytest.mark.parametrize(
    "return_plane_model, type_out, for_len",
    [(True, dict, "plane"), (False, PointCloud, """plane["PointCloud"]""")],
)
def test_plane_segmentation_plane_model(
    testpointcloud, return_plane_model, type_out, for_len
):
    plane = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=return_plane_model,
    )
    check.equal(type(plane), type_out)
    check.equal(len(eval(for_len)), 387)
