import pytest_check as check
import pandas as pd
import numpy as np
import open3d as o3d
from lidar.convert.convert import convert_df2pcd


def test_convert_df2pcd(testframe):
    converted: o3d.open3d_pybind.geometry.PointCloud = convert_df2pcd(testframe.data)
    check.equal(type(converted), o3d.open3d_pybind.geometry.PointCloud)
    check.equal(len(np.asarray(converted.points)), len(testframe))
