"""
Functions for file input and output.
"""

from pyntcloud.io import \
    FROM_FILE as POINTCLOUD_FROM_FILE  # needs to be here, ignore warnings

from pointcloudset.io.dataset.bag import dataset_from_rosbag
from pointcloudset.io.dataset.dir import dataset_from_dir, dataset_to_dir
from pointcloudset.io.dataset.pointcloud import dataset_from_pointclouds
from pointcloudset.io.pointcloud.csv import write_csv
from pointcloudset.io.pointcloud.open3d import from_open3d, to_open3d
from pointcloudset.io.pointcloud.pandas import from_dataframe, to_dataframe
from pointcloudset.io.pointcloud.pyntcloud import from_pyntcloud, to_pyntcloud

DATASET_FROM_FILE = {"BAG": dataset_from_rosbag, "DIR": dataset_from_dir}

DATASET_TO_FILE = {"DIR": dataset_to_dir}

DATASET_FROM_INSTANCE = {"POINTCLOUDS": dataset_from_pointclouds}

POINTCLOUD_TO_FILE = {"CSV": write_csv}

POINTCLOUD_FROM_INSTANCE = {
    "PYNTCLOUD": from_pyntcloud,
    "OPEN3D": from_open3d,
    "DATAFRAME": from_dataframe,
    "PANDAS": from_dataframe,
}
POINTCLOUD_TO_INSTANCE = {
    "PYNTCLOUD": to_pyntcloud,
    "OPEN3D": to_open3d,
    "DATAFRAME": to_dataframe,
    "PANDAS": to_dataframe,
}
