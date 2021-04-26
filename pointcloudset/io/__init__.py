"""
Functions for file input and output.
"""

from pyntcloud.io import (
    FROM_FILE as FRAME_FROM_FILE,
)  # needs to be here, ignore warnings

from pointcloudset.io.dataset.bag import dataset_from_rosbag
from pointcloudset.io.dataset.dir import dataset_from_dir, dataset_to_dir
from pointcloudset.io.dataset.frames import dataset_from_frames
from pointcloudset.io.frame.csv import write_csv
from pointcloudset.io.frame.open3d import from_open3d, to_open3d
from pointcloudset.io.frame.pandas import from_dataframe, to_dataframe
from pointcloudset.io.frame.pyntcloud import from_pyntcloud, to_pyntcloud

DATASET_FROM_FILE = {"BAG": dataset_from_rosbag, "DIR": dataset_from_dir}

DATASET_TO_FILE = {"DIR": dataset_to_dir}

DATASET_FROM_INSTANCE = {"FRAMES": dataset_from_frames}

FRAME_TO_FILE = {"CSV": write_csv}

FRAME_FROM_INSTANCE = {
    "PYNTCLOUD": from_pyntcloud,
    "OPEN3D": from_open3d,
    "DATAFRAME": from_dataframe,
    "PANDAS": from_dataframe,
}
FRAME_TO_INSTANCE = {
    "PYNTCLOUD": to_pyntcloud,
    "OPEN3D": to_open3d,
    "DATAFRAME": to_dataframe,
    "PANDAS": to_dataframe,
}
