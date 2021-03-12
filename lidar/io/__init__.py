"""
File input and output routines.
"""

from pyntcloud.io import FROM_FILE as FRAME_FROM_FILE

from .frame.csv import write_csv
from .frame.open3d import from_open3d, to_open3d
from .frame.pandas import from_dataframe, to_dataframe
from .frame.pyntcloud import from_pyntcloud, to_pyntcloud

from .dataset.bag import dataset_from_rosbag

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

DATASET_FROM_FILE = {"BAG": dataset_from_rosbag}