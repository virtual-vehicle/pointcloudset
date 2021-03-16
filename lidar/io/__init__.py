"""
File input and output routines.
"""

from pyntcloud.io import FROM_FILE as FRAME_FROM_FILE

from .dataset.bag import dataset_from_rosbag
from .dataset.dir import dataset_from_dir, dataset_to_dir
from .frame.csv import write_csv
from .frame.open3d import from_open3d, to_open3d
from .frame.pandas import from_dataframe, to_dataframe
from .frame.pyntcloud import from_pyntcloud, to_pyntcloud

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

DATASET_FROM_FILE = {"BAG": dataset_from_rosbag, "DIR": dataset_from_dir}
DATASET_TO_FILE = {"DIR": dataset_to_dir}