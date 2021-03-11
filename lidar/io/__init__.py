"""
File input and output routines.
"""

from pyntcloud.io import FROM_FILE as FRAME_FROM_FILE
from .frame.pyntcloud import from_pyntcloud, to_pyntcloud
from .frame.open3d import from_open3d, to_open3d
from .frame.csv import write_csv


FRAME_TO_FILE = {"CSV": write_csv}
FRAME_FROM_INSTANCE = {"PYNTCLOUD": from_pyntcloud, "OPEN3D": from_open3d}
FRAME_TO_INSTANCE = {"PYNTCLOUD": to_pyntcloud, "OPEN3D": to_open3d}
