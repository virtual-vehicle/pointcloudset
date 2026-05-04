from importlib.metadata import PackageNotFoundError, version

from .dataset import Dataset
from .pointcloud import PointCloud

try:
    __version__ = version("pointcloudset")
except PackageNotFoundError:
    __version__ = "0.0.0"
