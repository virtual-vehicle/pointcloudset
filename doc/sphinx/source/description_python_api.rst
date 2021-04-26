Description of Python API
========================================

The key components of the Python API are Datasets and Frames:
    * Dataset: Datasets are based on the classes :class:`lidar.dataset.Dataset` and :class:`lidar.dataset.DatasetCore`
    * Frame: Frames are based on the classes :class:`lidar.frame.Frame` and :class:`lidar.frame.FrameCore`

A Dataset consists of multiple Frames. Datasets and Frames use the functions of the following subpackages:
    * :mod:`lidar.diff`
    * :mod:`lidar.filter`
    * :mod:`lidar.geometry`
    * :mod:`lidar.io`
    * :mod:`lidar.plot`