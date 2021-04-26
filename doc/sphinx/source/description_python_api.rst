Description of Python API
========================================

The key components of the Python API are Datasets and Frames:
    * Dataset: Datasets are based on the classes :class:`pointcloudset.dataset.Dataset` and :class:`pointcloudset.dataset.DatasetCore`
    * Frame: Frames are based on the classes :class:`pointcloudset.frame.Frame` and :class:`pointcloudset.frame.FrameCore`

A Dataset consists of multiple Frames. Datasets and Frames use the functions of the following subpackages:
    * :mod:`pointcloudset.diff`
    * :mod:`pointcloudset.filter`
    * :mod:`pointcloudset.geometry`
    * :mod:`pointcloudset.io`
    * :mod:`pointcloudset.plot`