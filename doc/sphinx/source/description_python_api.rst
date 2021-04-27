Description of Python API
========================================

The key components of the Python API are Datasets and PointClouds:
    * Dataset: Datasets are based on the classes :class:`pointcloudset.dataset.Dataset` and :class:`pointcloudset.dataset_core.DatasetCore`
    * PointCloud: PointClouds are based on the classes :class:`pointcloudset.pointcloud.PointCloud` and :class:`pointcloudset.pointcloud_core.PointCloudCore`

A Dataset consists of multiple PointCloud objects. Datasets and PointCloud use the functions of the following subpackages:
    * :mod:`pointcloudset.diff`
    * :mod:`pointcloudset.filter`
    * :mod:`pointcloudset.geometry`
    * :mod:`pointcloudset.io`
    * :mod:`pointcloudset.plot`

For a more detailed documentation see the section below.