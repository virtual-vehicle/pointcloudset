pointcloudset
=========================================

*Analyze large datasets of point clouds recorded over time in an efficient way.*

.. image:: https://github.com/virtual-vehicle/pointcloudset/actions/workflows/tests_docker.yml/badge.svg
   :target: https://github.com/virtual-vehicle/pointcloudset/actions/workflows/tests_docker.yml
   :alt: test status

.. image:: images/coverage.svg
   :target: https://github.com/virtual-vehicle/pointcloudset/actions/workflows/tests.yml
   :alt: test coverage

.. image:: https://github.com/virtual-vehicle/pointcloudset/actions/workflows/doc.yml/badge.svg
   :target: https://virtual-vehicle.github.io/pointcloudset/
    :alt: Documentation Status

.. image:: https://github.com/virtual-vehicle/pointcloudset/actions/workflows/docker.yml/badge.svg
   :target: https://hub.docker.com/repository/docker/tgoelles/pointcloudset
   :alt: Docker

.. image:: https://badge.fury.io/py/pointcloudset.svg
    :target: https://badge.fury.io/py/pointcloudset
    :alt: PyPi badge

.. image:: https://pepy.tech/badge/pointcloudset/month
    :target: https://pepy.tech/project/pointcloudset
    :alt: PyPi badge

.. image:: https://joss.theoj.org/papers/10.21105/joss.03471/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.03471#
   :alt: JOSS badge

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff
   :alt: code style ruff


.. inclusion-marker-do-not-remove

`Code`_ | `Documentation`_

.. _Code: https://github.com/virtual-vehicle/pointcloudset
.. _Documentation: https://virtual-vehicle.github.io/pointcloudset/




Features
################################################
* Handles point clouds over time
* Directly read ROS files and many pointcloud file formats.
* Generate a dataset from multiple pointclouds. For example from thousands of .las files.
* Building complex pipelines with a clean and maintainable code

.. code-block:: python

   newpointcloud = pointcloud.limit("x",-5,5).filter("quantile","reflectivity", ">",0.5)

* Apply arbitrary functions to datasets of point clouds

.. code-block:: python

   def isolate_target(frame: PointCloud) -> PointCloud:
      return frame.limit("x",0,1).limit("y",0,1)

   def diff_to_pointcloud(pointcloud: PointCloud, to_compare: PointCloud) -> PointCloud:
      return pointcloud.diff("pointcloud", to_compare)

   result = dataset.apply(isolate_target).apply(diff_to_pointcloud, to_compare=dataset[0])

* Includes powerful aggregation method *agg* similar to pandas

.. code-block:: python

  dataset.agg(["min","max","mean","std"])

* Support for large files with lazy evaluation and parallel processing

.. image:: https://raw.githubusercontent.com/virtual-vehicle/pointcloudset/master/images/dask.gif
   :width: 600

* Support for numerical data per point (intensity, range, noise …)
* Interactive 3D visualisation

.. image:: https://raw.githubusercontent.com/virtual-vehicle/pointcloudset/master/images/tree.gif
   :width: 600

* High level processing based on dask, pandas, open3D and pyntcloud
* Docker image is available
* Optimised - but not limited to - automotive lidar
* A command line tool to convert ROS 1 & 2 files


Use case examples
################################################

- Post processing and analytics of a lidar dataset recorded by ROS
- A collection of multiple lidar scans from a terrestrial laser scanner
- Comparison of multiple point clouds to a ground truth
- Analytics of point clouds over time
- Developing algorithms on a single frame and then applying them to huge datasets


Installation with pip
################################################

Install python package with pip:

.. code-block:: console

   pip install pointcloudset

Installation with Docker
################################################

The easiest way to get started is to use the pre-build docker `tgoelles/pointcloudset`_.

.. _tgoelles/pointcloudset: https://hub.docker.com/repository/docker/tgoelles/pointcloudset

Quickstart
################################################

Reading ROS1 or ROS2 files:

.. code-block:: python

   import pointcloudset as pcs
   from pathlib import Path
   import urllib.request

   urllib.request.urlretrieve(
      "https://github.com/virtual-vehicle/pointcloudset/raw/master/tests/testdata/test.bag", "test.bag"
   )

   dataset = pcs.Dataset.from_file(Path("test.bag"), topic="/os1_cloud_node/points", keep_zeros=False)
   pointcloud = dataset[1]
   pointcloud.plot("x", hover_data=True)

You can also generate a dataset from multiple pointclouds form a large variety or formats like las, pcd, csv and more.

.. code-block:: python

   import pointcloudset as pcs
   from pathlib import Path
   import urllib.request

   urllib.request.urlretrieve(
      "https://github.com/virtual-vehicle/pointcloudset/raw/master/tests/testdata/las_files/test_tree.las",
      "test_tree.las",
   )
   urllib.request.urlretrieve(
      "https://github.com/virtual-vehicle/pointcloudset/raw/master/tests/testdata/las_files/test_tree.pcd",
      "test_tree.pcd",
   )

   las_pc = pcs.PointCloud.from_file(Path("test_tree.las"))
   pcd_pc = pcs.PointCloud.from_file(Path("test_tree.pcd"))
   dataset = pcs.Dataset.from_instance("pointclouds", [las_pc, pcd_pc])
   pointcloud = dataset[1]

   pointcloud.plot("z", hover_data=True)

* Read the `html documentation`_.
* Have a look at the `tutorial notebooks`_ in the documentation folder
* For even more usage examples you can have a look at the tests

.. _html documentation: https://virtual-vehicle.github.io/pointcloudset/
.. _tutorial notebooks: https://github.com/virtual-vehicle/pointcloudset/tree/master/doc/sphinx/source/tutorial_notebooks




CLI to convert ROS1 and ROS2 files: pointcloudset convert
##########################################################

The package includes a powerful CLI to convert pointclouds in ROS1 & 2 files into formats like pointcloudset and a folder with csv or las.
It is capable of handling both mcap and db3 ROS2 files.

.. code-block:: console

   pointcloudset convert test.bag --output-format las --output-dir converted_las

.. image:: https://raw.githubusercontent.com/virtual-vehicle/pointcloudset/master/images/cli_demo.gif
   :width: 600

You can view PointCloud2 messages with

.. code-block:: console

   pointcloudset topics test.bag

Tipp: If you have uv installed you can simply run:

.. code-block:: console

   uvx pointcloudset --help

Comparison to related packages
################################################

#. `ROS <http://wiki.ros.org/rosbag/Code%20API>`_ -  bagfiles can contain many point clouds from different sensors.
   The downside of the format is that it is only suitable for serial access and not well suited for data analytics and post processing.
#. `pyntcloud <https://github.com/daavoo/pyntcloud>`_ - Only for single point clouds. This package is used as the basis for the
   PointCloud object.
#. `open3d <https://github.com/intel-isl/Open3D>`_ - Only for single point clouds. Excellent package, which is used for some
   methods on the PointCloud.
#. `pdal <https://github.com/PDAL/PDAL>`_ - Works also with pipelines on point clouds but is mostly focused on single point cloud processing.
   Pointcloudset is purely in python and based on pandas DataFrames. In addition pointcloudset works in parallel to process large datasets.


Citation and contact
################################################

.. |orcid| image:: https://orcid.org/sites/default/files/images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-3925-6260>

|orcid| `Thomas Gölles <https://orcid.org/0000-0002-3925-6260>`_
email: thomas.goelles@v2c2.at

Please cite our `JOSS paper`_ if you use pointcloudset.

.. _JOSS paper: https://joss.theoj.org/papers/10.21105/joss.03471#

.. code-block:: bib

   @article{Goelles2021,
     doi = {10.21105/joss.03471},
     url = {https://doi.org/10.21105/joss.03471},
     year = {2021},
     publisher = {The Open Journal},
     volume = {6},
     number = {65},
     pages = {3471},
     author = {Thomas Goelles and Birgit Schlager and Stefan Muckenhuber and Sarah Haas and Tobias Hammer},
     title = {`pointcloudset`: Efficient Analysis of Large Datasets of Point Clouds Recorded Over Time},
     journal = {Journal of Open Source Software}
   }



