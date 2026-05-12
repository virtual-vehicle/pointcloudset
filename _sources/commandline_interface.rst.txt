pointcloudset CLI
======================

The best way to work with large ROS1 or ROS2 files is to convert the rosbag beforehand with
the provided command line tool.

It is also possible to convert whole or parts of ROS files to the native file formats
supported by pointcloudset:


.csv / .xyz
.las
.pcd

For text formats, the Python API supports both headered and headerless files.
``csv`` writes a header by default and ``xyz`` writes no header by default.
The CLI uses those defaults.


Example
--------------------------

An example to convert topic /os1_cloud_node/point within the test.bag rosbag to
the converted directory.

.. code-block:: console

   pointcloudset convert -t /os1_cloud_node/points -d converted test.bag

List all PointCloud2 topics in a ROS file.

.. code-block:: console

   pointcloudset topics test.bag


.. click:: pointcloudset.io.dataset.commandline:typer_click_object
   :prog: pointcloudset
   :nested: full