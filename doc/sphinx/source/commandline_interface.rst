pointcloudset CLI
======================

The best way to work with large ROS1 or ROS2 files is to convert the rosbag beforehand with
the provided command line tool.

It is also possible to convert whole or parts of ROS files to many poplar formats which
are supported by pyntcloud:


.asc / .pts / .txt / .csv / .xyz
.las
.npy / .npz
.obj
.off (with color support)
.pcd
.ply


Example
--------------------------

An example to convert topic /os1_cloud_node/point within the test.bag rosbag to
the converted directory.

.. code-block:: console

   uvx pointcloudset convert -t /os1_cloud_node/points -d converted test.bag

List all PointCloud2 topics in a ROS file.

.. code-block:: console

   pointcloudset topics test.bag


.. click:: pointcloudset.io.dataset.commandline:typer_click_object
   :prog: pointcloudset
   :nested: full

   uvx pointcloudset -d converted -o csv .