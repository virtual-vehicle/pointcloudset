rosbagconvert
======================

The best way to work with large ROS bagfiles is to convert the rosbag beforehand with
the provided command line tool.

It is also possible to convert whole or parts of rosbags to many poplar formats which
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

   rosbagconvert test.bag -t /os1_cloud_node/points -d converted



.. click:: pointcloudset.io.dataset.commandline:typer_click_object
   :prog: rosbagconvert
   :nested: full