bag2dataset
======================

For large ROS bagfiles best way is to convert the rosbag beforehand with the provided
command line tool.



Example
--------------------------

An example to convert topic /os1_cloud_node/point within the test.bag rosbag to
the same directory.

.. code-block:: console

   bag2dataset test.bag /os1_cloud_node/points .



.. click:: pointcloudset.io.dataset.commandline:typer_click_object
   :prog: bag2dataset
   :nested: full