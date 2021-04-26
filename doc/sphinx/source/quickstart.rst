Installation and Quickstart
========================================

Install python package with pip:

.. code-block:: console

   pip install pointcloudset


Open a jupyter notebook and start

.. code-block:: python

   from pointcloudset import Dataset, Frame
   from pathlib import Path

   dataset = Dataset.from_file(Path(rosbag_file.bag),topic="/os1_cloud_node/points",keep_zeros=False)
   frame = Frame.from_file(Path(lasfile.las))

* See the usage.ipynb notebook in the notebook folder for an interactive tutorial.
* For  more usage examples you can have a look at the tests.