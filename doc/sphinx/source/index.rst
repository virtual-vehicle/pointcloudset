Welcome to PyMultiLidar's documentation!
========================================

A package to work with automotive lidar data stored in ROS bag files.

.. image:: ../../assets/front.jpg
   :height: 200

Main features
========================================

* Direct import from ROS bagfiles
* Support for large files
* Interactive 3D visualisation
* Support for numerical data per point (intensity, range, noise …)
* High level processing based on pandas, open3D and pyntcloud


The main features of this package are the &#34;Dataset&#34; class consisting of many
&#34;Frame&#34; objects.

* See the usage.ipynb notebook in the notebook folder for an interactive tuturial.
* For  more useage examples you can have a look at the tests.

Installation in another project
========================================


.. code-block:: console

   pip install git+https://gitlab.v2c2.at/sensor-fdir/lidar@0.1.0

Development
========================================

How to Contribute
----------------------------------------

* add Issues in gitlab
* generate a branch and implement new features
* write tests for the new features
* make a merge request in gitlab

Guidelines
----------------------------------------

* use docstrings everywhere. The documentation in &#34;doc&#34; is generated with pdoc with $make doc
* Write tests for every method/function which manipulates data.
* Have a look at the Makefile and the available make commands.
* Use typehints when declaring a function, class or method.
* VS code settings in the dev container take care of linting with mypy and flake8 and code formatting with black.
* every 0.x release needs to have 100% code coverage with tests

Getting started
----------------------------------------

* install VS Code
* install the remote development extension in VS code (ms-vscode-remote.vscode-remote-extensionpack)
* install and run docker desktop
* clone the repository (you can use VC code for that as well)
* open the folder in VS code
* VS code asks to open the remote development environment - say yes
* The docker image is download on the first start, so this may take a while

Gitlab access to the docker registry
----------------------------------------

On your very first time accessing our gitlab docker registry you need to do the following steps:

1.) generate an access token:
* in gitlab =&gt; User settings =&gt; Access token
* generate a token with &#34;read_registry&#34; rights
* store the token safely (KeePass)

2.) In the terminal login to our registry:

.. code-block:: bash

   docker login registry-gitlab.v2c2.at -u <vornamenachname> -p <token>


.. toctree::
   :maxdepth: 5
   :caption: Contents:

   lidar


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
