"""
A package to work with automotive lidar data.

# Reading this documentation.
The main features of this package are the "Dataset" class consisting of many
"Frame" objects.


# Development

* use docstrings everywhere. The documentation in "doc" is generated with pdoc.
* Have a look at the Makefile and the available make commands
* use typehints when declaring a function, class or method.
* VS code settings in the dev container take care of linting and code formatting.
* write tests for every method/function wich manipulates data
* for example useage you can have a look at the tests
* every 0.x release needs to have 100% code coverage with tests



# Roadmap
## Roadmap for version 0.1.0
* implement the methods from the pipeline.
* overlay plots similar to the lebrig dataset
* update all docstrings & documentation



## Roadmap of 0.2.0
* apply frame processing to all frames
* save the dataset object
* open the dateset object
* save to rosbag
* better plotting
* extent processing
* test with ROS integration

## general things to add
* plotly plot with tootip of data for each point
* read dgps data and include in frame
* make a separate python package
* include a small test .bagfile

## Speedup potentials for future improvements
* use generators
* avoid converting between pdandas and open3D formats where possible

"""

from .dataset import Dataset
from .frame import Frame
