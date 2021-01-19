[![coverage report](https://gitlab.v2c2.at/sensor-fdir/lidar/badges/master/coverage.svg)](https://gitlab.v2c2.at/sensor-fdir/lidar/-/commits/master) 
[![pipeline status](https://gitlab.v2c2.at/sensor-fdir/lidar/badges/master/pipeline.svg)](https://gitlab.v2c2.at/sensor-fdir/lidar/-/commits/master)

# Automotive lidar analysis package

Use and read the documentation in doc. (open index.html in browser)

Don't add documentation here! Everything should be in the docstrings in the .py files.

**PLEASE READ the very short license file before using it!**

# Core Concept 
(maybe move to documentation later)
* pointclouds from lidar: Automotive lidar and terrestial lidar
* support for pointclouds over time - grouped in datasets
* apply processing pipeline to each frame in the dataset
* each point has x,y,z and an arbitrary amount of additional scalar variables (like intensity, and so on)
* data analytics - not online processing
* TODO: optional support for georeferenced point. i.e. each point has a geographical coordinate (3D). (Issue 46 & 47). Also support for caves without GPS signal.
* works on pointcloud2 messages from rosbags with any lidar (from ROS1 at the moment). (Currently Ouster OS1)
* possiblity to produce plots for publications
* static Comparison of lidar point clouds with "orginal_id"
* Comparison of pointcloud to "ground truth" of geometric primitives like planes, sphere ( and in future also to a mesh? )
* Support for pointclouds of lidars sensos where the beam is always send in the same direction (like the Ouster OS ). This makes computations easier.
* TODO: DIFFERENCES BETWEEN FrameA from one sensor and FrameB from another?? (Hausdorff distance between them? )

# Core Use
* read lidar data from a rosbags
* plot lidar point clouds from rosbags



# Development Core Rules
* high level interfaces
* use specialiced packages. Complex dependecies are ok since this package is for data analyitcs only and we privde a docker image.
* write unit tests for each method and function. 100% code coverage for all major releases
* use typehints
* docstrings in Google format
