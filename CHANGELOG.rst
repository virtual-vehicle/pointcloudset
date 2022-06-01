Changelog
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.


Unreleased
-------------

Added
~~~~~~
- bounding_box property for datasets
- animate for datasets as an experimental feature

Changed
~~~~~~
- time format to include milliseconds


0.5.1 - (2022-05-30)
-------------

Fixed
~~~~~~
- laspy in docker image based. Updated to > 2.00

Added
~~~~~~
- dask distributed library in docker image


0.5.0 - (2022-05-30)
-------------

Added
~~~~~~
- better support for data from terrestrial laser scanners
- has_original_id for datasets. Returns true if all pointclouds have original_id
- PointCloud.from_file now supports timestamp input or "from_file"
- diff with "nearest" to calculate distance to nearest point from another pointcloud

Changed
~~~~~~
- time format to 24h PR #45


Fixed
~~~~~~
- fixed typehints after changed open3D API
- plot overlay larger than length of px.colors.qualitative.Plotly Pr #45

Removed
- tqdm dependency (now covered by rich)


0.4.3 - (2022-05-10)
-------------

Fixed
~~~~~~
- missing packaged in base image

0.4.2 - (2022-05-10)
-------------

Changed
~~~~~~
- better entry point for docker images
- using pintcloudset docker images for github actions testing
- streamlined docker images with new base image

Fixed
~~~~~~
- bug with dask 2022.5.0 where meta.json was also read not just the parquet files

0.4.1 - (2022-02-22)
-------------

Fixed
~~~~~~
- now raw tag for pypi in rst files


0.4.0 - (2022-02-22)
-------------

Added
~~~~~~
- rosbagconvert CLI to export individual frames to pointcloudset dataset or files like
    csv or las.
- rosbagconvert has new options and structure


Changed
~~~~~~
- bag2daset has more functionallity and a new name: rosbagconvert
- using rich instead of tqdm
- using rich as a nice UI for the rosbagconvert



0.3.4 - (2022-02-18)
-------------

Fixed
~~~~~~
- now the docker containers runs also on arm64

Changed
~~~~~~
- used open3d version 0.14 as default, which comes with arm wheels
- use dask version 2022.02 as minimum, as there was a bug with 2021.10 and reading files
- using Python 3.9 as minimum



0.3.3 - (2021-09-27)
-------------

Fixed
~~~~~~
- point_size option had no effect when using overlays
- writing of dataset with an empty point cloud at the start

0.3.2 - (2021-08-18)
-------------

Fixed
~~~~~~
- conda environment name was still "base" now is "pointcloudset"
- automatic start of pointcloudset conda environment now working

Changed
~~~~~~
- use fixed version number of pointcloudset_base image

0.3.1 - (2021-08-17)
-------------

wrong release due to testing of github actions and bump2version


0.3.0 (2021-08-17)
-------------

Added
~~~~~~
- random_down_sample method for pointclouds.


Fixed
~~~~~~
- Better handling of plotting large point clouds: warn when number of points is above 300k (issue#18)


Changed
~~~~~~
- set conda environment name to "pointcloudset" not "base"
- better CD of docker images
- sticking to semantic versioning


0.2.3 (2021-07-12)
---------------------

Added
~~~~~~
- empty PointCloud object (issue#6)
- columns option to generate empty PointClouds with a specific schema (issue#6)
- support for reading and writing Datasets with empty frames (issue#6)
- check if all required files are written when saving a dataset