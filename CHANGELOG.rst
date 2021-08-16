Changelog
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
-------------

Added
~~~~~~
- limit the number of points which can be plotted (issue#18)
- random_down_sample method for pointclouds.


Changed
~~~~~~

- set conda environment name to "pointcloudset" not "base"


[0.2.3] - 2021-07-12
---------------------

Added
~~~~~~
- empty PointCloud object (issue#6)
- columns option to generate empty PointClouds with a specific schema (issue#6)
- support for reading and writing Datasets with empty frames (issue#6)
- check if all required files are written when saving a dataset