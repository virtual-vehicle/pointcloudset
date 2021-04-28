---
title: 'pointcloudset: A python package data analytics on pointclouds recorded over time'
tags:
  - Python
  - lidar
  - point cloud
  - ROS
authors:
  - name: Thomas Goelles^[corresponding author Thomas.Goelles@v2c2.at]
    orcid: 0000-0002-3925-6260
    affiliation: 1
  - name: Stefan Muckenhuber
    orcid: 0000-0003-1920-8437
    affiliation: "1,2"
  - name: Birgit Schlager
    orcid: 0000-0003-3290-5333
    affiliation: 1
  - name: Sarah Haas
    affiliation: 1
  - name: Tobias Hammer
    affiliation: 1
affiliations:
 - name: Virtual Vehicle Research GmbH, Inffeldgasse 21A, 8010 Graz, Austria
   index: 1
 - name: University of Graz, Heinrichstrasse 36, 8010 Graz, Austria
   index: 2
date: 1 February 2021
bibliography: paper.bib
---

# Summary

Point clouds are a popular way to represent three dimensional data. These point clouds can be acquired by lidar, RGB-D devices or photogrammetry. Some of these devices can record point clouds over time. Especially automotive lidars can record point clouds at 20 or more Hz producing millions of points per second. Analysing these collection of point clouds is a challenge due to the large file sizes and the diversity of sensors. The python package `pointcloudset` provides a way to handle, analyse and visualise huge datasets of point clouds. It features lazy evaluation and parallel processing and is well suited to develop point cloud algorithms and apply them to big datasets.


`pointcloudset` builds on several well established python libraries and packages for data processing and visualization, such as dask [@dask,@matthew_rocklin-proc-scipy-2015], pyntcloud [@pyntcloud], open3D [@Zhou:2018], plotly [@plotly:2015] and pandas [@reback2020pandas:2020,@mckinney-proc-scipy:2010].
# Statement of need
Considering recently emerging, promising lidar technologies, such as micro-electro-mechanical systems, optical phased array, vertical-cavity surface-emitting laser, single photon avalanche diode etc., combined with large efforts invested in particular by the automotive industry [@Warren:2019,@Hecht:2018,@Thakur:2016] to further develop low-cost lidar systems, lidar sensors have the potential to enable a new cost-efficient way to perceive and measure the environment. This will not only have a strong impact on automotive applications but bears also large potential for other research fields and application domains, such as robotics, geophysics, etc. Already today, state-of-the-art lidar sensors designed for automotive applications, such as the Ouster OS-1 [https://ouster.com] or the Velodyne Ultra Puck [https://velodynelidar.com], offer many advantages: they are small in size, light in weight, robust, have a low eye safety class, and support high scanning speed. The expected substantial decrease in costs and increase in performance in the upcoming years will open up many new application areas for lidar systems.

Other python packages like open3D and pyntcloud focus on processing single point clouds. On the other hand ROS (robot operating system) [@ros:2018] provides ways to store and access point cloud data stored in their .bag format. These `rosbags` are meant to be accessed in a serial fashion which is not ideal for post processing and is not suited to extract subsets of the point cloud dataset.


![Dataset object with main properties and ways to read and write data. figure.\label{fig:dataset}](./figures/data_pipeline2.pdf){ width=90% }

![PointCloud set with main properties and ways to read and write data. figure.\label{fig:pointcloud}](./figures/data_pipeline3.pdf){ width=90% }

\autoref{fig:dataset} illustrates the structure of the `Dataset` class including import and export possibilities. A Dataset consist of many PointCloud objects which can be accesses like list elements in Python. Alternativly a `PointCloud` object can also be created directly from files, as illustrated in \autoref{fig:pointcloud}.


# Acknowledgements

The publication was written at Virtual Vehicle Research GmbH in Graz, Austria. The authors would like to acknowledge the financial support within the COMET K2 Competence Centers for Excellent Technologies from the Austrian Federal Ministry for Climate Action (BMK), the Austrian Federal Ministry for Digital and Economic Affairs (BMDW), the Province of Styria (Dept. 12) and the Styrian Business Promotion Agency (SFG). The Austrian Research Promotion Agency (FFG) has been authorised for the programme management.

# References
