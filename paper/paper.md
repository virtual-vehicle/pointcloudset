---
title: '`pointcloudset`: Efficient analysis of large datasets of point clouds recorded over time'
tags:
  - Python
  - lidar
  - point cloud
  - ROS
authors:
  - name: Thomas Goelles
    orcid: 0000-0002-3925-6260
    affiliation: 1
  - name: Birgit Schlager
    orcid: 0000-0003-3290-5333
    affiliation: "1,2"
  - name: Stefan Muckenhuber
    orcid: 0000-0003-1920-8437
    affiliation: "1,3"
  - name: Sarah Haas
    affiliation: 1
  - name: Tobias Hammer
    affiliation: 1
affiliations:
 - name: Virtual Vehicle Research GmbH, Inffeldgasse 21A, 8010 Graz, Austria
   index: 1
 - name: Graz University of Technology, Rechbauerstrasse 12, 8010 Graz, Austria
   index: 2
 - name: University of Graz, Heinrichstrasse 36, 8010 Graz, Austria
   index: 3
date: 1 February 2021
bibliography: paper.bib
---

# Summary

Point clouds are a very common format for representing three dimensional data. Point clouds can be acquired by different sensor types and methods, such as lidar (light detection and ranging), radar (radio detection and ranging), RGB-D (red, green, blue, depth) cameras, photogrammetry, etc. In many cases multiple point clouds are recorded over time. E.g., automotive lidars record point clouds with very high acquisition frequencies (typically around 10-20Hz) resulting in millions of points per second. Analyzing such a large collection of point clouds is a big challenge due to the huge amount of measurement data. The python package `pointcloudset` provides a way to handle, analyse, and visualize large datasets consisting of multiple point clouds recorded over time. `pointcloudset` features lazy evaluation and parallel processing and is designed to enable development of new point cloud algorithms and their application on big datasets.

`pointcloudset` builds on several well established python libraries and packages for data processing and visualization, such as dask [@dask;@matthew_rocklin-proc-scipy-2015], pyntcloud [@pyntcloud], open3D [@Zhou:2018], plotly [@plotly:2015], and pandas [@reback2020pandas:2020;@mckinney-proc-scipy:2010].

# Statement of need
Considering recently emerging, promising lidar technologies, such as micro-electro-mechanical systems, optical phased array, vertical-cavity surface-emitting laser, single-photon avalanche diode etc., combined with large efforts invested in particular by the automotive industry [@Warren:2019;@Hecht:2018;@Thakur:2016] to further develop low-cost lidar systems, lidar sensors have the potential to enable a new cost-efficient way to perceive and measure the environment. This will not only have a strong impact on automotive applications but bears also large potential for other research fields and application domains, such as robotics, geophysics, etc. Already today, state-of-the-art lidar sensors designed for automotive applications, such as the Ouster OS-1 [https://ouster.com] or the Velodyne Ultra Puck [https://velodynelidar.com], offer many advantages compared to previous lidar systems used for e.g. 3D surveying or terrestrial laser scanning: automotive lidar sensors are small in size, light in weight, robust, have a low eye safety class, and support high scanning speed. The expected substantial decrease in costs and increase in performance in the upcoming years will open up many new application areas for lidar systems.

Apart from the progress in the lidar sector, technological improvements, as well as size and cost reduction can also be observed for other 3D sensing technologies, such as radar and RGB-D cameras. This will additionally open up new possibilities and application areas for 3D sensing methods and increases the importance of python packages that are able to process large amounts of point cloud data.

Other python packages for point clouds, such as open3D, pyntcloud, and pcl [@Rusu_ICRA2011_PCL] and its python bindings [@python-pcl], focus on processing single point clouds rather than on processing time series of point clouds. Another library is PDAL [@pdal_contributors_2018_2556738] which also works with pipelines on point clouds. However, it is focused on single point cloud processing as well. ROS (robot operating system) [@ros:2018] provides a way to store, access, and visualize multiple point clouds stored as `rosbags`. However, these `rosbags` are meant to be accessed only in a serial fashion, which is not ideal for post processing and not well suited for extracting subsets of the point cloud dataset.

Compared to mentioned packages, pointcloudset provides efficient analysis of time series of point clouds by parallel processing. E.g. the package is a helpful toolkit for post processing of lidar datasets recorded by ROS or for post processing of multiple lidar scans from terrestrial laser scanners. Algorithms can be developed for a single point cloud and can then be applied to big datasets of point clouds. 


![Dataset object with main properties and ways to read and write data. \label{fig:dataset}](./figures/data_pipeline2.pdf){ width=100% }

![PointCloud set with main properties and ways to read and write data. \label{fig:pointcloud}](./figures/data_pipeline3.pdf){ width=80% }

\autoref{fig:dataset} illustrates the structure of the `Dataset` class including import and export possibilities. A Dataset consists of many PointCloud objects which can be accessed like list elements in Python. Alternatively, a `PointCloud` object can also be created directly from files, as illustrated in \autoref{fig:pointcloud}.

# Contributions

T.G. developed the concept and architecture; T.G., B.S., and S.H. developed the software; T.G. wrote the automatic tests; B.S. and T.G. wrote the software documentation; T.H. and T.G. created Jupyter notebooks for example usage; S.M., T.G., and B.S. wrote the manuscript. All authors contributed to the manuscript and software testing.

# Acknowledgements

The publication was written at Virtual Vehicle Research GmbH in Graz, Austria. The authors would like to acknowledge the financial support within the COMET K2 Competence Centers for Excellent Technologies from the Austrian Federal Ministry for Climate Action (BMK), the Austrian Federal Ministry for Digital and Economic Affairs (BMDW), the Province of Styria (Dept. 12) and the Styrian Business Promotion Agency (SFG). The Austrian Research Promotion Agency (FFG) has been authorised for the programme management.

# References
