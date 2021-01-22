---
title: 'PyMultiLidar: A python package for lidar datasets with multiple point clouds'
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

A lidar (light detection and ranging) is an active perception sensor operating in the optical or infrared part of the electromagnetic spectrum. Lidar sensors use the time-of-flight principle to measure the distance between sensor and illuminated object, i.e. a light pulse is sent out, reflected at an object, and the travel time between transmission and reception of the light pulse is measured to calculate the distance. Sending several light pulses in different directions and/or applying an array of receivers pointing in different directions, allows to create a 3D (or depth) image of the surrounding environment. Such a 3D image is typically stored as point cloud, where each point has assigned a distance and angle relative to the sensor and additional optional parameters such as intensity or reflectance. Recording multiple point clouds, rather than a single acquisition, from different locations allows to create a more complete 3D representation of the surrounding environment, since objects can then be illuminated from different angles. Recording multiple point clouds from the same locations, allows to investigate dynamic changes of the environment. Importing, processing, analysis, comparison, and visualization of multiple point clouds requires a well-defined data structure, relative positioning, orientation and timing information, and specifically designed, efficient tools and data pipelines. `PyMultiLidar` provides all these functionalities for lidar datasets consisting of multiple point clouds.

# Statement of need

Considering recently emerging, promising lidar technologies, such as micro-electro-mechanical systems, optical phased array, vertical-cavity surface-emitting laser, single photon avalanche diode etc., combined with large efforts invested in particular by the automotive industry [@Warren:2019,@Hecht:2018,@Thakur:2016] to further develop low-cost lidar systems, lidar sensors have the potential to enable a new cost-efficient way to perceive and measure the environment. This will not only have a strong impact on automotive applications but bears also large potential for other research fields and application domains, such as robotics, geophysics, etc. Already today, state-of-the-art lidar sensors designed for automotive applications, such as the Ouster OS-1 [https://ouster.com] or the Velodyne Ultra Puck [https://velodynelidar.com], offer many advantages: they are small in size, light in weight, robust, have a low eye safety class, and support high scanning speed. The expected substantial decrease in costs and increase in performance in the upcoming years will open up many new application areas for lidar systems.

`PyMultiLidar` aims to support testing and development of new application areas for lidar sensors by providing an open-source, user-friendly data processing tool for multiple lidar point clouds. `PyMultiLidar` is a python package for importing, processing, visualizing, and exporting lidar point clouds with focus on applications for multiple point clouds.

![Data pipeline for lidar point clouds. The red box illustrates the functionality of `PyMultiLidar`. figure.\label{fig:data_pipeline}](./figures/data_pipeline.pdf){ width=90% }

\autoref{fig:data_pipeline} illustrates the overall lidar data pipeline of `PyMultiLidar` including import, export possibilities and the two main `PyMultiLidar` classes `Dataset` and `Frame`, which handle multiple and single point clouds respectively. Low-cost lidar systems typically support point cloud recording using ROS (robot operating system) [@ros:2018]. `PyMultiLidar` can directly import multiple point clouds from ROS. Single point clouds can also be imported in other common formats, such as `.las` or `.csv`. `PyMultiLidar` builds on several well established python libraries and packages for data processing and visualization, such as numpy [@harris2020array:2020], pandas [@reback2020pandas:2020,@mckinney-proc-scipy:2010], rospy [@ros:2018], open3D [@Zhou:2018], and plotly [@plotly:2015].

# Acknowledgements

The publication was written at Virtual Vehicle Research GmbH in Graz, Austria. The authors would like to acknowledge the financial support within the COMET K2 Competence Centers for Excellent Technologies from the Austrian Federal Ministry for Climate Action (BMK), the Austrian Federal Ministry for Digital and Economic Affairs (BMDW), the Province of Styria (Dept. 12) and the Styrian Business Promotion Agency (SFG). The Austrian Research Promotion Agency (FFG) has been authorised for the programme management.

# References
