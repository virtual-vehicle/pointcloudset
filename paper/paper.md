---
title: 'PyLidarMulti: A python data pipeline for importing, processing, and exporting mutliple lidar point clouds'
tags:
  - Python
  - lidar
  - ROS
  - Ouster OS1
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


...

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


# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
