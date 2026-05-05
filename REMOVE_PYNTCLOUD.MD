# GOAL

* no mor pyntcloud depencency because it has not been updated in 4 years and blocks usage of pandas3
* it was mostly used for file io which can be done with pandas.
* must be supported: las, csv, xyz (which is kind of csv), and pcd
* I added pypcd4 depencency

## File import export stategy

* las => laspy for in and out
* laz => laspy for in an out
* xyz => pandas for in and out
* csv => pandas for in and out
* pcd => via pypcd4

## Implementation strategy
* leave pyntclodu there and write tests for file io