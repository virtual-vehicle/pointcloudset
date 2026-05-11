"""
Global definitions for the library.
"""

import operator

OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}

PLOTLYSIZELIMIT = 300000

# Fixed cluster chunk sizes used by PointCloud.get_cluster.
GET_CLUSTER_CORE_QUERY_CHUNK_SIZE = 1024
GET_CLUSTER_BORDER_QUERY_CHUNK_SIZE = 4096
