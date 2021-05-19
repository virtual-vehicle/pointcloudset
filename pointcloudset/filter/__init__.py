"""
Utiliy functions for filtering of frames.
"""

from pointcloudset.filter.stat import (quantile_filter, remove_radius_outlier,
                                       value_filter)

ALL_FILTERS = {
    "QUANTILE": quantile_filter,
    "VALUE": value_filter,
    "RADIUSOUTLIER": remove_radius_outlier,
}
