"""Utiliy functions for filtering of frames."""

from .stat import quantile_filter, value_filter, remove_radius_outlier

ALL_FILTERS = {
    "QUANTILE": quantile_filter,
    "VALUE": value_filter,
    "RADIUSOUTLIER": remove_radius_outlier,
}
