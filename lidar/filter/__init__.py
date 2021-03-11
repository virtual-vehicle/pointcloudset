"""Utiliy functions for filtering of frames."""

from .stat import quantile_filter, value_filter

ALL_FILTERS = {"quantile": quantile_filter, "value": value_filter}
