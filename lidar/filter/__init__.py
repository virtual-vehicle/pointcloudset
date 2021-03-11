"""Utiliy functions for filtering of frames."""

from .stat import quantile_filter

ALL_FILTERS = {"quantile": quantile_filter}
