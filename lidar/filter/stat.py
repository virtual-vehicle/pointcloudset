"""Frame filters based on statisitcics"""

from lidar.config import OPS


def quantile_filter(frame, dim: str, relation: str = ">=", cut_quantile: float = 0.5):
    """Filtering based on quantile values of dimension dim of the data.

    Example:

    testframe.quantile_filter("intensity","==",0.5)

    Args:
        dim (str): column in data, for example "intensity"
        relation (str, optional): Any operator as string. Defaults to ">=".
        cut_quantile (float, optional): Qunatile to compare to. Defaults to 0.5.

    Returns:
        Frame: Frame which fullfils the criteria.
    """
    cut_value = frame.data[dim].quantile(cut_quantile)
    filter_array = OPS[relation](frame.data[dim], cut_value)
    return frame.apply_filter(filter_array.to_numpy())
