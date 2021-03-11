"""Frame filters based on statisitcics"""

from lidar.config import OPS


def quantile_filter(frame, dim: str, relation: str = ">=", cut_quantile: float = 0.5):
    """Filtering based on quantile values of dimension dim of the data.

    Example:

    testframe.filter("quantile","intensity","==",0.5)

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


def value_filter(frame, dim: "str", relation: str, value: float):
    """Limit the range of certain values in lidar Frame.

    Example:

    testframe.filter("value", "x", ">", 1.0)

    Args:
        dim (str): dimension to limit, any column in data not just x, y, or z
        relation (str): Any operator as string. Defaults to ">=".
        value (float): value to limit.
    Returns:
        Frame: filtered frame, were columns which did not match the criteria were
        dropped.
    """

    bool_array = (OPS[relation](frame.data[dim], value)).to_numpy()
    return frame.apply_filter(bool_array)
