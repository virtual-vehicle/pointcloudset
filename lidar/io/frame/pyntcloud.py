import pyntcloud


def from_pyntcloud(
    pyntcloud_data,
) -> dict:
    if not isinstance(pyntcloud_data, pyntcloud.PyntCloud):
        raise TypeError(
            f"Type {type(pyntcloud_data)} not supported for conversion."
            f"Expected pyntcloud.PyntCloud"
        )
    return {"data": pyntcloud_data.points}
