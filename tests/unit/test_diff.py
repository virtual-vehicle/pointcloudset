import numpy as np
import pytest_check as check

from lidar import Frame

# def test_point_difference(testframe: Frame):
#     difference = testframe.calculate_single_point_difference(testframe, 4624)
#     check.equal(len(difference), 1)
#     check.equal(
#         (
#             difference.values
#             == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4624.0]])
#         ).all(),
#         True,
#     )
#     check.equal(
#         (
#             (difference.columns).values
#             == np.array(
#                 [
#                     "x difference",
#                     "y difference",
#                     "z difference",
#                     "intensity difference",
#                     "t difference",
#                     "reflectivity difference",
#                     "ring difference",
#                     "noise difference",
#                     "range difference",
#                     "original_id",
#                 ]
#             )
#         ).all(),
#         True,
#     )


# def test_point_difference2(testset: Dataset):
#     testframe1 = testset[0]
#     testframe2 = testset[1]
#     diff = testframe1.calculate_single_point_difference(testframe2, 4692)
#     check.equal(len(diff), 1)
#     check.equal(diff.original_id.values[0], 4692)
#     check.equal(
#         np.allclose(
#             diff.values,
#             [
#                 -5.73961735e-02,
#                 1.63454115e-02,
#                 -6.20609522e-03,
#                 -3.00000000e00,
#                 3.40000000e02,
#                 6.55350000e04,
#                 0.00000000e00,
#                 1.40000000e01,
#                 4.29496724e09,
#                 4.69200000e03,
#             ],
#         ),
#         True,
#     )
#     types = [str(types) for types in diff.dtypes.values]
#     check.equal(
#         types,
#         [
#             "float32",
#             "float32",
#             "float32",
#             "float32",
#             "uint32",
#             "uint16",
#             "uint8",
#             "uint16",
#             "uint32",
#             "uint32",
#         ],
#     )


# def test_point_all_difference(testframe0: Frame, testframe_mini_real: Frame):
#     res = testframe_mini_real.calculate_all_point_differences(testframe0)
#     check.equal(type(res), Frame)
#     check.equal(len(res), len(testframe_mini_real))
#     manual_diff = (
#         testframe_mini_real.data[testframe_mini_real.data["original_id"] == 9701][
#             "x"
#         ].values
#         - testframe0.data[testframe0.data["original_id"] == 9701]["x"].values
#     )[0]
#     check.equal(
#         res.data[res.data["original_id"] == 9701]["x difference"].values[0], manual_diff
#     )
#     check.equal(
#         len(res.data.columns.values),
#         19,
#     )


def test_distances_to_origin(testframe_mini: Frame):
    newframe = testframe_mini.diff("origin")
    check.equal(type(newframe), Frame)
    check.equal(
        np.allclose(
            testframe_mini.data["distance to point: [0 0 0]"].values,
            np.asarray(
                [
                    0.0,
                    1.73205081,
                    1.73205081,
                    937.15579489,
                    1058.09727232,
                    906.10064672,
                    991.83926827,
                    475.99837556,
                ]
            ),
        ),
        True,
    )


def test_calculate_distance_to_plane1(testframe_mini: Frame):
    newframe = testframe_mini.diff(
        "plane", target=np.array([1, 0, 0, 0]), absolute_values=False
    )
    check.equal(type(newframe), Frame)
    check.equal(
        str(list(testframe_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [1 0 0 0]']",
    )
    check.equal(testframe_mini.data["distance to plane: [1 0 0 0]"][1], 1.0)


def test_calculate_distance_to_plane1_2(testframe_mini: Frame):
    newframe = testframe_mini.diff("plane", np.array([1, 0, 0, 0]))
    check.equal(type(newframe), Frame)
    check.equal(
        str(list(testframe_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [1 0 0 0]']",
    )
    check.equal(testframe_mini.data["distance to plane: [1 0 0 0]"][1], 1.0)


def test_calculate_distance_to_plane2(testframe_mini: Frame):
    testframe_mini.diff("plane", target=np.array([-1, 0, 0, 0]), absolute_values=False)
    check.equal(
        str(list(testframe_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [-1  0  0  0]']",
    )
    check.equal(testframe_mini.data["distance to plane: [-1  0  0  0]"][1], -1.0)


def test_calculate_distance_to_plane3(testframe_mini: Frame):
    testframe_mini.diff("plane", target=np.array([-1, 0, 0, 0]), absolute_values=True)
    check.equal(testframe_mini.data["distance to plane: [-1  0  0  0]"][1], 1.0)


def test_calculate_distance_to_point(testframe_mini: Frame):
    testframe_mini.diff("point", target=np.array([-1, 0, 0]))
    check.equal(testframe_mini.data["distance to point: [-1  0  0]"][0], 1.0)
