import numpy as np
import pytest
import pytest_check as check
from lidar import Dataset, Frame


def test_frame_difference_equal(testframe_mini_real: Frame):
    res = testframe_mini_real.diff("frame", testframe_mini_real)
    x_test = res.data["x difference"].values
    y_test = res.data["x difference"].values
    z_test = res.data["x difference"].values
    check.equal(type(res), Frame)
    check.equal(len(res), len(testframe_mini_real))
    check.equal(
        list(res.data.columns),
        [
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "noise",
            "range",
            "original_id",
            "x difference",
            "y difference",
            "z difference",
            "intensity difference",
            "t difference",
            "reflectivity difference",
            "ring difference",
            "noise difference",
            "range difference",
        ],
    )
    check.equal(np.all(x_test == 0.0), True)
    check.equal(np.all(y_test == 0.0), True)
    check.equal(np.all(z_test == 0.0), True)


def test_frame_difference1_no_orignal_id(
    testframe_mini: Frame, testframe_mini_real: Frame
):
    with pytest.raises(ValueError):
        testframe_mini.diff("frame", testframe_mini_real)


def test_frame_difference1_no_orignal_id2(
    testframe_mini: Frame, testframe_mini_real: Frame
):
    with pytest.raises(ValueError):
        testframe_mini_real.diff("frame", testframe_mini)


def test_frame_difference1(testframe0: Frame, testframe_mini: Frame):
    res = testframe_mini.diff("frame", testframe0)
    check.equal(type(res), Frame)
    check.equal(len(res), len(testframe_mini))
    x_test = res.data["x difference"].values
    y_test = res.data["x difference"].values
    z_test = res.data["x difference"].values
    check.equal(
        list(res.data.columns),
        [
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "noise",
            "range",
            "original_id",
            "x difference",
            "y difference",
            "z difference",
            "intensity difference",
            "t difference",
            "reflectivity difference",
            "ring difference",
            "noise difference",
            "range difference",
        ],
    )


def test_frame_difference(testframe0: Frame, testframe_mini_real: Frame):
    res = testframe_mini_real.diff("frame", testframe0)
    # check.equal(len(res), len(testframe_mini_real))
    # manual_diff = (
    #     testframe_mini_real.data[testframe_mini_real.data["original_id"] == 9701][
    #         "x"
    #     ].values
    #     - testframe0.data[testframe0.data["original_id"] == 9701]["x"].values
    # )[0]
    # check.equal(
    #     res.data[res.data["original_id"] == 9701]["x difference"].values[0], manual_diff
    # )
    # check.equal(
    #     len(res.data.columns.values),
    #     19,
    # )


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
