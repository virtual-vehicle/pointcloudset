import pytest
import pytest_check as check

from lidar import Dataset, Frame


def test_init(testbag1):
    dataset = Dataset(testbag1, "/os1_cloud_node/points")
    check.equal(type(dataset), Dataset)


def test_iter(testset):
    frame_list = [str(frame.timestamp) for frame in testset]
    check.equal(frame_list, ["1592833242657266645", "1592833242755911566"])


def test_iter2(testset):
    frame_list = [frame for frame in testset]
    check.equal(type(frame_list[0]), Frame)


def test_init_error(testbag1):
    with pytest.raises(IOError):
        Dataset(testbag1, "/fake")


def test_testset(testset):
    check.equal(len(testset), 2)
    check.equal(type(testset.orig_file), str)


def test_testset_topic(testset):
    topic = testset.topic
    check.equal(type(topic), str)
    check.equal(topic, "/os1_cloud_node/points")


def test_testset_keep_zeros(testset):
    check.equal(testset.keep_zeros, False)


def test_testset_keep_zeros_true(testset_withzero):
    check.equal(testset_withzero.keep_zeros, True)


def test_getitem(testset: Dataset):
    check.equal(type(testset[0]), Frame)


def test_getitem_late(testset):
    check.equal(type(testset[1]), Frame)


def test_getitem_2times(testset):
    check.equal(type(testset[0]), Frame)
    check.equal(type(testset[1]), Frame)
    check.equal(type(testset[0]), Frame)
    check.equal(type(testset[1]), Frame)


def test_getitem_slice(testset):
    check.equal(type(testset[0:2]), list)
    check.equal(type(testset[0:2][0]), Frame)


def test_getitem_error(testset: Dataset):
    with pytest.raises(TypeError):
        testset["fake"]


def test_getitem_timerange(testset):
    dataset = testset.get_frames_between_timestamps(
        1592833242.6053855, 1592833242.7881582
    )
    check.equal(type(dataset[0:2]), list)
    check.equal(type(dataset[0:2][0]), Frame)


def test_getitem_error3(testset):
    with pytest.raises(ValueError):
        testset[2:0]


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(
        str(testset),
        "Lidar Dataset with 2 frame(s), from file /workspaces/lidar/tests/testdata/test.bag",
    )


def test_repr(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(repr(testset), "Dataset(/workspaces/lidar/tests/testdata/test.bag)")


def test_testset_len_withzero(testset_withzero):
    check.equal(testset_withzero.keep_zeros, True)
    check.equal(len(testset_withzero), 2)


def test_size(testset: Dataset):
    check.equal(testset.size, 16040180)


def test_start_time(testset: Dataset):
    check.equal(testset.start_time, testset.bag.get_start_time())


def test_first_frame_time(testset: Dataset):
    frame0 = testset[0]
    check.equal(testset._first_frame_time, frame0.timestamp.to_sec())


def test_end_time(testset: Dataset):
    check.equal(testset.end_time, 1592833242.7881582)
    check.greater(testset.end_time, testset.start_time)


def test_time(testset: Dataset):
    check.greater(testset.end_time, testset.start_time)


def test_time_step(testset: Dataset):
    check.almost_equal(testset.time_step, 0.1, abs=0.02)


def test_time_of_step0(testset: Dataset):
    frame0 = testset[0]
    time0 = testset.approximate_time_of_frame(0)
    diff = frame0.timestamp.to_sec() - time0
    check.almost_equal(diff, 0.0)


def test_time_of_step1(testset: Dataset):
    frame1 = testset[1]
    time1 = testset.approximate_time_of_frame(1)
    diff = frame1.timestamp.to_sec() - time1
    check.less(diff, 0.5 * testset.time_step)


def test_get_frame_fast0(testset: Dataset):
    frame0 = testset[0]
    frame_fast = testset.get_frame_fast(0)
    check.equal(frame0.timestamp.to_sec(), frame_fast.timestamp.to_sec())


def test_get_frame_fast1(testset: Dataset):
    frame1 = testset[1]
    frame_fast = testset.get_frame_fast(1)
    check.equal(frame1.timestamp.to_sec(), frame_fast.timestamp.to_sec())
