from tqdm import tqdm
from typing import List, Callable
from ..frame import Frame


def apply_pipeline(
    frame_list: List[Frame], pipeline: Callable[[Frame], Frame],
) -> List[Frame]:
    """Applies a function to all frames inside the the given list of frames.

    Example:

    test_list = [testset[0], testset[1]]

    def pipeline1(frame: Frame):
        return frame.limit("x", 0, 1)

    testset_result = apply_pipeline(test_list, pipeline=pipeline1)


    Args:
        frame_list (List of Frames): A list containing lidar frames to process.
        pipeline (Callable): A function with a chain of processing on frames. It
        must take a frame as argument and return a frame.

    Returns:
        List[Frame]: A list with frames to which the pipeline function has been applied.
    """
    result_list = []
    for frame in tqdm(frame_list):
        result_list.append(pipeline(frame))
    return result_list
