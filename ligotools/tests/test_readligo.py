import numpy as np
from ligotools.readligo import dq_channel_to_seglist, dq2segs

def test_dq_channel_to_seglist_basic():
    # Construct a fake 1 Hz data quality channel:
    # 1 means "good data", 0 means "bad data"
    channel = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    fs = 1  # sampling frequency (1 Hz), so segment indices match array positions

    segments = dq_channel_to_seglist(channel, fs=fs)

    # Expected: (start, stop) index pairs where channel == 1
    # - segment 1: indices 1 to 3
    # - segment 2: indices 4 to 7
    expected = [slice(1, 3), slice(4, 7)]

    assert len(segments) == 2
    assert segments[0].start == expected[0].start
    assert segments[0].stop  == expected[0].stop
    assert segments[1].start == expected[1].start
    assert segments[1].stop  == expected[1].stop


def test_dq2segs_basic():
    # Fake DEFAULT DQ channel: 1s means "valid data"
    channel = np.array([1, 1, 0, 1, 1])
    gps_start = 100  # pretend GPS start time

    seglist = dq2segs(channel, gps_start)

    # Expected segments:
    # channel[0:2] → (100, 102)
    # channel[3:5] → (103, 105)
    expected = [(100, 102), (103, 105)]

    assert seglist.seglist == expected
