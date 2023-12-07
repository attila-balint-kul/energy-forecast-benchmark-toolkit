import pandas as pd

from enfobench.evaluation.filters import hampel


def test_hampel():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    expected_s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    expected_mask = pd.Series([False] * 9 + [True] + [False] * 9)

    filtered, mask = hampel(s, window_size=5, n_sigmas=3.0)

    assert (filtered == expected_s).all()
    assert (mask == expected_mask).all()
