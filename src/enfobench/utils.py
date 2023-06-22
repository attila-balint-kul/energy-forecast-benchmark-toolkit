import pandas as pd
from pandas import Timedelta


def steps_in_horizon(horizon: Timedelta, freq: str) -> int:
    """Return the number of steps in a given horizon."""
    periods = horizon / pd.Timedelta(freq)
    assert periods.is_integer(), "Horizon is not a multiple of the frequency"
    return int(periods)
