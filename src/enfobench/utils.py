import pandas as pd
from pandas import Timedelta


def steps_in_horizon(horizon: Timedelta, freq: str) -> int:
    """Return the number of steps in a given horizon."""
    freq = "1" + freq if not freq[0].isdigit() else freq
    periods = horizon / pd.Timedelta(freq)
    assert periods.is_integer(), "Horizon is not a multiple of the frequency"
    return int(periods)


def create_forecast_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    last_date = history["ds"].iloc[-1]
    inferred_freq = history["ds"].dt.freq
    freq = "1" + inferred_freq if not inferred_freq[0].isdigit() else inferred_freq
    return pd.date_range(
        start=last_date + pd.Timedelta(freq),
        periods=horizon,
        freq=freq,
    )
