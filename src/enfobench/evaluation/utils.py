from datetime import timedelta

import pandas as pd


def steps_in_horizon(horizon: pd.Timedelta, freq: str) -> int:
    """Return the number of steps in a given horizon.

    Args:
        horizon: The horizon to be split into steps.
        freq: The frequency of the horizon.

    Returns:
        The number of steps in the horizon.
    """
    freq = "1" + freq if not freq[0].isdigit() else freq
    periods = horizon / pd.Timedelta(freq)
    if not periods.is_integer():
        msg = f"Horizon {horizon} is not a multiple of the frequency {freq}"
        raise ValueError(msg)
    return int(periods)


def periods_in_duration(target: pd.DatetimeIndex, duration: timedelta | pd.Timedelta | str) -> int:
    """Return the number of periods in a given duration.

    Args:
        target: The target variable.
        duration: The duration of the season in a timedelta format.

    Returns:
        The period count.
    """
    if isinstance(duration, timedelta):
        duration = pd.Timedelta(duration)
    elif isinstance(duration, str):
        duration = pd.Timedelta(duration)
    elif isinstance(duration, pd.Timedelta):
        pass
    else:
        msg = f"Duration must be one of [pd.Timedelta, timedelta, str], got {type(duration)}"
        raise ValueError(msg)

    delta_t = target[1] - target[0]
    periods = duration / delta_t
    if not periods.is_integer():
        msg = f"Season length '{duration}' is not a multiple of the frequency '{delta_t}'"
        raise ValueError(msg)
    return int(periods)


def create_forecast_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """Creates a DatetimeIndex for a forecast horizon.

    Args:
        history: The history of the time series.
        horizon: The forecast horizon.

    Returns:
        The time index for the forecast horizon.
    """
    last_date = history.index[-1]
    freq = history.index.inferred_freq
    if freq is None:
        msg = "Cannot create forecast index for a history without frequency"
        raise ValueError(msg)
    freq = "1" + freq if not freq[0].isdigit() else freq

    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(freq),
        periods=horizon,
        freq=freq,
    )
    return forecast_index


def generate_cutoff_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
) -> list[pd.Timestamp]:
    """Generate cutoff dates for cross-validation between two dates.

    The cutoff dates are separated by a fixed step size and the last cutoff date is a horizon away from the end date.

    Args:
        start_date: Start date of the time series.
        end_date: End date of the time series.
        horizon: Forecast horizon.
        step: Step size between cutoff dates.

    Examples
    --------
    >>> generate_cutoff_dates(
    ...     start_date=pd.Timestamp("2020-01-01"),
    ...     end_date=pd.Timestamp("2020-01-05"),
    ...     horizon=pd.Timedelta("2 days"),
    ...     step=pd.Timedelta("1 day"),
    ... )
    [
        Timestamp('2020-01-01 00:00:00'),
        Timestamp('2020-01-02 00:00:00'),
        Timestamp('2020-01-03 00:00:00'),
    ]
    """
    if horizon <= pd.Timedelta(0):
        msg = f"Horizon must be positive, got {horizon}."
        raise ValueError(msg)

    if step <= pd.Timedelta(0):
        msg = f"Step must be positive, got {step}."
        raise ValueError(msg)

    if horizon > end_date - start_date:
        msg = f"Horizon is longer than the evaluation period: {horizon} > {end_date - start_date}."
        raise ValueError(msg)

    if end_date <= start_date:
        msg = f"End date must be after the starting date, got {end_date} <= {start_date}."
        raise ValueError(msg)

    cutoff_dates = []

    cutoff = start_date
    while cutoff <= end_date - horizon:
        cutoff_dates.append(cutoff)
        cutoff += step

    if not cutoff_dates:
        msg = f"No cutoff dates between {start_date} and {end_date} with horizon {horizon} and step {step}."
        raise ValueError(msg)
    return cutoff_dates
