import warnings

import pandas as pd


def steps_in_horizon(horizon: pd.Timedelta, freq: str) -> int:
    """Return the number of steps in a given horizon.

    Parameters
    ----------
    horizon:
        The horizon to be split into steps.
    freq:
        The frequency of the horizon.

    Returns
    -------
        The number of steps in the horizon.
    """
    freq = "1" + freq if not freq[0].isdigit() else freq
    periods = horizon / pd.Timedelta(freq)
    if not periods.is_integer():
        raise ValueError("Horizon is not a multiple of the frequency")
    return int(periods)


def create_forecast_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """Create time index for a forecast horizon.

    Parameters
    ----------
    history:
        The history of the time series.
    horizon:
        The forecast horizon.

    Returns
    -------
        The time index for the forecast horizon.
    """
    last_date = history["ds"].iloc[-1]
    inferred_freq = history["ds"].dt.freq
    freq = "1" + inferred_freq if not inferred_freq[0].isdigit() else inferred_freq
    return pd.date_range(
        start=last_date + pd.Timedelta(freq),
        periods=horizon,
        freq=freq,
    )


def create_perfect_forecasts_from_covariates(
    covariates: pd.DataFrame,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    **kwargs,
) -> pd.DataFrame:
    """Create forecasts from covariates.

    Sometimes external forecasts are not available for the entire horizon. This function creates
    external forecast dataframe from external covariates as a perfect forecast.

    Parameters
    ----------
    covariates:
        The external covariates.
    horizon:
        The forecast horizon.
    step:
        The step size between forecasts.

    Returns
    -------
        The external forecast dataframe.
    """
    if kwargs.get("start") is not None:
        start = kwargs.get("start")
    else:
        start = covariates.index[0]

    last_date = covariates.index[-1]

    forecasts = []
    while start + horizon <= last_date:
        forecast = covariates.loc[
            (covariates.index > start) & (covariates.index <= start + horizon)
        ]
        forecast.insert(0, "cutoff_date", start)
        forecast.rename_axis("ds", inplace=True)
        forecast.reset_index(inplace=True)

        if len(forecast) == 0:
            warnings.warn(
                f"Covariates not found for {start} - {start + horizon}, cannot make forecast at step {start}",
                UserWarning,
                stacklevel=2,
            )

        forecasts.append(forecast)
        start += step

    forecast_df = pd.concat(forecasts, ignore_index=True)
    return forecast_df
