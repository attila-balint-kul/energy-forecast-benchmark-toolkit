import warnings

import pandas as pd


def create_perfect_forecasts_from_covariates(
    past_covariates: pd.DataFrame,
    *,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create forecasts from covariates.

    Sometimes external forecasts are not available for the entire horizon. This function creates
    external forecast dataframe from external covariates as a perfect forecast.

    Args:
        past_covariates: The external covariates.
        horizon: The forecast horizon.
        step: The step size between forecasts.
        start: The start date of the forecast. If None, the first date of the covariates is used.

    Returns:
        The external forecast dataframe.
    """
    start = start or past_covariates.index[0]

    if start < past_covariates.index[0]:
        msg = f"start={start} must be after the start of the past_covariates={past_covariates.index[0]}"
        raise ValueError(msg)

    if start > past_covariates.index[-1]:
        msg = f"start={start} must be before the end of the past_covariates={past_covariates.index[-1]}"
        raise ValueError(msg)

    last_date = past_covariates.index[-1]

    forecasts = []
    while start + horizon <= last_date:
        forecast = past_covariates.loc[start : start + horizon]
        forecast.rename_axis("timestamp", inplace=True)
        forecast.reset_index(inplace=True)
        forecast.insert(0, "cutoff_date", pd.to_datetime(start, unit="ns"))
        forecast.set_index(["cutoff_date", "timestamp"], inplace=True)

        if len(forecast) == 0:
            warnings.warn(
                f"Covariates not found for {start} - {start + horizon}, cannot make forecast at step {start}",
                UserWarning,
                stacklevel=2,
            )
        else:
            forecasts.append(forecast)
        start += step

    forecast_df = pd.concat(forecasts, ignore_index=False)
    forecast_df.reset_index(inplace=True)
    forecast_df.sort_values(by=["cutoff_date", "timestamp"], inplace=True)
    return forecast_df
