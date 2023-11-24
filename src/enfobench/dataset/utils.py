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
    last_date = past_covariates.index[-1]

    forecasts = []
    while start + horizon <= last_date:
        forecast = past_covariates.loc[(past_covariates.index > start) & (past_covariates.index <= start + horizon)]
        forecast.rename_axis("timestamp", inplace=True)
        forecast.reset_index(inplace=True)
        forecast["cutoff_date"] = start.isoformat()  # pd.concat fails if cutoff_date is a Timestamp

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
    forecast_df["cutoff_date"] = pd.to_datetime(forecast_df["cutoff_date"])  # convert back to Timestamp
    return forecast_df
