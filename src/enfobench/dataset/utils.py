import warnings

import pandas as pd


def create_perfect_forecasts_from_covariates(
    past_covariates: pd.DataFrame,
    *,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    **kwargs,
) -> pd.DataFrame:
    """Create forecasts from covariates.

    Sometimes external forecasts are not available for the entire horizon. This function creates
    external forecast dataframe from external covariates as a perfect forecast.

    Args:
        past_covariates: The external covariates.
        horizon: The forecast horizon.
        step: The step size between forecasts.

    Returns:
        The external forecast dataframe.
    """
    start = kwargs.get("start", past_covariates.index[0])
    last_date = past_covariates.index[-1]

    forecasts = []
    while start + horizon <= last_date:
        forecast = past_covariates.loc[(past_covariates.index > start) & (past_covariates.index <= start + horizon)]
        forecast.insert(0, "cutoff_date", start)
        forecast.rename_axis("timestamp", inplace=True)
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
