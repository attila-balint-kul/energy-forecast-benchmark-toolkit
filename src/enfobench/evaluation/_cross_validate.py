from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from enfobench.evaluation.client import ForecastClient
from enfobench.evaluation.protocols import Model
from enfobench.utils import steps_in_horizon


def generate_cutoff_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
) -> list[pd.Timestamp]:
    """Generate cutoff dates for cross-validation.

    Parameters
    ----------
    start:
        Start date of the time series.
    end:
        End date of the time series.
    horizon:
        Forecast horizon.
    step:
        Step size between cutoff dates.
    """
    cutoff_dates = []

    cutoff = start
    while cutoff <= end - horizon:
        cutoff_dates.append(cutoff)
        cutoff += step

    if not cutoff_dates:
        raise ValueError("No dates for cross-validation")
    return cutoff_dates


def cross_validate(
    model: Model | ForecastClient,
    start: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    y: pd.Series,
    level: list[int] | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """Cross-validate a model.

    Parameters
    ----------
    model:
        Model to cross-validate.
    start:
        Start date of the time series.
    horizon:
        Forecast horizon.
    step:
        Step size between cutoff dates.
    y:
        Time series target values.
    level:
        Prediction intervals to compute.
        (Optional, if not provided, simple point forecasts will be computed.)
    freq:
        Frequency of the time series.
        (Optional, if not provided, it will be inferred from the time series index.)
    """
    cutoff_dates = generate_cutoff_dates(start, y.index[-1], horizon, step)
    horizon_length = steps_in_horizon(horizon, freq or y.index.inferred_freq)

    # Cross-validation
    forecasts = []
    for cutoff in tqdm(cutoff_dates):
        # make sure that there is no data leakage
        history = y.loc[y.ds <= cutoff, ["ds", "y"]]

        forecast = model.predict(
            horizon_length,
            y=history,
            level=level,
        )
        forecast = forecast.fillna(0)
        forecast["cutoff"] = cutoff
        forecasts.append(forecast)

    crossval_df = pd.concat(forecasts)
    return crossval_df
