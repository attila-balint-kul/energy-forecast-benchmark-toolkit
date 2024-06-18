import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from enfobench.core import Dataset, Model
from enfobench.evaluation.client import ForecastClient
from enfobench.evaluation.utils import create_forecast_index, generate_cutoff_dates, steps_in_horizon


def _compute_metric(forecast: pd.DataFrame, metric: Callable) -> float:
    """Compute a single metric value.

    Args:
        forecast: Forecast to evaluate.
        metric: Metric to evaluate.

    Returns:
        Metric value.
    """
    metric_value = metric(forecast.y, forecast.yhat)
    return metric_value


def _compute_metrics(forecast: pd.DataFrame, metrics: dict[str, Callable]) -> dict[str, float]:
    """Compute multiple metric values.

    Args:
        forecast: Forecast to evaluate.
        metrics: Metric to evaluate.

    Returns:
        Metrics dictionary with metric names as keys and metric values as values.
    """
    metric_values = {metric_name: _compute_metric(forecast, metric) for metric_name, metric in metrics.items()}
    return metric_values


def _evaluate_group(forecasts: pd.DataFrame, metrics: dict[str, Callable], index: Any) -> pd.DataFrame:
    clean_df = forecasts.dropna(subset=["y"])
    if clean_df.empty:
        ratio = 0.0
        metrics = {metric_name: np.nan for metric_name in metrics}
    else:
        ratio = len(clean_df) / len(forecasts)
        metrics = _compute_metrics(clean_df, metrics)
    df = pd.DataFrame({**metrics, "weight": ratio}, index=[index])
    return df


def evaluate_metrics(
    forecasts: pd.DataFrame,
    metrics: dict[str, Callable],
    *,
    groupby: str | None = None,
) -> pd.DataFrame:
    """Evaluate multiple metrics on forecasts.

    Args:
        forecasts: Forecasts to evaluate.
        metrics: Metric to evaluate.
        groupby: Column to group forecasts by. (Optional, if not provided, forecasts will not be grouped.)

    Returns:
        Metric values for each cutoff with their weight.
    """
    if groupby is None:
        return _evaluate_group(forecasts, metrics, 0)

    if groupby not in forecasts.columns:
        msg = f"Groupby column {groupby} not found in forecasts."
        raise ValueError(msg)

    metrics_df = pd.concat(
        [_evaluate_group(group_df, metrics, value) for value, group_df in tqdm(forecasts.groupby(groupby))]
    )

    metrics_df.rename_axis(groupby, inplace=True)
    metrics_df.reset_index(inplace=True)
    return metrics_df


def cross_validate(
    model: Model | ForecastClient,
    dataset: Dataset,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    level: list[int] | None = None,
) -> pd.DataFrame:
    """Cross-validate a model.

    Args:
        model: Model to cross-validate.
        dataset: Dataset to cross-validate on.
        start_date: Start date of the time series.
        end_date: End date of the time series.
        horizon: Forecast horizon.
        step: Step size between cutoff dates.
        level: Prediction intervals to compute. (Optional, if not provided, simple point forecasts will be computed.)
    """
    if start_date <= dataset.target_available_since:
        msg = f"start_date={start_date} must be after the start of the dataset={dataset.target_available_since}"
        raise ValueError(msg)

    if start_date >= dataset.target_available_until:
        msg = f"start_date={start_date} must be before the end of the dataset={dataset.target_available_until}"
        raise ValueError(msg)

    initial_training_data = start_date - dataset.target_available_since
    if initial_training_data < pd.Timedelta("7 days"):
        warnings.warn("Initial training data is less than 7 days.", stacklevel=2)

    if end_date < dataset.target_available_since:
        msg = f"end_date={end_date} must be after the start of the dataset={dataset.target_available_since}"
        raise ValueError(msg)

    if end_date > dataset.target_available_until:
        msg = f"end_date={end_date} must be before the end of the dataset={dataset.target_available_until}"
        raise ValueError(msg)

    cutoff_dates = generate_cutoff_dates(start_date, end_date, horizon, step)
    horizon_length = steps_in_horizon(horizon, dataset.target_freq)

    forecasts = []
    for cutoff_date in tqdm(cutoff_dates):
        history = dataset.get_history(cutoff_date)
        past_covariates = dataset.get_past_covariates(cutoff_date)
        future_covariates = dataset.get_future_covariates(cutoff_date)

        forecast = model.forecast(
            horizon_length,
            history=history,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            metadata=dataset.metadata,
            level=level,
        )
        if not isinstance(forecast, pd.DataFrame) or not isinstance(forecast.index, pd.DatetimeIndex):
            msg = (
                f"Forecast must be a DataFrame with a DatetimeIndex, "
                f"got {type(forecast)} with index {type(forecast.index)}."
            )
            raise ValueError(msg)

        expected_index = create_forecast_index(history, horizon_length)
        if not (expected_index == forecast.index).all():
            msg = "Forecast index does not match the expected index."
            raise ValueError(msg)

        forecast_contains_nans = forecast.isna().any(axis=None)
        if forecast_contains_nans:
            msg = "Forecast contains NaNs, make sure to fill in missing values."
            raise ValueError(msg)

        if len(forecast) != horizon_length:
            msg = f"Forecast does not match the requested horizon length {horizon_length}, got {len(forecast)}."
            raise ValueError(msg)

        forecast.rename_axis("timestamp", inplace=True)
        forecast.reset_index(inplace=True)
        forecast["cutoff_date"] = pd.to_datetime(cutoff_date, unit="ns")
        forecast.set_index(["cutoff_date", "timestamp"], inplace=True)
        forecasts.append(forecast)

    crossval_df = pd.concat(forecasts).reset_index()

    # Merge the forecast with the target
    crossval_df = crossval_df.merge(dataset._target, left_on="timestamp", right_index=True)
    crossval_df.sort_values(by=["cutoff_date", "timestamp"], inplace=True)
    return crossval_df
