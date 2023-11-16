import warnings
from collections.abc import Callable

import pandas as pd
from tqdm import tqdm

from enfobench.dataset import Dataset
from enfobench.evaluation.client import ForecastClient
from enfobench.evaluation.model import Model
from enfobench.evaluation.utils import generate_cutoff_dates, steps_in_horizon


def evaluate_metric_on_forecast(forecast: pd.DataFrame, metric: Callable) -> float:
    """Evaluate a single metric on a single forecast.

    Args:
        forecast: Forecast to evaluate.
        metric: Metric to evaluate.

    Returns:
        Metric value.
    """
    _nonempty_df = forecast.dropna(subset=["y"])
    metric_value = metric(_nonempty_df.y, _nonempty_df.yhat)
    return metric_value


def evaluate_metrics_on_forecast(forecast: pd.DataFrame, metrics: dict[str, Callable]) -> dict[str, float]:
    """Evaluate multiple metrics on a single forecast.

    Args:
        forecast: Forecast to evaluate.
        metrics: Metric to evaluate.

    Returns:
        Metric value.
    """
    metric_values = {
        metric_name: evaluate_metric_on_forecast(forecast, metric) for metric_name, metric in metrics.items()
    }
    return metric_values


def evaluate_metric_on_forecasts(forecasts: pd.DataFrame, metric: Callable) -> pd.DataFrame:
    """Evaluate a single metric on a set of forecasts made at different cutoff points.

    Args:
        forecasts: Forecasts to evaluate.
        metric: Metric to evaluate.

    Returns:
        Metric values for each cutoff with their weight.
    """
    metrics = {
        cutoff: evaluate_metric_on_forecast(group_df, metric) for cutoff, group_df in forecasts.groupby("cutoff_date")
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
    return metrics_df


def evaluate_metrics_on_forecasts(forecasts: pd.DataFrame, metrics: dict[str, Callable]) -> pd.DataFrame:
    """Evaluate multiple metrics on a set of forecasts made at different cutoff points.

    Args:
        forecasts: Forecasts to evaluate.
        metrics: Metric to evaluate.

    Returns:
        Metric values for each cutoff with their weight.
    """
    metric_dfs = [
        evaluate_metric_on_forecasts(forecasts, metric_func).rename(columns={"value": metric_name})
        for metric_name, metric_func in metrics.items()
    ]
    metrics_df = pd.concat(metric_dfs, axis=1)
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
        msg = f"Start date must be after the start of the dataset: {start_date} <= {dataset.target_available_since}."
        raise ValueError(msg)

    initial_training_data = start_date - dataset.target_available_since
    if initial_training_data < pd.Timedelta("7 days"):
        warnings.warn("Initial training data is less than 7 days.", stacklevel=2)

    if end_date > dataset.target_available_until:
        msg = f"End date must be before the end of the dataset: {end_date} > {dataset.target_available_until}."
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
            level=level,
        )
        if not isinstance(forecast, pd.DataFrame) or not isinstance(forecast.index, pd.DatetimeIndex):
            msg = (
                f"Forecast must be a DataFrame with a DatetimeIndex, "
                f"got {type(forecast)} with index {type(forecast.index)}."
            )
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
    return crossval_df
