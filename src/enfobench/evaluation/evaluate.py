import warnings
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from enfobench.evaluation.client import ForecastClient
from enfobench.evaluation.protocols import Dataset, Model
from enfobench.evaluation.utils import steps_in_horizon


def evaluate_metric_on_forecast(forecast: pd.DataFrame, metric: Callable) -> float:
    """Evaluate a single metric on a single forecast.

    Parameters:
    -----------
    forecast:
        Forecast to evaluate.
    metric:
        Metric to evaluate.

    Returns:
    --------
    metric_value:
        Metric value.
    """
    _nonempty_df = forecast.dropna(subset=["y"])
    metric_value = metric(_nonempty_df.y, _nonempty_df.yhat)
    return metric_value


def evaluate_metrics_on_forecast(
    forecast: pd.DataFrame, metrics: Dict[str, Callable]
) -> Dict[str, float]:
    """Evaluate multiple metrics on a single forecast.

    Parameters:
    -----------
    forecast:
        Forecast to evaluate.
    metrics:
        Metric to evaluate.

    Returns:
    --------
    metric_value:
        Metric value.
    """
    metric_values = {
        metric_name: evaluate_metric_on_forecast(forecast, metric)
        for metric_name, metric in metrics.items()
    }
    return metric_values


def evaluate_metric_on_forecasts(forecasts: pd.DataFrame, metric: Callable) -> pd.DataFrame:
    """Evaluate a single metric on a set of forecasts made at different cutoff points.

    Parameters:
    -----------
    forecasts:
        Forecasts to evaluate.
    metric:
        Metric to evaluate.

    Returns:
    --------
    metrics_df:
        Metric values for each cutoff with their weight.
    """
    metrics = {
        cutoff: evaluate_metric_on_forecast(group_df, metric)
        for cutoff, group_df in forecasts.groupby("cutoff_date")
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
    return metrics_df


def evaluate_metrics_on_forecasts(
    forecasts: pd.DataFrame, metrics: Dict[str, Callable]
) -> pd.DataFrame:
    """Evaluate multiple metrics on a set of forecasts made at different cutoff points.

    Parameters:
    -----------
    forecasts:
        Forecasts to evaluate.
    metrics:
        Metric to evaluate.

    Returns:
    --------
    metrics_df:
        Metric values for each cutoff with their weight.
    """
    metric_dfs = [
        evaluate_metric_on_forecasts(forecasts, metric_func).rename(columns={"value": metric_name})
        for metric_name, metric_func in metrics.items()
    ]
    metrics_df = pd.concat(metric_dfs, axis=1)
    return metrics_df


def generate_cutoff_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
) -> List[pd.Timestamp]:
    """Generate cutoff dates for cross-validation between two dates.

    The cutoff dates are separated by a fixed step size and the last cutoff date is a horizon away from the end date.

    Parameters
    ----------
    start_date:
        Start date of the time series.
    end_date:
        End date of the time series.
    horizon:
        Forecast horizon.
    step:
        Step size between cutoff dates.

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
        raise ValueError("Horizon must be positive.")

    if step <= pd.Timedelta(0):
        raise ValueError("Step must be positive.")

    if horizon > end_date - start_date:
        raise ValueError(
            f"Horizon is longer than the evaluation period: {horizon} > {end_date - start_date}."
        )

    if end_date <= start_date:
        raise ValueError("End date must be after the starting date.")

    cutoff_dates = []

    cutoff = start_date
    while cutoff <= end_date - horizon:
        cutoff_dates.append(cutoff)
        cutoff += step

    if not cutoff_dates:
        raise ValueError("No dates for cross-validation")
    return cutoff_dates


def cross_validate(
    model: Union[Model, ForecastClient],
    dataset: Dataset,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    level: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Cross-validate a model.

    Parameters
    ----------
    model:
        Model to cross-validate.
    dataset:
        Dataset to cross-validate on.
    start_date:
        Start date of the time series.
    end_date:
        End date of the time series.
    horizon:
        Forecast horizon.
    step:
        Step size between cutoff dates.
    level:
        Prediction intervals to compute.
        (Optional, if not provided, simple point forecasts will be computed.)
    """
    if start_date <= dataset.start_date:
        raise ValueError("Start date must be after the start of the target values.")

    initial_training_data = start_date - dataset.start_date
    if initial_training_data < pd.Timedelta("7 days"):
        warnings.warn("Initial training data is less than 7 days.", stacklevel=2)

    if end_date > dataset.end_date:
        raise ValueError("End date is beyond the target values.")

    cutoff_dates = generate_cutoff_dates(start_date, end_date, horizon, step)
    horizon_length = steps_in_horizon(horizon, dataset.freq)

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
        # TODO: validate forecast df with pandera schema
        forecast = forecast.fillna(0)
        forecast["cutoff_date"] = cutoff_date
        forecasts.append(forecast)

    crossval_df = pd.concat(forecasts)
    return crossval_df
