from typing import Callable, Dict

import pandas as pd


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
        for cutoff, group_df in forecasts.groupby("cutoff")
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
