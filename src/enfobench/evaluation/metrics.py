from typing import Callable

import numpy as np
import pandas as pd
from numpy import ndarray


def check_consistent_length(*arrays: ndarray) -> None:
    """Check that all arrays have consistent length.

    Checks whether all input arrays have the same length.

    Parameters
    ----------
    *arrays : list or tuple of input arrays.
        Objects that will be checked for consistent length.
    """
    if any([X.ndim != 1 for X in arrays]):
        raise ValueError("Found multi dimensional array in inputs.")

    lengths = [len(X) for X in arrays]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: {lengths}"
        )


def check_has_no_nan(*arrays: ndarray) -> None:
    """Check that all arrays have no NaNs.

    Parameters
    ----------
    *arrays : list or tuple of input arrays.
        Objects that will be checked for NaNs.
    """
    for X in arrays:
        if np.isnan(X).any():
            raise ValueError(
                f"Found NaNs in input variables: {X}"
            )


def mean_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean absolute error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_consistent_length(y_true, y_pred)
    check_has_no_nan(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_bias_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean bias error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_consistent_length(y_true, y_pred)
    check_has_no_nan(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean squared error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_consistent_length(y_true, y_pred)
    check_has_no_nan(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Root mean squared error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_consistent_length(y_true, y_pred)
    check_has_no_nan(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: ndarray, y_pred: ndarray) -> float:
    """Mean absolute percentage error regression loss.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    """
    check_consistent_length(y_true, y_pred)
    check_has_no_nan(y_true, y_pred)
    if np.any(y_true == 0):
        raise ValueError("Found zero in true values. MAPE is undefined.")
    return float(100. * np.mean(np.abs((y_true - y_pred) / y_true)))


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
    _nonempty_df = forecast.dropna(subset=['y'])
    metric_value = metric(_nonempty_df.y, _nonempty_df.yhat)
    return metric_value


def evaluate_metrics_on_forecast(forecast: pd.DataFrame, metrics: dict[str, Callable]) -> dict[
    str, float]:
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
        for cutoff, group_df in forecasts.groupby('cutoff')
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
    return metrics_df


def evaluate_metrics_on_forecasts(forecasts: pd.DataFrame,
                                  metrics: dict[str, Callable]) -> pd.DataFrame:
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
        evaluate_metric_on_forecasts(forecasts, metric_func).rename(columns={'value': metric_name})
        for metric_name, metric_func in metrics.items()
    ]
    metrics_df = pd.concat(metric_dfs, axis=1)
    return metrics_df
