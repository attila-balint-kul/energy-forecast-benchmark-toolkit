import pandas as pd
import pytest
from starlette.testclient import TestClient

from enfobench.dataset import Dataset
from enfobench.evaluation import (
    ForecastClient,
    cross_validate,
)
from enfobench.evaluation.evaluate import _compute_metric, _compute_metrics, evaluate_metrics
from enfobench.evaluation.metrics import mean_absolute_error, mean_squared_error
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import generate_cutoff_dates


@pytest.mark.parametrize(
    "horizon,step",
    [
        ("1 day", "1 day"),
        ("1 day", "1 hour"),
        ("1 day", "15 minutes"),
        ("7 days", "1 day"),
        ("7 days", "1 hour"),
        ("7 days", "15 minutes"),
    ],
)
def test_generate_cutoff_dates(horizon, step):
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2020-01-10")
    horizon = pd.Timedelta(horizon)
    step = pd.Timedelta(step)

    cutoff_dates = generate_cutoff_dates(start_date, end_date, horizon, step)

    assert all(cutoff_date >= start_date for cutoff_date in cutoff_dates)
    assert all(cutoff_date + horizon <= end_date for cutoff_date in cutoff_dates)
    assert all(
        cutoff - previous_cutoff == step
        for cutoff, previous_cutoff in zip(cutoff_dates[1:], cutoff_dates[:-1], strict=True)
    )


def test_generate_cutoff_dates_raises_with_empty_time_series():
    with pytest.raises(ValueError):
        generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-01"),
            horizon=pd.Timedelta("1 day"),
            step=pd.Timedelta("1 day"),
        )


def test_generate_cutoff_dates_raises_with_longer_horizon():
    with pytest.raises(ValueError):
        generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-01"),
            horizon=pd.Timedelta("7 days"),
            step=pd.Timedelta("1 day"),
        )


def test_generate_cutoff_dates_raises_with_negative_horizon():
    with pytest.raises(ValueError):
        generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-05"),
            horizon=pd.Timedelta("-1 day"),
            step=pd.Timedelta("1 day"),
        )


def test_generate_cutoff_dates_raises_with_negative_step():
    with pytest.raises(ValueError):
        generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-05"),
            horizon=pd.Timedelta("1 day"),
            step=pd.Timedelta("-1 day"),
        )


def test_cross_validate_univariate_locally(model, target):
    forecasts = cross_validate(
        model=model,
        dataset=Dataset(target),
        start_date=pd.Timestamp("2020-01-10"),
        end_date=pd.Timestamp("2020-01-21"),
        horizon=pd.Timedelta("7 days"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns


def test_cross_validate_univariate_via_server(model, target):
    app = server_factory(model=model)
    test_client = TestClient(app)
    model = ForecastClient(client=test_client)

    forecasts = cross_validate(
        model=model,
        dataset=Dataset(target),
        start_date=pd.Timestamp("2020-01-10"),
        end_date=pd.Timestamp("2020-01-21"),
        horizon=pd.Timedelta("7 days"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns


def test_cross_validate_multivariate_locally(model, target, covariates, external_forecasts):
    forecasts = cross_validate(
        model=model,
        dataset=Dataset(target, covariates, external_forecasts),
        start_date=pd.Timestamp("2020-01-10"),
        end_date=pd.Timestamp("2020-01-21"),
        horizon=pd.Timedelta("7 days"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns


def test_cross_validate_multivariate_via_server(model, target, covariates, external_forecasts):
    app = server_factory(model=model)
    test_client = TestClient(app)
    model = ForecastClient(client=test_client)

    forecasts = cross_validate(
        model=model,
        dataset=Dataset(target, covariates, external_forecasts),
        start_date=pd.Timestamp("2020-01-10"),
        end_date=pd.Timestamp("2020-01-21"),
        horizon=pd.Timedelta("7 days"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns


def test_compute_metric(clean_forecasts):
    metric_value = _compute_metric(clean_forecasts, mean_absolute_error)

    assert 0 < metric_value < 1


def test_compute_metrics(clean_forecasts):
    metric_values = _compute_metrics(clean_forecasts, {"MAE": mean_absolute_error, "MSE": mean_squared_error})

    assert isinstance(metric_values, dict)
    assert 0 < metric_values["MAE"] < 1
    assert 0 < metric_values["MSE"] < 1


def test_compute_metric_on_forecast_with_missing_values_raises_error(forecasts_with_missing_values):
    with pytest.raises(ValueError):
        _compute_metric(forecasts_with_missing_values, mean_absolute_error)


def test_compute_metrics_on_forecast_with_missing_values_raises_error(forecasts_with_missing_values):
    with pytest.raises(ValueError):
        _compute_metrics(forecasts_with_missing_values, {"MAE": mean_absolute_error, "MSE": mean_squared_error})


def test_evaluate_metrics_on_clean_forecasts(clean_forecasts):
    metrics = evaluate_metrics(clean_forecasts, {"MAE": mean_absolute_error, "MSE": mean_squared_error})

    assert isinstance(metrics, pd.DataFrame)
    assert "MAE" in metrics.columns
    assert "MSE" in metrics.columns
    assert "weight" in metrics.columns
    assert len(metrics) == 1
    assert 0 < metrics["MAE"].iloc[0] < 1
    assert 0 < metrics["MSE"].iloc[0] < 1
    assert metrics["weight"].iloc[0] == 1


def test_evaluate_metrics_on_forecasts_with_missing_values(forecasts_with_missing_values):
    metrics = evaluate_metrics(forecasts_with_missing_values, {"MAE": mean_absolute_error, "MSE": mean_squared_error})

    assert isinstance(metrics, pd.DataFrame)
    assert "MAE" in metrics.columns
    assert "MSE" in metrics.columns
    assert "weight" in metrics.columns
    assert len(metrics) == 1
    assert 0 < metrics["MAE"].iloc[0] < 1
    assert 0 < metrics["MSE"].iloc[0] < 1
    assert pytest.approx(metrics["weight"].iloc[0], 0.1) == 1 - 0.3


def test_evaluate_metrics_on_clean_forecasts_grouped_by(clean_forecasts):
    metrics = evaluate_metrics(
        clean_forecasts,
        {"MAE": mean_absolute_error, "MSE": mean_squared_error},
        groupby="cutoff_date",
    )

    grouped_values = clean_forecasts["cutoff_date"].unique()

    assert isinstance(metrics, pd.DataFrame)
    assert "MAE" in metrics.columns
    assert "MSE" in metrics.columns
    assert "weight" in metrics.columns
    assert "cutoff_date" in metrics.columns
    assert len(metrics) == len(grouped_values)
    assert all(0 < metric < 1 for metric in metrics["MAE"])
    assert all(0 < metric < 1 for metric in metrics["MSE"])
    assert all(metric == 1 for metric in metrics["weight"])


def test_evaluate_metrics_on_forecasts_with_missing_values_grouped_by(forecasts_with_missing_values):
    metrics = evaluate_metrics(
        forecasts_with_missing_values,
        {"MAE": mean_absolute_error, "MSE": mean_squared_error},
        groupby="cutoff_date",
    )

    grouped_values = forecasts_with_missing_values["cutoff_date"].unique()

    assert isinstance(metrics, pd.DataFrame)
    assert "MAE" in metrics.columns
    assert "MSE" in metrics.columns
    assert "weight" in metrics.columns
    assert "cutoff_date" in metrics.columns
    assert len(metrics) == len(grouped_values)
    assert all(0 < metric < 1 for metric in metrics["MAE"])
    assert all(0 < metric < 1 for metric in metrics["MSE"])
    assert all(0 <= metric <= 1 for metric in metrics["weight"])
