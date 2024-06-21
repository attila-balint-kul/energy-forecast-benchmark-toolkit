import pandas as pd
import pytest
from starlette.testclient import TestClient

from enfobench.evaluation import (
    ForecastClient,
    cross_validate,
)
from enfobench.evaluation.evaluate import _compute_metric, _compute_metrics, evaluate_metrics
from enfobench.evaluation.metrics import mean_absolute_error, mean_squared_error
from enfobench.evaluation.server import server_factory


def test_cross_validate_univariate_locally(helpers, model):
    dataset = helpers.generate_univariate_dataset(start="2020-01-01", end="2021-01-01", freq="30T")

    start_date = pd.Timestamp("2020-02-01 10:00:00")
    end_date = pd.Timestamp("2020-04-01")
    horizon = pd.Timedelta("38 hours")
    step = pd.Timedelta("1 day")

    forecasts = cross_validate(model, dataset, start_date=start_date, end_date=end_date, horizon=horizon, step=step)

    assert isinstance(forecasts, pd.DataFrame)
    assert "y" in forecasts.columns
    assert "yhat" in forecasts.columns

    assert "timestamp" in forecasts.columns
    assert (forecasts.timestamp >= start_date).all()
    assert forecasts.timestamp.iloc[0] == start_date

    assert "cutoff_date" in forecasts.columns
    assert (start_date.time() == forecasts.cutoff_date.dt.time).all()

    for cutoff_date, forecast in forecasts.groupby("cutoff_date"):
        assert len(forecast) == 38 * 2  # 38 hours with half-hour series
        assert forecast.timestamp.iloc[0] == cutoff_date
        assert (forecast.timestamp >= cutoff_date).all()

    assert list(forecasts.cutoff_date.unique()) == list(
        pd.date_range(start="2020-02-01 10:00:00", end="2020-03-30 10:00:00", freq="1D", inclusive="both")
    )


def test_cross_validate_univariate_via_server(helpers, model):
    dataset = helpers.generate_univariate_dataset(start="2020-01-01", end="2020-03-01", freq="30T")

    app = server_factory(model=model)
    test_client = TestClient(app)
    model = ForecastClient(client=test_client)

    forecasts = cross_validate(
        model=model,
        dataset=dataset,
        start_date=pd.Timestamp("2020-02-01 10:00:00"),
        end_date=pd.Timestamp("2020-03-01"),
        horizon=pd.Timedelta("38 hours"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "y" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns
    assert list(forecasts.cutoff_date.unique()) == list(
        pd.date_range(start="2020-02-01 10:00:00", end="2020-02-28 10:00:00", freq="1D", inclusive="both")
    )


def test_cross_validate_multivariate_locally(helpers, model):
    dataset = helpers.generate_multivariate_dataset(
        columns=["x", "z"], start="2020-01-01", end="2020-03-01", freq="30T"
    )

    forecasts = cross_validate(
        model=model,
        dataset=dataset,
        start_date=pd.Timestamp("2020-02-01 10:00:00"),
        end_date=pd.Timestamp("2020-03-01"),
        horizon=pd.Timedelta("38 hours"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "y" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns
    assert list(forecasts.cutoff_date.unique()) == list(
        pd.date_range(start="2020-02-01 10:00:00", end="2020-02-28 10:00:00", freq="1D", inclusive="both")
    )


def test_cross_validate_multivariate_via_server(helpers, model):
    dataset = helpers.generate_multivariate_dataset(
        columns=["x", "z"], start="2020-01-01", end="2020-03-01", freq="30T"
    )

    app = server_factory(model=model)
    test_client = TestClient(app)
    model = ForecastClient(client=test_client)

    forecasts = cross_validate(
        model=model,
        dataset=dataset,
        start_date=pd.Timestamp("2020-02-01 10:00:00"),
        end_date=pd.Timestamp("2020-03-01"),
        horizon=pd.Timedelta("38 hours"),
        step=pd.Timedelta("1 day"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "y" in forecasts.columns
    assert "yhat" in forecasts.columns
    assert "cutoff_date" in forecasts.columns
    assert list(forecasts.cutoff_date.unique()) == list(
        pd.date_range(start="2020-02-01 10:00:00", end="2020-02-28 10:00:00", freq="1D", inclusive="both")
    )


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


if __name__ == "__main__":
    pytest.main()
