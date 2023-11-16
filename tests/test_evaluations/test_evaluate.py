import pandas as pd
import pytest
from starlette.testclient import TestClient

from enfobench.dataset import Dataset
from enfobench.evaluation import (
    ForecastClient,
    cross_validate,
)
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
