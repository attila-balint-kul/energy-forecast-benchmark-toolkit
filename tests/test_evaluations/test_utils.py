from datetime import timedelta

import pandas as pd
import pytest

import enfobench.dataset.utils
from enfobench.evaluation import utils


@pytest.mark.parametrize(
    "horizon, freq, expected",
    [
        ("1 day", "15T", 96),
        ("1 day", "1H", 24),
        ("7 days", "1H", 7 * 24),
        ("1D", "1D", 1),
        ("1H", "1H", 1),
    ],
)
def test_steps_in_horizon(horizon, freq, expected):
    assert utils.steps_in_horizon(pd.Timedelta(horizon), freq) == expected


def test_steps_in_horizon_raises_with_non_multiple_horizon():
    with pytest.raises(ValueError):
        utils.steps_in_horizon(pd.Timedelta("36 minutes"), "15T")


def test_create_forecast_index(target):
    history = target
    horizon = 96
    last_date = history.index[-1]

    index = utils.create_forecast_index(history, horizon)

    assert isinstance(index, pd.DatetimeIndex)
    assert index.freq == target.index.freq
    assert len(index) == horizon
    assert all(idx > last_date for idx in index)


def test_create_perfect_forecasts_from_covariates(covariates):
    forecasts = enfobench.dataset.utils.create_perfect_forecasts_from_covariates(
        covariates,
        horizon=pd.Timedelta("7 days"),
        step=pd.Timedelta("1D"),
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "timestamp" in forecasts.columns
    assert "cutoff_date" in forecasts.columns
    assert all(col in forecasts.columns for col in covariates.columns)


@pytest.mark.parametrize(
    "freq, duration, expected",
    [
        ("15T", "15T", 1),
        ("15T", "1H", 4),
        ("15T", "1D", 96),
        ("15T", "2D", 2 * 96),
        ("15T", "1W", 7 * 96),
        ("15T", pd.Timedelta("15T"), 1),
        ("15T", pd.Timedelta("1H"), 4),
        ("15T", pd.Timedelta("1D"), 96),
        ("15T", pd.Timedelta("2D"), 2 * 96),
        ("15T", pd.Timedelta("1W"), 7 * 96),
        ("15T", timedelta(minutes=15), 1),
        ("15T", timedelta(hours=1), 4),
        ("15T", timedelta(days=1), 96),
        ("15T", timedelta(days=2), 2 * 96),
        ("15T", timedelta(weeks=1), 7 * 96),
        ("30T", "30T", 1),
        ("30T", "1H", 2),
        ("30T", "1D", 48),
        ("30T", "2D", 2 * 48),
        ("30T", "1W", 7 * 48),
        ("30T", pd.Timedelta("30T"), 1),
        ("30T", pd.Timedelta("1H"), 2),
        ("30T", pd.Timedelta("1D"), 48),
        ("30T", pd.Timedelta("2D"), 2 * 48),
        ("30T", pd.Timedelta("1W"), 7 * 48),
        ("30T", timedelta(minutes=30), 1),
        ("30T", timedelta(hours=1), 2),
        ("30T", timedelta(days=1), 48),
        ("30T", timedelta(days=2), 2 * 48),
        ("30T", timedelta(weeks=1), 7 * 48),
        ("1H", "1H", 1),
        ("1H", "1D", 24),
        ("1H", "2D", 2 * 24),
        ("1H", "1W", 7 * 24),
        ("1H", pd.Timedelta("1H"), 1),
        ("1H", pd.Timedelta("1D"), 24),
        ("1H", pd.Timedelta("2D"), 2 * 24),
        ("1H", pd.Timedelta("1W"), 7 * 24),
        ("1H", timedelta(hours=1), 1),
        ("1H", timedelta(days=1), 24),
        ("1H", timedelta(days=2), 2 * 24),
        ("1H", timedelta(weeks=1), 7 * 24),
    ],
)
def test_periods_in_duration(freq, duration, expected):
    target = pd.date_range(start="2020-01-01", end="2020-02-01", freq=freq)

    assert utils.periods_in_duration(target, duration) == expected
