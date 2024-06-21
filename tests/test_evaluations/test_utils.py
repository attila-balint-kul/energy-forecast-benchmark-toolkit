from datetime import timedelta

import pandas as pd
import pytest

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


def test_periods_in_duration_raises_with_wrong_types():
    target = pd.date_range(start="2020-01-01", end="2020-02-01", freq="15T")
    with pytest.raises(ValueError):
        utils.periods_in_duration(target, 3)


def test_periods_in_duration_raises_if_duration_is_not_multiples():
    target = pd.date_range(start="2020-01-01", end="2020-02-01", freq="15T")
    with pytest.raises(ValueError):
        utils.periods_in_duration(target, "4T")


@pytest.mark.parametrize(
    "start_date,end_date,horizon,step,expected",
    [
        ("2020-01-01", "2020-01-05", "2 days", "1 day", ["2020-01-01", "2020-01-02", "2020-01-03"]),
        ("2020-01-01", "2020-01-05", "1 day", "1 day", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    ],
)
def test_generate_cutoff_dates(start_date, end_date, horizon, step, expected):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    horizon = pd.Timedelta(horizon)
    step = pd.Timedelta(step)

    cutoff_dates = utils.generate_cutoff_dates(start_date, end_date, horizon, step)

    assert cutoff_dates == [pd.Timestamp(date) for date in expected]


def test_generate_cutoff_date_raises_with_longer_horizon():
    with pytest.raises(ValueError):
        utils.generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-05"),
            horizon=pd.Timedelta("7 days"),
            step=pd.Timedelta("1 day"),
        )


def test_generate_cutoff_date_raises_with_wrong_end_and_start_dates():
    with pytest.raises(ValueError):
        utils.generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-01"),
            horizon=pd.Timedelta("1 day"),
            step=pd.Timedelta("1 day"),
        )


def test_generate_cutoff_date_raises_with_negative_horizon():
    with pytest.raises(ValueError):
        utils.generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-05"),
            horizon=pd.Timedelta("-1 day"),
            step=pd.Timedelta("1 day"),
        )


def test_generate_cutoff_date_raises_with_negative_step():
    with pytest.raises(ValueError):
        utils.generate_cutoff_dates(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-05"),
            horizon=pd.Timedelta("1 day"),
            step=pd.Timedelta("-1 day"),
        )


@pytest.mark.parametrize(
    "freq,horizon",
    [
        ("1H", 24),
        ("1H", 38),
        ("30T", 48),
        ("30T", 72),
        ("15T", 96),
        ("15T", 144),
    ],
)
def test_create_forecast_index(helpers, freq, horizon):
    dataset = helpers.generate_univariate_dataset(start="2020-01-01", end="2020-01-31", freq=freq)

    cutoff_date = helpers.random_date(dataset.target_available_since, dataset.target_available_until, freq)
    history = dataset.get_history(cutoff_date)

    index = utils.create_forecast_index(history, horizon)

    assert isinstance(index, pd.DatetimeIndex)
    assert index.freq == history.index.freq
    assert len(index) == horizon
    assert index[0] == cutoff_date
    assert (index >= cutoff_date).all()
