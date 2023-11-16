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
