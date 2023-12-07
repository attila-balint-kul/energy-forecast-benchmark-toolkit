from random import randrange

import pandas as pd
import pytest

from enfobench.dataset import Dataset


def random_date(start: pd.Timestamp, end: pd.Timestamp, resolution: int = 1) -> pd.Timestamp:
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = int(delta.total_seconds())
    random_second = randrange(0, int_delta, resolution)  # noqa: S311
    return start + pd.Timedelta(seconds=random_second)


@pytest.mark.parametrize("resolution", [1, 60, 900, 3600])
def test_get_history(target, resolution):
    ds = Dataset(target=target)
    cutoff_date = random_date(ds._first_available_target_date, ds._last_available_target_date, resolution=resolution)

    history = ds.get_history(cutoff_date)

    assert history.index.name == "timestamp"
    assert isinstance(history.index, pd.DatetimeIndex)
    assert "y" in history.columns
    assert len(history.columns) == 1
    assert (history.index <= cutoff_date).all()


@pytest.mark.parametrize("resolution", [1, 60, 900, 3600])
def test_get_past_covariates(target, covariates, resolution):
    ds = Dataset(target=target, past_covariates=covariates)
    cutoff_date = random_date(ds._first_available_target_date, ds._last_available_target_date, resolution=resolution)

    past_cov = ds.get_past_covariates(cutoff_date)

    assert past_cov.index.name == "timestamp"
    assert isinstance(past_cov.index, pd.DatetimeIndex)
    for col in covariates.columns:
        assert col in past_cov.columns
    assert (past_cov.index <= cutoff_date).all()


@pytest.mark.parametrize("resolution", [1, 60, 900, 3600])
def test_get_future_covariates(target, covariates, external_forecasts, resolution):
    ds = Dataset(target=target, past_covariates=covariates, future_covariates=external_forecasts)
    cutoff_date = random_date(ds._first_available_target_date, ds._last_available_target_date, resolution=resolution)

    future_cov = ds.get_future_covariates(cutoff_date)

    assert future_cov.index.name == "timestamp"
    assert isinstance(future_cov.index, pd.DatetimeIndex)
    for col in external_forecasts.columns:
        if col not in ["timestamp", "cutoff_date"]:
            assert col in future_cov.columns
    assert (future_cov.cutoff_date <= cutoff_date).all()
    assert (future_cov.index > cutoff_date).all()
