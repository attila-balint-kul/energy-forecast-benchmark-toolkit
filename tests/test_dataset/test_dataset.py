import pandas as pd
import pytest

from enfobench.core import Dataset


@pytest.mark.parametrize("resolution", ["15T", "30T", "1H"])
def test_get_history(helpers, resolution):
    target = helpers.generate_target(start="2020-01-01", end="2021-01-01", freq=resolution)
    ds = Dataset(target)

    cutoff_date = helpers.random_date(ds.target_available_since, ds.target_available_until, resolution)

    history = ds.get_history(cutoff_date)

    assert history.index.name == "timestamp"
    assert isinstance(history.index, pd.DatetimeIndex)
    assert "y" in history.columns
    assert len(history.columns) == 1
    assert cutoff_date not in history.index
    assert (history.index < cutoff_date).all()


@pytest.mark.parametrize("resolution", ["15T", "30T", "1H"])
def test_get_past_covariates(helpers, resolution):
    target = helpers.generate_target(start="2020-01-01", end="2021-01-01", freq=resolution)
    past_covariates = helpers.generate_covariates(
        start="2020-01-01",
        end="2021-01-01",
        freq=resolution,
        columns=["a", "b"],
    )
    ds = Dataset(target, past_covariates)

    cutoff_date = helpers.random_date(ds.target_available_since, ds.target_available_until, resolution)

    past_cov = ds.get_past_covariates(cutoff_date)

    assert past_cov.index.name == "timestamp"
    assert isinstance(past_cov.index, pd.DatetimeIndex)
    assert "a" in past_cov.columns
    assert "b" in past_cov.columns
    assert cutoff_date in past_cov.index
    assert (past_cov.index <= cutoff_date).all()


@pytest.mark.parametrize("resolution", ["6H", "1D"])
def test_get_future_covariates(helpers, resolution):
    target = helpers.generate_target(start="2020-01-01", end="2021-01-01", freq="30T")
    future_covariates = helpers.generate_future_covariates(
        start="2020-01-01",
        end="2021-01-01",
        freq=resolution,
        columns=["a", "b"],
    )
    ds = Dataset(target, future_covariates=future_covariates)

    cutoff_date = helpers.random_date(ds.target_available_since, ds.target_available_until, resolution)

    future_cov = ds.get_future_covariates(cutoff_date)

    assert future_cov.index.name == "timestamp"
    assert isinstance(future_cov.index, pd.DatetimeIndex)
    assert "a" in future_cov.columns
    assert "b" in future_cov.columns
    assert (future_cov.cutoff_date <= cutoff_date).all()
    assert (future_cov.index > cutoff_date).all()


if __name__ == "__main__":
    pytest.main()
