from random import randrange

import pandas as pd

from enfobench.evaluation.protocols import Dataset


def random_date(start: pd.Timestamp, end: pd.Timestamp, resolution: int = 1) -> pd.Timestamp:
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = int(delta.total_seconds())
    random_second = randrange(0, int_delta, resolution)
    return start + pd.Timedelta(seconds=random_second)


def test_univariate_second(target):
    ds = Dataset(target=target)

    cutoff_date = random_date(ds.start_date, ds.end_date)
    print(cutoff_date)
    history = ds.get_history(cutoff_date)
    assert (history.ds <= cutoff_date).all()


def test_univariate_minute(target):
    ds = Dataset(target=target)
    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=60)
    print(cutoff_date)
    history = ds.get_history(cutoff_date)
    assert (history.ds <= cutoff_date).all()


def test_univariate_quarter(target):
    ds = Dataset(target=target)
    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=900)
    print(cutoff_date)
    history = ds.get_history(cutoff_date)
    assert (history.ds <= cutoff_date).all()


def test_univariate_hour(target):
    ds = Dataset(target=target)
    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=3600)
    print(cutoff_date)
    history = ds.get_history(cutoff_date)
    assert (history.ds <= cutoff_date).all()


def test_multivariate_second(target, covariates):
    ds = Dataset(target=target, covariates=covariates)

    cutoff_date = random_date(ds.start_date, ds.end_date)
    print(cutoff_date)
    past_cov = ds.get_past_covariates(cutoff_date)
    assert (past_cov.ds <= cutoff_date).all()


def test_multivariate_minute(target, covariates):
    ds = Dataset(target=target, covariates=covariates)

    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=60)
    print(cutoff_date)
    past_cov = ds.get_past_covariates(cutoff_date)
    assert (past_cov.ds <= cutoff_date).all()


def test_multivariate_quarter(target, covariates):
    ds = Dataset(target=target, covariates=covariates)

    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=900)
    print(cutoff_date)
    past_cov = ds.get_past_covariates(cutoff_date)
    assert (past_cov.ds <= cutoff_date).all()


def test_multivariate_hour(target, covariates):
    ds = Dataset(target=target, covariates=covariates)

    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=3600)
    print(cutoff_date)
    past_cov = ds.get_past_covariates(cutoff_date)
    assert (past_cov.ds <= cutoff_date).all()


def test_external_forecasts_second(target, covariates, external_forecasts):
    ds = Dataset(target=target, covariates=covariates, external_forecasts=external_forecasts)

    cutoff_date = random_date(ds.start_date, ds.end_date)
    print(cutoff_date)
    future_cov = ds.get_future_covariates(cutoff_date)
    assert (future_cov.cutoff_date <= cutoff_date).all()
    assert (future_cov.ds > cutoff_date).all()


def test_external_forecasts_minute(target, covariates, external_forecasts):
    ds = Dataset(target=target, covariates=covariates, external_forecasts=external_forecasts)

    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=60)
    print(cutoff_date)
    future_cov = ds.get_future_covariates(cutoff_date)
    assert (future_cov.cutoff_date <= cutoff_date).all()
    assert (future_cov.ds > cutoff_date).all()


def test_external_forecasts_quarter(target, covariates, external_forecasts):
    ds = Dataset(target=target, covariates=covariates, external_forecasts=external_forecasts)

    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=900)
    print(cutoff_date)
    future_cov = ds.get_future_covariates(cutoff_date)
    assert (future_cov.cutoff_date <= cutoff_date).all()
    assert (future_cov.ds > cutoff_date).all()


def test_external_forecasts_hour(target, covariates, external_forecasts):
    ds = Dataset(target=target, covariates=covariates, external_forecasts=external_forecasts)

    cutoff_date = random_date(ds.start_date, ds.end_date, resolution=3600)
    print(cutoff_date)
    future_cov = ds.get_future_covariates(cutoff_date)
    assert (future_cov.cutoff_date <= cutoff_date).all()
    assert (future_cov.ds > cutoff_date).all()
