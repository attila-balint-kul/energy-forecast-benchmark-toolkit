from random import randrange

import numpy as np
import pandas as pd
import pytest

from enfobench import AuthorInfo, Dataset, ForecasterType, ModelInfo
from enfobench.evaluation.utils import create_forecast_index


class TestModel:
    def __init__(self, param1: int):
        self.param1 = param1

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="TestModel",
            authors=[
                AuthorInfo("Author 1", "author-1@institution.org"),
            ],
            type=ForecasterType.point,
            params={
                "param1": 1,
            },
        )

    def forecast(
        self,
        horizon: int,
        history,
        past_covariates=None,  # noqa: ARG002
        future_covariates=None,  # noqa: ARG002
        level=None,  # noqa: ARG002
        metadata=None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        index = create_forecast_index(history, horizon)
        y_hat = np.full(horizon, fill_value=history["y"].mean()) + self.param1
        return pd.DataFrame(
            index=index,
            data={
                "yhat": y_hat,
            },
        )


@pytest.fixture(scope="function")
def model():
    return TestModel(1)


class Helpers:
    @staticmethod
    def generate_target(**kwargs) -> pd.DataFrame:
        index = pd.date_range(**kwargs)
        target = pd.DataFrame(
            index=index,
            data={
                "y": np.random.random(len(index)),
            },
        )
        return target

    @staticmethod
    def generate_covariates(columns: list[str], **kwargs):
        index = pd.date_range(**kwargs)
        df = pd.DataFrame(
            index=index,
            data={c: np.random.random(len(index)) for c in columns},
        )
        return df

    @staticmethod
    def generate_future_covariates(columns: list[str], **kwargs):
        cutoff_dates = pd.date_range(**kwargs)
        forecasts = []

        for cutoff_date in cutoff_dates:
            index = pd.date_range(cutoff_date + pd.Timedelta(hours=1), cutoff_date + pd.Timedelta(days=7), freq="H")
            forecast = pd.DataFrame(
                data={"timestamp": index, **{c: np.random.random(len(index)) for c in columns}},
            )
            forecast["cutoff_date"] = cutoff_date
            forecasts.append(forecast)

        return pd.concat(forecasts, ignore_index=True)

    @staticmethod
    def generate_univariate_dataset(**kwargs):
        target = Helpers.generate_target(**kwargs)
        return Dataset(target)

    @staticmethod
    def generate_multivariate_dataset(columns: list[str], **kwargs):
        target = Helpers.generate_target(**kwargs)
        past_covariates = Helpers.generate_covariates(columns, **kwargs)
        return Dataset(
            target=target,
            past_covariates=past_covariates,
        )

    @staticmethod
    def random_date(start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> pd.Timestamp:
        """
        This function will return a random datetime between two datetime
        objects.
        """
        resolution_in_seconds = int(pd.Timedelta(resolution).total_seconds())
        delta = end - start
        int_delta = int(delta.total_seconds())
        random_second = randrange(0, int_delta, resolution_in_seconds)  # noqa: S311
        return start + pd.Timedelta(seconds=random_second)


@pytest.fixture(scope="session")
def helpers():
    return Helpers


@pytest.fixture(scope="session")
def clean_forecasts() -> pd.DataFrame:
    cutoff_dates = pd.date_range("2020-01-01", "2021-01-01", freq="D")

    forecasts = []
    for cutoff_date in cutoff_dates:
        index = pd.date_range(cutoff_date, cutoff_date + pd.Timedelta(hours=25), freq="H")
        forecast = pd.DataFrame(
            data={
                "timestamp": index,
                "yhat": np.random.random(len(index)),
                "y": np.random.random(len(index)),
            },
        )
        forecast["cutoff_date"] = cutoff_date
        forecasts.append(forecast)

    forecast_df = pd.concat(forecasts, ignore_index=True)
    assert not forecast_df.isna().any(axis=None)
    return forecast_df


@pytest.fixture(scope="session")
def forecasts_with_missing_values() -> pd.DataFrame:
    cutoff_dates = pd.date_range("2020-01-01", "2021-01-01", freq="D")

    forecasts = []
    for cutoff_date in cutoff_dates:
        index = pd.date_range(cutoff_date, cutoff_date + pd.Timedelta(hours=25), freq="H")
        forecast = pd.DataFrame(
            data={
                "timestamp": index,
                "yhat": np.random.random(len(index)),
                "y": np.random.random(len(index)),
            },
        )
        forecast["cutoff_date"] = cutoff_date
        forecasts.append(forecast)

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_df.loc[forecast_df["y"] <= 0.3, "y"] = np.nan
    assert forecast_df.isna().any(axis=None)
    return forecast_df
