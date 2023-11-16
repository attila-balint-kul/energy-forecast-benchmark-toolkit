import numpy as np
import pandas as pd
import pytest

from enfobench import AuthorInfo, ForecasterType, ModelInfo
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


@pytest.fixture(scope="session")
def target() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", "2020-02-01", freq="30T")
    y = pd.DataFrame(
        index=index,
        data={
            "y": np.random.random(len(index)),
        },
    )
    return y


@pytest.fixture(scope="session")
def covariates() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", "2020-02-01", freq="H")
    df = pd.DataFrame(
        index=index,
        data={
            "cov_1": np.random.random(len(index)),
            "cov_2": np.random.random(len(index)),
        },
    )
    return df


@pytest.fixture(scope="session")
def external_forecasts() -> pd.DataFrame:
    cutoff_dates = pd.date_range("2020-01-01", "2020-02-01", freq="D")
    forecasts = []

    for cutoff_date in cutoff_dates:
        index = pd.date_range(cutoff_date + pd.Timedelta(hours=1), cutoff_date + pd.Timedelta(days=7), freq="H")
        forecast = pd.DataFrame(
            data={
                "timestamp": index,
                "forecast_1": np.random.random(len(index)),
                "forecast_2": np.random.random(len(index)),
            },
        )
        forecast["cutoff_date"] = cutoff_date
        forecasts.append(forecast)

    return pd.concat(forecasts, ignore_index=True)
