import numpy as np
import pandas as pd
import pytest

from enfobench.evaluation import ForecasterType, ModelInfo
from enfobench.evaluation.utils import create_forecast_index


class TestModel:
    def __init__(self, param1: int):
        self.param1 = param1

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="TestModel",
            type=ForecasterType.point,
            params={
                "param1": 1,
            },
        )

    def forecast(
        self,
        horizon: int,
        history,
        past_covariates=None,
        future_covariates=None,
        level=None,
        **kwargs,
    ):
        index = create_forecast_index(history, horizon)
        return pd.DataFrame(
            data={
                "ds": index,
                "yhat": np.full(horizon, fill_value=history["y"].mean()) + self.param1,
            }
        )


@pytest.fixture(scope="function")
def model():
    return TestModel(1)


@pytest.fixture(scope="session")
def target() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", "2020-02-01", freq="30T")
    y = pd.Series(np.random.random(len(index)), index=index)
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
        index = pd.date_range(
            cutoff_date + pd.Timedelta(hours=1), cutoff_date + pd.Timedelta(days=7), freq="H"
        )
        forecast = pd.DataFrame(
            data={
                "ds": index,
                "forecast_1": np.random.random(len(index)),
                "forecast_2": np.random.random(len(index)),
            },
        )
        forecast["cutoff_date"] = cutoff_date
        forecasts.append(forecast)

    return pd.concat(forecasts, ignore_index=True)
