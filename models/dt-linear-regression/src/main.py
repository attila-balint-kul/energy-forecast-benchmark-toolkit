import os

import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.linear_model import LinearRegression

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration
from typing import Literal


class DartsLinearRegressionModel:
    def __init__(self, seasonality: str, model_type: Literal['DirectMultiModel', 'DirectMultiOutput', 'Recursive']):
        self.seasonality = seasonality.upper()
        self.model_type = model_type

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.LinearRegression.{self.model_type}.{self.seasonality}",
            authors=[
                AuthorInfo(name="Mohamad Khalil", email="coo17619@newcastle.ac.uk"),
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.point,
            params={
                "model_type": self.model_type,
                "seasonality": self.seasonality,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        # Create model
        periods = periods_in_duration(history.index, duration=self.seasonality)
        if self.model_type == 'Recursive':
            model = RegressionModel(
                model=LinearRegression(),
                lags=list(range(-periods, 0)),
                output_chunk_length=1,
                multi_models=False,
            )
        elif self.model_type == 'DirectMultiOutput':
            model = RegressionModel(
                model=LinearRegression(),
                lags=list(range(-periods, 0)),
                output_chunk_length=horizon,
                multi_models=False,
            )
        elif self.model_type == 'DirectMultiModel':
            model = RegressionModel(
                model=LinearRegression(),
                lags=list(range(-periods, 0)),
                output_chunk_length=horizon,
                multi_models=True,
            )
        else:
            msg = f"Unknown model type: {self.model_type}"
            raise ValueError(msg)

        # Fit model
        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        # Postprocess forecast
        forecast = pred.pd_dataframe().rename(columns={"y": "yhat"}).fillna(history.y.mean())
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")
model_type = os.getenv("ENFOBENCH_MODEL_TYPE")

# Instantiate your model
model = DartsLinearRegressionModel(seasonality=seasonality, model_type=model_type)

# Create a forecast server by passing in your model
app = server_factory(model)
