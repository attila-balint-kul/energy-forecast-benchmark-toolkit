import os

import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.baselines import NaiveMovingAverage

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration


class DartsNaiveMovingAverageModel:
    def __init__(self, history: str):
        self.history = history.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.NaiveMovingAverage.{self.history}",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.point,
            params={
                "history": self.history,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        # Create model
        periods = periods_in_duration(history.index, duration=self.history)
        model = NaiveMovingAverage(input_chunk_length=periods)

        # Fit model
        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        # Postprocess forecast
        forecast = pred.pd_dataframe().rename(columns={"y": "yhat"}).fillna(history.y.mean())
        return forecast


# Load parameters
history = os.getenv("ENFOBENCH_MODEL_HISTORY", "1D")

# Instantiate your model
model = DartsNaiveMovingAverageModel(history)

# Create a forecast server by passing in your model
app = server_factory(model)
