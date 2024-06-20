import os

import pandas as pd
from statsforecast.models import AutoCES

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class AutoCESModel:
    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()
        self._last_prediction = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.AutoCES.{self.seasonality}.RP7D",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.quantile,
            params={
                "seasonality": self.seasonality,
            },
        )

    def _forecast(self, y: pd.Series, level: list[int] | None = None) -> pd.DataFrame:
        periods = periods_in_duration(y.index, duration=self.seasonality)
        model = AutoCES(season_length=periods)

        # Make forecast for 7 days
        periods_in_7_days = periods_in_duration(y.index, duration="7D")
        pred = model.forecast(y=y.values, h=periods_in_7_days, level=level)

        # Create index for forecast
        index = create_forecast_index(history=y.to_frame("y"), horizon=periods_in_7_days)

        # Postprocess forecast
        self._last_prediction = pd.DataFrame(index=index, data=pred).rename(columns={"mean": "yhat"}).fillna(y.mean())
        return self._last_prediction

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        level: list[int] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        y = history.y.fillna(history.y.mean())

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        if self._last_prediction is None or not index.isin(self._last_prediction.index).all():
            self._forecast(y=y, level=level)

        forecast = self._last_prediction.loc[index]
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY", "1D")

# Instantiate your model
model = AutoCESModel(seasonality=seasonality)

# Create a forecast server by passing in your model
app = server_factory(model)
