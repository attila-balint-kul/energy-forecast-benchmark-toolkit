import os

import pandas as pd
from statsforecast.models import SeasonalNaive

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class SeasonalNaiveModel:
    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.SeasonalNaive.{self.seasonality}",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.quantile,
            params={
                "seasonality": self.seasonality,
            },
        )

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

        # Create model
        periods = periods_in_duration(y.index, duration=self.seasonality)
        model = SeasonalNaive(season_length=periods)

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, level=level, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Postprocess forecast
        forecast = pd.DataFrame(index=index, data=pred).rename(columns={"mean": "yhat"}).fillna(y.mean())
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")

# Instantiate your model
model = SeasonalNaiveModel(seasonality)

# Create a forecast server by passing in your model
app = server_factory(model)
