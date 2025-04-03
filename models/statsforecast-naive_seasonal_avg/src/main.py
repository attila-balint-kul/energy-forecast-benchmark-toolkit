import os

import pandas as pd
from statsforecast.models import SeasonalWindowAverage

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class SeasonalWindowAverageModel:
    def __init__(self, seasonality: str, window_size: int):
        self.seasonality = seasonality.upper()
        self.window_size = window_size

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast-SeasonalWindowAverage-{self.seasonality}-W{self.window_size}",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.point,
            params={
                "seasonality": self.seasonality,
                "window_size": self.window_size,
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
        model = SeasonalWindowAverage(season_length=periods, window_size=self.window_size)

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Postprocess forecast
        forecast = pd.DataFrame(
            index=index,
            data={
                "yhat": pred["mean"],
            },
        ).fillna(y.mean())
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY", "1D")
window_size = int(os.getenv("ENFOBENCH_MODEL_WINDOW_SIZE", "28"))

# Instantiate your model
model = SeasonalWindowAverageModel(seasonality, window_size)

# Create a forecast server by passing in your model
app = server_factory(model)
