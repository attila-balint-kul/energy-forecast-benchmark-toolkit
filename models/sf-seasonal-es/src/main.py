import os

import pandas as pd
from statsforecast.models import SeasonalExponentialSmoothing

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class SeasonalExponentialSmoothingModel:
    def __init__(self, seasonality: str, alpha: float):
        self.seasonality = seasonality.upper()

        if alpha < 0 or alpha > 1:
            msg = "Alpha parameter must be between 0 and 1"
            raise ValueError(msg)
        self._alpha = round(alpha, 3)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.SeasonalExponentialSmoothing.{self.seasonality}.A{self._alpha:.3f}",
            authors=[AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")],
            type=ForecasterType.point,
            params={
                "seasonality": self.seasonality,
                "alpha": self._alpha,
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
        model = SeasonalExponentialSmoothing(season_length=periods, alpha=self._alpha)

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Postprocess forecast
        forecast = (
            pd.DataFrame(
                index=index,
                data={
                    "yhat": pred["mean"],
                },
            )
            .rename(columns={"mean": "yhat"})
            .fillna(y.mean())
        )
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")
alpha = float(os.getenv("ENFOBENCH_MODEL_ALPHA"))

# Instantiate your model
model = SeasonalExponentialSmoothingModel(seasonality=seasonality, alpha=alpha)

# Create a forecast server by passing in your model
app = server_factory(model)
